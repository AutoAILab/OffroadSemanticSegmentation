from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
import torch

class Self_Attention(nn.Module):
        def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
            super(Self_Attention, self).__init__()
            self.heads = heads
            self.norm = nn.LayerNorm(dim)
            self.inner_dim = dim_head * heads
            self.dim_head = dim_head
            self.scale = dim_head ** -0.5
            self.attend = nn.Softmax(dim = -1)
            self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False)
            
            self.dropout = nn.Dropout(dropout)
            self.to_out = nn.Sequential(
                nn.Linear(self.inner_dim, dim),
            )
        
        def forward(self, x):
            x = self.norm(x)
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b q (h d) -> b h q d', h = self.heads), qkv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            
            out = self.to_out(out)
            return out

class FPN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Self_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FPN(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class MaskVit(nn.Module):
    def __init__(self, image_size = (420, 420), num_channels = 3, patch_height = 420, patch_width = 420, num_patches = 100, num_classes = 150, transformer_depth = 8, pool = 'mean'):
        super(MaskVit, self).__init__() 

        self.patch_dim = patch_height * patch_width * num_channels # h*w*c 
        self.depth = transformer_depth
        self.num_patches = num_patches # num_masks q = m  
        self.queries = self.num_patches
        self.dim = 768 
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))  
        self.pos_embeding = nn.Parameter(torch.randn(1, self.queries, self.dim))  

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b m c h w -> b m (c h w)'),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.Transformer = Transformer(dim = self.dim, depth = 12, heads = 8, dim_head = 128, mlp_dim = 1024, dropout = 0.1)
        self.to_latent = nn.Identity()

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.Linear(self.dim, num_classes),
            nn.Softmax(dim = -1))
    
    def forward(self, x):
        x = self.to_patch_embedding(x) # (b, q, dim) or (b, m, dim)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = x.shape[0]) 
        # x = torch.cat((cls_tokens, x), dim = 1) # (b, q+1, dim)

        x += self.pos_embeding # (b, q+1, dim)
        x = self.Transformer(x)

        # x = x.mean(dim = -1) if self.pool == 'mean' else x[:, 0]
        # print(x.shape)
        # x = self.to_latent(x)

        x = self.mlp_head(x)
                
        return x