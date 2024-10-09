import torch
import torch.nn as nn
from datasets.rellis3d import Rellis3D

#  make dataloader on Rellis3D dataset
def make_dataloader():
    dataset = Rellis3D(split_type='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader

dl = make_dataloader()
print(next(dl))