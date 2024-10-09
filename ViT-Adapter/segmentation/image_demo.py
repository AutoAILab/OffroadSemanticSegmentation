# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mmseg.core import get_classes
import cv2
import os.path as osp
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Image file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    for img_path in os.listdir(args.img):
        load_img_path = os.path.join(args.img, img_path)
         # test a single image
        result = inference_segmentor(model, load_img_path)
        # show the results
        if hasattr(model, 'module'):
            model = model.module

        img = model.show_result(load_img_path, result,
                                palette=get_palette(args.palette),
                                show=False, opacity=args.opacity)
        mmcv.mkdir_or_exist(args.out)
        out_path = osp.join(args.out, osp.basename(load_img_path))
        # cv2.imwrite(out_path, img)
        color_palettes = [[108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
                  [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
                  [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
                  [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]
        class_names = ("dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
                    "object", "asphalt", "building", "log", "person", "fence", "bush", 
                    "concrete", "barrier", "puddle", "mud", "rubble")
        patches = []
        
        for color, name in zip(color_palettes, class_names):
            rgb_color = [c / 255 for c in color]  # Convert RGB values to range [0, 1]
            patch = mpatches.Patch(color=rgb_color, label=name)
            patches.append(patch)
        
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig(out_path)
        print(f"Result is save at {out_path}")
        plt.close()

if __name__ == '__main__':
    main()