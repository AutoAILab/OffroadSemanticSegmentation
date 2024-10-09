import argparse
import os.path as osp
import numpy as np
import mmcv
# import cv2
from PIL import Image
import os
import shutil

rellis_dir = "../../../../data/rellis/Rellis3D/"
annotation_folder = "annotations/"

path_train_annotation = '../../../../data/rellis/Rellis3D/annotations/training'
path_val_annotation = '../../../../data/rellis/Rellis3D/annotations/validation'
path_test_annotation = '../../../../data/rellis/Rellis3D/annotations/test'

path_train_image = '../../../../data/rellis/Rellis3D/images/training'
path_val_image = '../../../../data/rellis/Rellis3D/images/validation'
path_test_image = '../../../../data/rellis/Rellis3D/images/test'

os.makedirs(path_train_annotation, exist_ok=True)
os.makedirs(path_val_annotation, exist_ok=True)
os.makedirs(path_test_annotation, exist_ok=True)
os.makedirs(path_train_image, exist_ok=True)
os.makedirs(path_val_image, exist_ok=True)
os.makedirs(path_test_image, exist_ok=True)


IDs =    [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 29, 30, 31, 32, 33, 34]
Groups = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 1, 16, 4, 17, 18]



ID_seq = {}
ID_group = {}
for n, label in enumerate(IDs):
    ID_seq[label] = n
    ID_group[label] = Groups[n]

# 0 -- Background: void, sky, sign
# 1 -- Level1 (smooth) - Navigable: concrete, asphalt
# 2 -- Level2 (rough) - Navigable: gravel, grass, dirt, sand, mulch
# 3 -- Level3 (bumpy) - Navigable: Rock, Rock-bed
# 4 -- Non-Navigable (forbidden) - water
# 5 -- Obstacle - tree, pole, vehicle, container/generic-object, building, log, 
#                 bicycle(could be removed), person, fence, bush, picnic-table, bridge,

CLASSES = ("void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
            "object", "asphalt", "building", "log", "person", "fence", "bush", 
            "concrete", "barrier", "puddle", "mud", "rubble")

PALETTE = [[0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
            [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]


def raw_to_seq(seg):
    h, w = seg.shape
    out1 = np.zeros((h, w))
    out2 = np.zeros((h, w))
    for i in IDs:
        out1[seg==i] = ID_seq[i]
        out2[seg==i] = ID_group[i]

    return out1, out2

with open(osp.join(rellis_dir, 'train.lst'), 'r') as r:
    i = 0
    for l in r:
        print("train: {}".format(i))
        image, annotation = l.split(' ')
        annotation = annotation.strip()
        image = image.strip()

        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rellis_dir + annotation)
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        out1, out2 = raw_to_seq(gt_semantic_seg)
        
        annotation_save_path = os.path.join(path_train_annotation, annotation.split("/")[-1])
        image_curr_path = os.path.join(rellis_dir, image) 

        shutil.copy(image_curr_path, path_train_image)
        mmcv.imwrite(out2, annotation_save_path)

        i += 1

with open(osp.join(rellis_dir, 'val.lst'), 'r') as r:
    i = 0
    for l in r:
        print("val: {}".format(i))
        image, annotation = l.split(' ')
        annotation = annotation.strip()
        image = image.strip()

        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rellis_dir + annotation)
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        out1, out2 = raw_to_seq(gt_semantic_seg)
        
        annotation_save_path = os.path.join(path_val_annotation, annotation.split("/")[-1])
        image_curr_path = os.path.join(rellis_dir, image) 

        shutil.copy(image_curr_path, path_val_image)
        mmcv.imwrite(out2, annotation_save_path)

        i += 1

with open(osp.join(rellis_dir, 'test.lst'), 'r') as r:
    i = 0
    for l in r:
        print("test: {}".format(i))
        image, annotation = l.split(' ')
        annotation = annotation.strip()
        image = image.strip()

        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        img_bytes = file_client.get(rellis_dir + annotation)
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        out1, out2 = raw_to_seq(gt_semantic_seg)
        
        annotation_save_path = os.path.join(path_test_annotation, annotation.split("/")[-1])
        image_curr_path = os.path.join(rellis_dir, image) 

        shutil.copy(image_curr_path, path_test_image)
        mmcv.imwrite(out2, annotation_save_path)

        i += 1



print("successful")