#!/bin/bash
export PYTHONPATH=/project/bli4/autoai/nobel/OffRoadSemanticSegmentation/OffRoadSemanticSegmentation/benchmarks/RELLIS-3D/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/:$PYTHONPATH
python tools/test.py --cfg experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-3_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET test.lst \
                     OUTPUT_DIR /project/bli4/autoai/nobel/OffRoadSemanticSegmentation/OffRoadSemanticSegmentation/benchmarks/RELLIS-3D/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/prediction \
                     TEST.MODEL_FILE /project/bli4/autoai/nobel/OffRoadSemanticSegmentation/OffRoadSemanticSegmentation/benchmarks/RELLIS-3D/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/pretrained_models/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/best.pth

