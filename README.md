# Pajigsaw network

The training code is adapted from https://github.com/microsoft/Swin-Transformer

## Dataset generation

## Training

## Evaluation
Command to evaluation:
>```CUDA_VISIBLE_DEVICES=0 python3.8 -m torch.distributed.run --standalone --nproc_per_node 1 evaluation.py --cfg configs/pajigsaw/eval_erosion7_4bin_patch8_64.yaml --data-path /data1/mvu/datasets/Puzzle --batch-size 128 --pretrained output/div2k_erosion7_4bin_patch8_64/default/best_model.pth --opts DATA.EROSION_RATIO 0.07```
