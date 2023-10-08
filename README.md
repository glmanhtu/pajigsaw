# Pajigsaw network

The training code is adapted from https://github.com/microsoft/Swin-Transformer

## Dataset generation

## Training

## Evaluation
Command to evaluation:
>```CUDA_VISIBLE_DEVICES=0 python3.8 -m torch.distributed.run --standalone --nproc_per_node 1 evaluation.py --cfg configs/pajigsaw/eval_erosion7_4bin_patch8_64.yaml --data-path /data1/mvu/datasets/Puzzle --batch-size 128 --pretrained output/div2k_erosion7_4bin_patch8_64/default/best_model.pth --opts DATA.EROSION_RATIO 0.07```

## Hisfrag20
>```PYTHONPATH=$PYTHONPATH:. python3.8 scripts/generate_hisfrag_test.py --data-path /path/to/HisFrag20/train --output-path /path/to/HisFrag20-512/train```

>```PYTHONPATH=$PYTHONPATH:. python3.8 scripts/generate_hisfrag_test.py --data-path /path/to/HisFrag20/test --output-path /path/to/HisFrag20-512/test```
 
>```python3.8 -u -m torch.distributed.run --nproc_per_node 2 --standalone main.py --cfg configs/pajigsaw/hisfrag20_patch16_512.yaml --data-path /path/to/HisFrag20-512 --batch-size 64 --opts TRAIN.WARMUP_EPOCHS 5 TRAIN.WEIGHT_DECAY 0. TRAIN.EPOCHS 150 TRAIN.BASE_LR 3e-4 DATA.TEST_BATCH_SIZE 512 MODEL.PJS.NUM_HEADS 16 PRINT_FREQ 15``` 