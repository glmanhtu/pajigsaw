#!/usr/bin/env bash
#SBATCH --job-name=hisfrag
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/beegfs/mvu/pajigsaw
#SBATCH --output=/beegfs/mvu/pajigsaw/output/hisfrag20_patch16_512/default/hisfrag-train-%x-%j.out
#SBATCH -e /beegfs/mvu/pajigsaw/output/hisfrag20_patch16_512/default/hisfrag-train-%x-%j.err

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=13641
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### the command to run
### srun ~/apps/bin/python3.10 hisfrag_test.py --cfg configs/pajigsaw/hisfrag20_patch16_512.yaml --data-path /beegfs/mvu/datasets/HisFrag20 --batch-size 96 --pretrained output/hisfrag20_patch16_512/default/best_model.pth --opts DATA.NUM_WORKERS 4 DATA.TEST_BATCH_SIZE 512 PRINT_FREQ 30
srun ~/apps/bin/python3.10 hisfrag.py --cfg configs/pajigsaw/hisfrag20_patch16_512.yaml --data-path /beegfs/mvu/datasets/HisFrag20 --batch-size 64 --opts TRAIN.WARMUP_EPOCHS 5 TRAIN.WEIGHT_DECAY 0.05 TRAIN.EPOCHS 300 TRAIN.BASE_LR 3e-4 DATA.NUM_WORKERS 3 PRINT_FREQ 50