#!/usr/bin/env bash
#SBATCH --job-name=hisfrag
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint=p100
#SBATCH --cpus-per-task=3
#SBATCH --mem=64gb
#SBATCH --chdir=/beegfs/mvu/pajigsaw
#SBATCH --output=/beegfs/mvu/pajigsaw/output/hisfrag-eval-%x-%j.out
#SBATCH -e /beegfs/mvu/pajigsaw/output/hisfrag-eval-%x-%j.err

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=13640
export WORLD_SIZE=10

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### the command to run
srun ~/apps/bin/python3.10 hisfrag_test.py --cfg configs/pajigsaw/hisfrag20_patch16_512.yaml --data-path /beegfs/mvu/datasets/HisFrag20 --batch-size 256 --pretrained output/hisfrag20_patch16_512/default/best_model.pth --opts DATA.NUM_WORKERS 3 DATA.TEST_BATCH_SIZE 384 PRINT_FREQ 30