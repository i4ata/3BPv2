#!/bin/bash
#SBATCH --time=00:59:00
#SBATCH --job-name=bodies_job
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1

module purge
module load Python/3.9.6-GCCcore-11.2.0

source ~/env/bin/activate

python train.py

deactivate
