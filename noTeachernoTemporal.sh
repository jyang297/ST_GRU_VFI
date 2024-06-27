#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=220:0:0
#SBATCH --mail-user=jyang297@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:V100l:2
module purge
# Change python version if necessary
module load StdEnv/2023
module load python/3.11
module load opencv/4.9

# Source the virtual env
cd ~/$projects/projects/def-jyzhao/jyang297
source ~/projects/def-jyzhao/jyang297/lstm/bin/activate

# Run python script here
cd ~/projects/def-jyzhao/jyang297/Ablation/noTeachernoTemporal_ST_GRU_VFI && torchrun --nproc_per_node=2 trainNoTeacherNoTemporal.py
