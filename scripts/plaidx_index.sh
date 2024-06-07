#!/bin/sh
#SBATCH --job-name=index
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate colbert

export PATH=/home/dju/others/ninja:$PATH

# CUDAVER=cuda-11
# export PATH=/usr/local/$CUDAVER/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH
# export CUDA_PATH=/usr/local/$CUDAVER
# export CUDA_ROOT=/usr/local/$CUDAVER
# export CUDA_HOME=/usr/local/$CUDAVER
# export CUDA_HOST_COMPILER=/usr/bin/gcc
# export CPATH=/home/dju/miniconda3/envs/plaid/bin/gcc

python plaidx/index.py \
    --corpus /home/dju/datasets/neuclir-csl/collections \
    --model_name_or_path eugene-yang/plaidx-xlmr-large-mlir-neuclir \
    --exp_name neuclir-csl.plaidx
