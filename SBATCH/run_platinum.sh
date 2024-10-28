#!/bin/bash
#SBATCH -t 10:00
#SBATCH --job-name=run_platinum
#SBATCH --mem=10G

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4

#SBATCH --output=./outs/%x_%a.out
#SBATCH --array=0

# runs across all folds for a model
# should produce a matrix for each fold

# Then to get most accurate mutagenesis you can average these matrices
# and visualize them with src.analysis.mutagenesis_plot.plot_sequence
ROOT_DIR="/lustre06/project/6069023"
cd ${ROOT_DIR}/jyaacoub/MutDTA
source .venv/bin/activate

python -u run_platinum.py \
        --model_opt davis_DG \
        --fold ${SLURM_ARRAY_TASK_ID} \ 
        --out_dir ./
