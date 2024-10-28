#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --job-name=retrain-davis_GVPL

#SBATCH -A def-sushant

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=j.yaacoub@mail.utoronto.ca

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
##SBATCH --dependency=

#SBATCH --output=./outs/%x_%a.out

# we want to run across all 5 folds at once
#SBATCH --array=0-4
module load StdEnv/2020 && module load gcc/9.3.0 && module load arrow/12.0.1

cd /home/jyaacoub/projects/def-sushant/jyaacoub/MutDTA/
source .venv/bin/activate


cd /cluster/home/t122995uhn/projects/MutDTA/
                    
source .venv/bin/activate

python -u train_test.py --model_opt GVPL \
                     --data_opt davis \
                     \
                     --feature_opt nomsa \
                     --edge_opt binary \
                     --ligand_feature_opt gvp \
                     --ligand_edge_opt binary \
                     \
                     --learning_rate 0.00020535607176845963 \
                     --batch_size 128 \
                     --dropout 0.08845592454543601 \
                     --output_dim 512 \
                     \
                     --train \
                     --fold_selection ${SLURM_ARRAY_TASK_ID} \
                     --num_epochs 2000
