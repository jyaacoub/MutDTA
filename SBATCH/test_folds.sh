#!/bin/bash
#SBATCH -t 40 
#SBATCH --job-name=test_davis_aflow
#SBATCH --mem=10G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.yaacoub@mail.utoronto.ca

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8

#SBATCH --output=./outs/%x_%a.out
#SBATCH --array=0-4
module load StdEnv/2020 && module load gcc/9.3.0 && module load arrow/12.0.1

cd /home/jyaacoub/projects/def-sushant/jyaacoub/MutDTA/
                    
source .venv/bin/activate

python -u test.py --model_opt DG \
                    --data_opt davis \
                     \
                     --feature_opt nomsa \
                     --edge_opt aflow \
                     --ligand_feature_opt original \
                     --ligand_edge_opt binary \
                     \
                     --learning_rate 0.0008279387625584954 \
                     --batch_size 128 \
                     --dropout 0.3480347297724069 \
                     --output_dim 256 \
                     \
                     --save_pred_test \
                     --fold_selection ${SLURM_ARRAY_TASK_ID} 


