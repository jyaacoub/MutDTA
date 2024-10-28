#!/bin/bash
#SBATCH -t 40 
#SBATCH --job-name=test_kiba_GVPL
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

python -u test.py --model_opt GVPL \
                    --data_opt kiba \
                     \
                     --feature_opt nomsa \
                     --edge_opt aflow \
                     --ligand_feature_opt gvp \
                     --ligand_edge_opt binary \
                     \
                     --learning_rate 0.00005480618584919115 \
                     --batch_size 32 \
                     --dropout 0.0808130125360696 \
                     --output_dim 512 \
                     --num_GVPLayers 4 \
                     \
                     --train \
                     --fold_selection ${SLURM_ARRAY_TASK_ID} \
                     --num_epochs 2000

# kiba_GVPL only 
python -u test.py --model_opt GVPL \
                      --data_opt kiba \
                     \
                     --feature_opt nomsa \
                     --edge_opt binary \
                     --ligand_feature_opt gvp \
                     --ligand_edge_opt binary \
                     \
                     --learning_rate 0.00003372637625954074 \
                     --batch_size 32 \
                     --dropout 0.09399264336737133 \
                     --output_dim 512 \
                     --num_GVPLayers 4 \
                     \
                     --train \
                     --fold_selection ${SLURM_ARRAY_TASK_ID} \
                     --num_epochs 2000

