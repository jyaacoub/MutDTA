#!/bin/bash
# WARNING: DDP submission doesnt work with ARRAYS!
#SBATCH -t 20
#SBATCH --job-name=DDP_sub-folds #NOTE: change this and fold_selection arg below

#SBATCH --mail-type=FAIL
##SBATCH --mail-user=j.yaacoub@mail.utoronto.ca

#SBATCH -c 4
#SBATCH --mem=4G

#SBATCH --output=/home/jyaacoub/projects/def-sushant/jyaacoub/MutDTA/SBATCH/outs/%x.out
##SBATCH --array=0-4 # WARNING: Doesnt work with submitit

module load StdEnv/2020 && module load gcc/9.3.0 && module load arrow/12.0.1
ROOT_DIR="/lustre06/project/6069023"
cd ${ROOT_DIR}/jyaacoub/MutDTA/
                    
source .venv/bin/activate
# 3days == 4320 mins
# batch size is local batch sizes
for fold in {0..4} #{0..4}
do
	echo "------------ FOLD $fold ------------"
	# pdbbind_esm
	python -u train_test_DDP.py --model_opt GVPL_ESM \
						--data_opt PDBbind \
						\
						--feature_opt nomsa \
						--edge_opt binary \
						--ligand_feature_opt gvp \
						--ligand_edge_opt binary \
						\
						--learning_rate 0.00009326978279144084 \
						--batch_size 32 \
						\
						--dropout 0.06042135076915635 \
						--dropout_prot 0.0 \
						--output_dim 512 \
						--num_GVPLlayers 2 \
						--pro_dropout_gnn 0.10542576796343962 \
						--pro_extra_fc_lyr False \
						--pro_emb_dim 128 \
						\
						--train \
						--fold_selection $fold \
						--num_epochs 2000 \
						-odir ./DDP_outs/pdbbind_gvpl_esm/%j \
						-s_t 4320 -s_m 18GB -s_nn 1 -s_ng 4 -s_cp 4
	# # kiba_esm
	# python -u train_test_DDP.py --model_opt EDI \
	# 					--data_opt kiba \
	# 					\
	# 					--feature_opt nomsa \
	# 					--edge_opt binary \
	# 					--ligand_feature_opt original \
	# 					--ligand_edge_opt binary \
	# 					\
	# 					--learning_rate 0.0001 \
	# 					--batch_size 12 \
	# 					\
	# 					--dropout 0.4 \
	# 					--dropout_prot 0.0 \
	# 					--output_dim 128 \
	# 					--pro_extra_fc_lyr False \
	# 					--pro_emb_dim 512 \
	# 					\
	# 					--train \
	# 					--fold_selection 0 \
	# 					--num_epochs 2000 \
	# 					-odir ./slurm_outs/kiba_esm/%j \
	# 					-s_t 4320 -s_m 18GB -s_nn 1 -s_ng 4 -s_cp 4
done

