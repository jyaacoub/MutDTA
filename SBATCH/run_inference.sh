#!/bin/bash
#SBATCH -t 30:00
#SBATCH --job-name=run_mutagenesis_davis_esm
#SBATCH --mem=10G

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4

#SBATCH --output=./outs/%x_%a.out
#SBATCH --array=0

ROOT_DIR="/lustre06/project/6069023"
CSV_OUT="${ROOT_DIR}/jyaacoub/MutDTA/SBATCH/samples/out/inference_test.csv" # this is used for outputs
proj_dir="${ROOT_DIR}/jyaacoub/MutDTA"
ligand_sdf="${proj_dir}/SBATCH/samples/input/inference/VWW_ideal.sdf"
pdb_files=${proj_dir}/SBATCH/samples/input/inference/alphaflow_pdbs/*.pdb

module load StdEnv/2020 && module load gcc/9.3.0 && module load arrow/12.0.1
cd ${proj_dir}
source .venv/bin/activate


python -u inference.py \
            --ligand_sdf $ligand_sdf \
            --pdb_files $pdb_files \
            --csv_out ${CSV_OUT} \
            --model_opt PDBbind_gvpl_aflow \
            --fold ${SLURM_ARRAY_TASK_ID} \
            --ligand_id "1a30_ligand" \
            --batch_size 8 
# batch size is used for when we have multiple input pdbs and we want to optimize inference times
# ./results/model_checkpoints/ours/GVPLM_PDBbind0D_nomsaF_binaryE_128B_0.0002LR_0.2D_2000E_gvpLF_binaryLE.model
# ./results/model_checkpoints/ours/GVPLM_PDBbind0D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE.model
# ./results/model_checkpoints/ours/GVPLM_PDBbind0D_nomsaF_binaryE_128B_0.0002LR_0.46616D_2000E_gvpLF_binaryLE.model_tmp