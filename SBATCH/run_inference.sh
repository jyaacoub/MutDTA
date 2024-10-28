#!/bin/bash
#SBATCH -t 30:00
#SBATCH --job-name=run_mutagenesis_davis_esm
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
CSV_OUT="${ROOT_DIR}/jyaacoub/MutDTA/SBATCH/outs/inference_test.csv" # this is used for outputs
BIN_DIR="${ROOT_DIR}/jyaacoub/bin" # for modeller
proj_dir="${ROOT_DIR}/jyaacoub/MutDTA"

module load StdEnv/2020 && module load gcc/9.3.0 && module load arrow/12.0.1
cd ${proj_dir}
source .venv/bin/activate

# NOTE: To get SMILE from a .mol2 or .sdf file you can use RDKIT:
#
#    from rdkit import Chem
#    Chem.MolToSmiles(Chem.MolFromMol2File(x), isomericSmiles=False)
# OR just pass in an sdf file and the script will extract the SMILE

python -u inference.py \
            --ligand_sdf "${proj_dir}/SBATCH/samples/inference/VWW_ideal.sdf" \
            --pdb_files ${proj_dir}/SBATCH/samples/inference/alphaflow_pdbs/*.pdb \
            --csv_out ${CSV_OUT} \
            --model_opt PDBbind_gvpl_aflow \
            --fold ${SLURM_ARRAY_TASK_ID} \
            --ligand_id "1a30_ligand" \
            --batch_size 8 
# batch size is used for when we have multiple input pdbs and we want to optimize inference times
# ./results/model_checkpoints/ours/GVPLM_PDBbind0D_nomsaF_binaryE_128B_0.0002LR_0.2D_2000E_gvpLF_binaryLE.model
# ./results/model_checkpoints/ours/GVPLM_PDBbind0D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE.model
# ./results/model_checkpoints/ours/GVPLM_PDBbind0D_nomsaF_binaryE_128B_0.0002LR_0.46616D_2000E_gvpLF_binaryLE.model_tmp