#!/bin/bash
#SBATCH -t 30:00
#SBATCH --job-name=run_mutagenesis_davis_DG_custom
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
OUT_DIR="${ROOT_DIR}/jyaacoub/MutDTA/SBATCH/outs/mutagenesis_tests" # this is used for outputs
BIN_DIR="${ROOT_DIR}/jyaacoub/bin" # for modeller
proj_dir="${ROOT_DIR}/jyaacoub/MutDTA"

module load StdEnv/2020 && module load gcc/9.3.0 && module load arrow/12.0.1
# Modeller is needed for this to run... (see: Generic install - https://salilab.org/modeller/10.5/release.html#unix)
export PYTHONPATH="${PYTHONPATH}:${BIN_DIR}/modeller10.5/lib/x86_64-intel8/python3.3:${BIN_DIR}/modeller10.5/lib/x86_64-intel8:${BIN_DIR}/modeller10.5/modlib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${BIN_DIR}/modeller10.5/lib/x86_64-intel8"

cd ${proj_dir}
source .venv/bin/activate

# NOTE: To get SMILE from a .mol2 or .sdf file you can use RDKIT:
#
#    from rdkit import Chem
#    mol2smile = lambda x: Chem.MolToSmiles(Chem.MolFromMol2File(x), isomericSmiles=False)
#    mol2smile("${ROOT_DIR}/jyaacoub/scratch/pdbbind_demo/1a30/1a30_ligand.mol2")

python -u run_mutagenesis.py \
                    --ligand_smile "CC(C)CC(NC(=O)C(CC(=O)[O-])NC(=O)C([NH3+])CCC(=O)[O-])C(=O)[O-]" \
                    --ligand_id "1a30_ligand" \
                    --pdb_file "${proj_dir}/SBATCH/samples/input/mutagenesis/P67870.pdb" \
                    --out_path "${OUT_DIR}/" \
                    --mutations E5A E6A V7A \
                    --model_opt davis_DG \
                    --fold ${SLURM_ARRAY_TASK_ID}
