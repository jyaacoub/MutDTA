#!/bin/bash
#SBATCH -t 10:00
#SBATCH --job-name=run_mutagenesis
#SBATCH --mem=10G

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4

#SBATCH --output=./%x_%a.out
#SBATCH --array=0

# runs across all folds for a model
# should produce a matrix for each fold

# Then to get most accurate mutagenesis you can average these matrices
# and visualize them with src.analysis.mutagenesis_plot.plot_sequence

# Modeller is needed for this to run... (see: Generic install - https://salilab.org/modeller/10.5/release.html#unix)
export PYTHONPATH="${PYTHONPATH}:/home/jyaacoub/bin/modeller10.5/lib/x86_64-intel8/python3.3:/home/jyaacoub/bin/modeller10.5/lib/x86_64-intel8:/home/jyaacoub/bin/modeller10.5/modlib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/jyaacoub/bin/modeller10.5/lib/x86_64-intel8"

cd /home/jyaacoub/projects/def-sushant/jyaacoub/MutDTA
source .venv/bin/activate

# NOTE: To get SMILE from a .mol2 or .sdf file you can use RDKIT:
#
#    from rdkit import Chem
#    mol2smile = lambda x: Chem.MolToSmiles(Chem.MolFromMol2File(x), isomericSmiles=False)
#    mol2smile("/home/jyaacoub/scratch/pdbbind_demo/1a30/1a30_ligand.mol2")

python -u run_mutagenesis.py \
                    --ligand_smile "CC(C)CC(NC(=O)C(CC(=O)[O-])NC(=O)C([NH3+])CCC(=O)[O-])C(=O)[O-]" \
                    --ligand_smile_name "1a30_ligand" \
                    --pdb_file "/home/jyaacoub/projects/def-sushant/jyaacoub/data/kiba/alphaflow_io/out_pdb_MD-distilled/P67870.pdb" \
                    --out_path "/home/jyaacoub/scratch/mutagenesis_tests/" \
                    --res_start 0 \
                    --res_end 5 \
                    --model_opt davis_DG \
                    --fold ${SLURM_ARRAY_TASK_ID}
