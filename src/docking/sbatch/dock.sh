#!/bin/bash
#SBATCH -t 10
#SBATCH -o slurm-outputs/docking/%x-%j.out
#SBATCH --job-name=vina_dock

#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

# adding needed libraries to path:

# Docking and tools
export PATH=/cluster/home/t122995uhn/lib/AutoDock_Vina/bin/:$PATH

echo which vina:     "$(which vina)"

cd /cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts

# The following args are needed:
# Usage: MutDTA/src/docking/bash_scripts/run_vina.sh <path> <vina_path> <shortlist>
#         path - path to PDBbind dir containing pdb for protein to convert to pdbqt.
#         shortlist (optional) - path to csv file containing a list of pdbcodes to process.
#                    Doesnt matter what the file is as long as the first column contains the pdbcodes.

./run_vina.sh "/cluster/projects/kumargroup/jean/data/refined-set/" "/cluster/home/t122995uhn/projects/MutDTA/data/PDBbind/kd_ki/X.csv"
