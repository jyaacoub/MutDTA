#!/bin/bash
#SBATCH -t 180
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/docking/prep/%x-%j.out
#SBATCH --job-name=docking_prep

#SBATCH -p all
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

# adding needed libraries to path:

# Docking and tools
export PATH=/cluster/home/t122995uhn/lib/AutoDock_Vina/bin/:$PATH
export PATH=/cluster/home/t122995uhn/lib/mgltools_x86_64Linux2_1.5.7/bin/:$PATH
export PATH=/cluster/home/t122995uhn/lib/obabel-install/bin/:$PATH

echo which obabel:   "$(which obabel)"
echo which vina:     "$(which vina)"
echo which pythonsh: "$(which pythonsh)"

# Prepares PDBbind dataset for docking
cd /cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts

# activating python with correct packages
#module load py10 << needed?
source ../../../.venv/bin/activate

# Usage: PDBbind_prepare.sh path ADT_path template [OPTIONS]
#        path     - path to PDBbind dir containing pdb for protein to convert to pdbqt (ABSOLUTE PATH).
#        ADT_path - path to MGL root  (e.g.: '~/mgltools_x86_64Linux2_1.5.7/')
#        template - path to conf template file (create empty file if you want vina defaults).
# Options:
#        -sl --shortlist: path to csv file containing a list of pdbcodes to process.
#               Doesn't matter what the file is as long as the first column contains the pdbcodes.
#        -cd --config-dir: path to store new configurations in.
#               Default is to store it with the prepared receptor as <PDBCode>_conf.txt

#
# test protein prep with: pythonsh ${ADT}/prepare_receptor4.py -r ./data/refined-set/3ao2/3ao2_protein.pdb -o ./3ao2.pdbqt

./PDBbind_prepare.sh "/cluster/projects/kumargroup/jean/data/refined-set/" "/cluster/home/t122995uhn/lib/mgltools_x86_64Linux2_1.5.7/" "/cluster/home/t122995uhn/projects/MutDTA/data/PDBbind/kd_ki/X.csv"
