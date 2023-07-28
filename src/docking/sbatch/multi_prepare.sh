#!/bin/bash
#SBATCH -t 180
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/docking/prep/11/%x-%A_%a.out #NOTE: change prep#
#SBATCH --job-name=v_prep11 #NOTE: change prep#

##SBATCH --mail-type=ALL
##SBATCH --mail-user=j.yaacoub@mail.utoronto.ca

#SBATCH -p all
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2

#SBATCH --array=0 #NOTE: N total processes

prep_num=11 #NOTE: change prep#
flexible=true

# adding needed libraries to path:

# Docking and tools
export PATH=/cluster/home/t122995uhn/lib/AutoDock_Vina/bin/:$PATH
export PATH=/cluster/home/t122995uhn/lib/mgltools_x86_64Linux2_1.5.7/bin/:$PATH
export PATH=/cluster/home/t122995uhn/lib/obabel-install/bin/:$PATH

echo which obabel:   "$(which obabel)"
echo which vina:     "$(which vina)"
echo which pythonsh: "$(which pythonsh)"

# activating python with correct packages
#module load py10 << needed?
source /cluster/home/t122995uhn/projects/MutDTA/.venv/bin/activate

# Usage: PDBbind_prepare.sh path ADT_path template [OPTIONS]
#        path     - path to PDBbind dir containing pdb for protein to convert to pdbqt (ABSOLUTE PATH).
#        ADT_path - path to MGL root  (e.g.: '~/mgltools_x86_64Linux2_1.5.7/')
#        template - path to conf template file (create empty file if you want vina defaults).
# Options:
#        -sl --shortlist: path to csv file containing a list of pdbcodes to process.
#               Doesn't matter what the file is as long as the first column contains the pdbcodes.
#        -cd --config-dir: path to store new configurations in.
#               Default is to store it with the prepared receptor as <PDBCode>_conf.txt

prepsh="/cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts/PDBbind_prepare.sh"
PDBbind="/cluster/projects/kumargroup/jean/data/refined-set/"
ADT="/cluster/home/t122995uhn/lib/mgltools_x86_64Linux2_1.5.7/"
template="/cluster/projects/kumargroup/jean/data/vina_conf/run${prep_num}.conf"

shortlist="/cluster/projects/kumargroup/jean/data/shortlists/refined-set/${SLURM_ARRAY_TASK_ID}.csv "
conf_dir="/cluster/projects/kumargroup/jean/data/vina_conf/run${prep_num}"

if [ ! -d $conf_dir ]; then
  mkdir $conf_dir
fi

if $flexible; then
  # -f flag for flexible receptor
  $prepsh $PDBbind $ADT $template -sl $shortlist -cd $conf_dir -f
else
  $prepsh $PDBbind $ADT $template -sl $shortlist -cd $conf_dir
fi
