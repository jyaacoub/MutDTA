#!/bin/bash
#SBATCH -t 180
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/docking/prep/9/%x-%A_%a.out #NOTE: change prep#
#SBATCH --job-name=conf_prep9 #NOTE: change prep#

#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.yaacoub@mail.utoronto.ca

#SBATCH -p all
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2

#SBATCH --array=0-20 #NOTE: N total processes

prep_num=9 #NOTE: change prep#

# adding needed libraries to path:

# activating python with correct packages
#module load py10 << needed?
source /cluster/home/t122995uhn/projects/MutDTA/.venv/bin/activate

# Usage: ./bash_scripts/prep_conf_only.sh <path> <template> <shortlist> [<config_dir>]
#          path      - path to PDBbind dir containing pdb for protein to convert to pdbqt (ABSOLUTE PATH).
#          template  - path to conf template file (create empty file if you want vina defaults).
#          shortlist - path to csv file containing a list of pdbcodes to process.
#                      Doesnt matter what the file is as long as the first column contains the pdbcodes.
# Options:
#          config_dir (optional) - path to store new configurations in. default is to store it with the protein as {PDBCode}_conf.txt

prepsh="/cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts/prep_conf_only.sh"

path="/cluster/projects/kumargroup/jean/data/refined-set/"
template="/cluster/projects/kumargroup/jean/data/vina_conf/run${prep_num}.conf"
shortlist="/cluster/projects/kumargroup/jean/data/shortlists/kd_ki/${SLURM_ARRAY_TASK_ID}.csv "

conf_dir="/cluster/projects/kumargroup/jean/data/vina_conf/run${prep_num}"

$prepsh $path $template $shortlist $conf_dir