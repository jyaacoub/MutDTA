#!/bin/bash
#SBATCH -t 2-00:00:00 #days-hours:minutes:seconds
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/docking/run11/%x-%A_%a.out #NOTE: change run#

#SBATCH --job-name=r11_vina_dock #NOTE: change run#
##SBATCH --mail-type=ALL
##SBATCH --mail-user=j.yaacoub@mail.utoronto.ca

#SBATCH --mem=4G
#SBATCH --cpus-per-task=8

#SBATCH --array=0-40 #NOTE: N total processes
# takes ~1.5 days on 5 process nodes with 3,000 pdbs

# adding needed libraries to path:

# Docking and tools
export PATH=/cluster/home/t122995uhn/lib/AutoDock_Vina/bin/:$PATH
run_num=11 #NOTE: change run#
echo which vina:     "$(which vina)"
echo "run_num: $run_num"

# The following args are needed:
# Usage: MutDTA/src/docking/bash_scripts/run_vina.sh <path> <vina_path> <shortlist>
#         path - path to PDBbind dir containing pdb for protein to convert to pdbqt.
#         shortlist (optional) - path to csv file containing a list of pdbcodes to process.
#                    Doesnt matter what the file is as long as the first column contains the pdbcodes.
# 
#NOTe: To test vina: run vina --config "$conf" --out "${dir}/${code}_vina_out.pdbqt" --log "${dir}/${code}_vina_log.txt" --seed 904455071
# e.g.: vina --config /cluster/projects/kumargroup/jean/data/refined-set/1a1e/1a1e_conf.txt --seed 904455071
# shortlist file is different for each job in the array
conf_dir="/cluster/projects/kumargroup/jean/data/vina_conf/run${run_num}/"
shortlist=/cluster/projects/kumargroup/jean/data/shortlists/refined-set/${SLURM_ARRAY_TASK_ID}.csv
vina=/cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts/run_vina.sh
data="/cluster/projects/kumargroup/jean/data/refined-set/"

echo "shortlist: $shortlist"
echo "conf_dir: $conf_dir"
echo "data: $data"

$vina $data $shortlist $conf_dir

# Then extract with `extract_vina_out.py`
