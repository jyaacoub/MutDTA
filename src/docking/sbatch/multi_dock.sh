#!/bin/bash
#SBATCH -t 4-00:00:00 #days-hours:minutes:seconds
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/docking/run7/%x-%A_%a.out

#SBATCH --job-name=r7_vina_dock
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.yaacoub@mail.utoronto.ca

#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

#SBATCH --array=0-4 # 5 total processes
# takes ~1.5 days on 5 process nodes with 3,000 pdbs

# adding needed libraries to path:

# Docking and tools
export PATH=/cluster/home/t122995uhn/lib/AutoDock_Vina/bin/:$PATH
run_num=7
echo which vina:     "$(which vina)"
echo "run_num: $run_num" # specify ahead of time with run_num=4

# The following args are needed:
# Usage: MutDTA/src/docking/bash_scripts/run_vina.sh <path> <vina_path> <shortlist>
#         path - path to PDBbind dir containing pdb for protein to convert to pdbqt.
#         shortlist (optional) - path to csv file containing a list of pdbcodes to process.
#                    Doesnt matter what the file is as long as the first column contains the pdbcodes.
# 
#NOTE: To test vina: run vina --config "$conf" --out "${dir}/${code}_vina_out.pdbqt" --log "${dir}/${code}_vina_log.txt" --seed 904455071
# e.g.: vina --config /cluster/projects/kumargroup/jean/data/refined-set/1a1e/1a1e_conf.txt --seed 904455071
# shortlist file is different for each job in the array
conf_dir="/cluster/projects/kumargroup/jean/data/vina_conf/run${run_num}/"
shortlist=/cluster/projects/kumargroup/jean/data/shortlists/no_err_50/${SLURM_ARRAY_TASK_ID}.csv

echo "shortlist: $shortlist"
echo "conf_dir: $conf_dir"

/cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts/run_vina.sh "/cluster/projects/kumargroup/jean/data/refined-set/" $shortlist $conf_dir

# Then extract with `extract_vina_out.py`