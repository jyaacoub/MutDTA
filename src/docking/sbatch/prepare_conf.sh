#!/bin/bash
#SBATCH -t 180
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/docking/prep/%x-%j.out
#SBATCH --job-name=prep_conf

#SBATCH -p all
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

/cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts/prep_conf_only.sh /cluster/projects/kumargroup/jean/data/refined-set /cluster/projects/kumargroup/jean/data/vina_conf/run7.conf /cluster/projects/kumargroup/jean/data/shortlists/kd_ki/info.csv /cluster/projects/kumargroup/jean/data/vina_out/run7
