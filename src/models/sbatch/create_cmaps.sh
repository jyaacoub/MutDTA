#!/bin/bash
#SBATCH -t 180
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/models/%x-%j.out
#SBATCH --job-name=create_cmap

#SBATCH -p all
#SBATCH --mem=500M
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

#this will create contact maps for all proteins in PDBbind
