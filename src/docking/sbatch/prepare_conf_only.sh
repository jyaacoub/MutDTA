#!/bin/bash
#SBATCH -t 180
#SBATCH -o /cluster/projects/kumargroup/jean/slurm-outputs/docking/prep/%x-%j.out
#SBATCH --job-name=prep_conf

#SBATCH -p all
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

# Assumes we are working with PDBbind dataset (https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz") 
# that has the following file structure:  
#
# refined-set/
#     1a1e/
#       1a1e_ligand.mol2  
#       1a1e_ligand.pdbqt  
#       1a1e_ligand.sdf  
#       1a1e_pocket.pdb  
#       1a1e_protein.pdb
#     1a28/
#       ...
#     ...
echo $(pwd)

# Check if the required arguments are provided
if [ $# -lt 2 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 <path> <template> <shortlist> [<conf_dir>]"
  echo -e "\t path - path to PDBbind dir containing pdb for protein to convert to pdbqt (ABSOLUTE PATH)."
  echo -e "\t template - path to conf template file"
  echo -e "\t shortlist - path to csv file containing a list of pdbcodes to process."
  echo -e "\t            Doesnt matter what the file is as long as the first column contains the pdbcodes."
  echo -e "\t conf_dir (optional) - path to store new configurations in. default is to store it with the protein as {PDBCode}_conf.txt"
  exit 1
fi
# e.g. use:
# prepare_conf_only.sh /cluster/projects/kumargroup/jean/data/refined-set ./data/vina_conf/run4.conf 
#                       ./data/shortlists/no_err_50/sample.csv ./data/vina_conf/run4/

echo -e "\n### Starting ###\n"
PDBbind_dir=$1
template=$2
shortlist=$3
#NOTE: Hard coded path for pyconf 
pyconf_path="/cluster/home/t122995uhn/projects/MutDTA/src/docking/prep_conf.py"

if [ $# -eq 4 ]; then
  conf_dir=$4
else
  conf_dir=""
fi

# Checking if shortlist file exists
#   [not empty] and [not a file]
if [[ ! -z "$shortlist" ]] && [[ ! -f "$shortlist" ]]; then
  echo "shortlist file does not exist: ${shortlist}"
  exit 1
fi

if [[ ! -z "$shortlist" ]]; then
  # if shortlist is provided, use it
  echo "Using shortlist file: ${shortlist}"
  # getting all the pdbcodes from the shortlist file
  codes=$(awk -F',' 'NR>1 {print $1}' "$shortlist") #TODO: fix this! (see run_vina.sh)

  # Verifying that all pdbcodes in shortlist exist in PDBbind dir
  dirs=""
  for code in $codes; do
    if [[ ! -d "${PDBbind_dir}/${code}" ]]; then
      echo "PDBbind dir (${PDBbind_dir}) does not contain ${code} specified in shortlist file"
      exit 1
    fi
    dirs="${dirs} ${PDBbind_dir}/${code}"
  done
  total=$(echo "$dirs" | wc -w)
else # otherwise use all
  # looping through all the pdbcodes in the PDBbind dir select everything except index and readme dir
  dirs=$(find "$PDBbind_dir" -mindepth 1 -maxdepth 1 -type d -not -name "index" -not -name "readme")
  # getting count of dirs
  total=$(echo "$dirs" | wc -l)
fi

count=0

# loop through each pdbcodes
for dir in $dirs; do
  code=$(basename "$dir")
  echo -e "Processing $code \t: $((++count)) / $total"

  # new protien and lig files
  protein="${dir}/${code}_protein.pdbqt"
  ligand="${dir}/${code}_ligand.pdbqt"

  # preparing config file with binding site info
  pocket="${dir}/${code}_pocket.pdb"

  # getting out path for conf file
  if [[ ! -z  $conf_dir ]]; then  # if not empty arg
    out_p="${conf_dir}/${code}_conf.txt"
  else
    out_p="${dir}/${code}_conf.txt"
  fi

  python $pyconf_path -r $protein -l $ligand -pp $pocket -o $out_p -c $template

  #checking error code
  if [ $? -ne 0 ]; then
    echo "prep_conf.py failed to prepare config file for $code"
    exit 1
  fi
done
