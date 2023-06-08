#!/bin/bash
# Assumes we are working with PDBbind dataset (https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz") 
# that has the following file structure:  
#
# refined-set/
#     1a1e/
#       1a1e_conf.txt   **
#       1a1e_ligand.mol2  
#       1a1e_ligand.pdbqt ** 
#       1a1e_ligand.sdf  
#       1a1e_pocket.pdb  
#       1a1e_protein.pdb
#       1a1e_protein.pdbqt **
#     1a28/
#       ...
#     ...
# ** created by PDBbind_prepare.sh

# Check if the required arguments are provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <path> <shortlist>"
    echo -e "\t path - path to PDBbind dir containing pdb for protein to convert to pdbqt."
    echo -e "\t shortlist - path to csv file containing a list of pdbcodes to process."
    echo -e "\t            Doesnt matter what the file is as long as the first column contains the pdbcodes."
    echo -e "\t conf_dir (optional) - path to configuration dir for each pdb. Default is the same path as pdb file itself."
    exit 1
fi
# e.g. use: run_vina.sh /home/jyaacoub/projects/MutDTA/data/PDBbind/raw/refined-set /home/jyaacoub/projects/MutDTA/data/PDBbind/kd_ki/X.csv

echo -e "\n### Starting ###\n"
PDBbind_dir=$1
shortlist=$2

if [ $# -eq 3 ]; then
    conf_dir=$3
else
    conf_dir=""
fi

# pre-run checks:
# Checking if shortlist file exists
#   [not empty] and [not a file]
if [[ ! -z "$shortlist" ]] && [[ ! -f "$shortlist" ]]; then
  echo "shortlist file does not exist: ${shortlist}"
  exit 1
fi

# if shortlist is provided we use it to get the list of pdbcodes to process
if [[ ! -z "$shortlist" ]]; then
  echo "Using shortlist file: ${shortlist}"
  # getting all the pdbcodes from the shortlist file ignoring first row...
  if [[ $(head -n 1 $shortlist) == "PDBCode"* ]]; then
    codes=$(cut -d',' -f1 $shortlist | tail -n +2)
  else
    codes=$(cut -d',' -f1 $shortlist)
  fi

  # Verifying that all pdbcodes in shortlist exist in PDBbind dir
  dirs=""
  for code in $codes; do
    if [[ ! -d "${PDBbind_dir}/${code}" ]]; then
      echo "PDBbind dir does not contain ${code} specified in shortlist file"
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
errors=0

# reset error file
echo "" > "./vina_error_pdbs.txt"

for dir in $dirs; do
    code=$(basename "$dir")
    if [[ ! -z  $conf_dir ]]; then
      conf="${conf_dir}/${code}_conf.txt"
    else
      conf="${dir}/${code}_conf.txt"
    fi
    
    echo -e "Processing $code \t: $((++count)) / $total \t: $((errors)) errors"

    # Checking if conf file exists
    if [[ ! -f "$conf" ]]; then
        echo -e "\tERROR: conf file does not exist: ${conf}"
        errors=$((errors+1))
        # output to error file
        echo "$code" >> "./vina_error_pdbs.txt"
        continue
    fi

    # checking to make sure the output file does not already exist
    if [[ -f "${dir}/${code}_vina_out.pdbqt" ]]; then
        echo -e "\tSkipping...output file already exists: ${dir}/${code}_vina_out.pdbqt"
        continue
    fi

    # seed and all other default args are specified in prep_conf.py 
    vina --config "$conf"

    # Checking error code
    if [ $? -ne 0 ]; then
        echo "Error running vina for ${code}"
        errors=$((errors+1))
        # output to error file
        echo "$code" >> "./vina_error_pdbs.txt"
    fi
done
