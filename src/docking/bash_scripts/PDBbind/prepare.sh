#!/bin/bash
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

# Check if the required arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <path>"
    echo -e "\t path - path to PDBbind dir containing pdb for protein to convert to pdbqt."
    echo -e "\t ADT_path - path to MGL root  (e.g.: '~/mgltools_x86_64Linux2_1.5.7/')"
    exit 1
fi
# e.g. use: prep_lig.sh data/PDBbind/raw/refined-set /home/jyaacoub/mgltools_x86_64Linux2_1.5.7/

echo -e "\n### Starting ###\n"
PDBbind_dir=$1
ADT_path="${2}/MGLToolsPckgs/AutoDockTools/Utilities24/"

# pre-run checks:
# Checking if obabel command is available (install from https://openbabel.org/wiki/Category:Installation)
if ! command -v obabel >/dev/null 2>&1; then
  echo "obabel is not installed or not in the system PATH"
  exit 1
fi

# Checking if pythonsh command is available (part of MGLTools)
if [[ ! -f "${2}/bin/pythonsh" ]]; then
  echo "pythonsh is not installed or not in the system PATH"
  exit 1
fi

# Checking if prepare_receptor4.py exists (part of MGLTools)
if [[ ! -f "${ADT_path}/prepare_receptor4.py" ]]; then
  echo "prepare_receptor4.py does not exist in the specified location: ${ADT_path}"
  exit 1
fi

# Check to see if ../prep_conf.py file exists
if [[ ! -f "../../prep_conf.py" ]]; then
  echo "../../prep_conf.py does not exist, make sure to run this script from the src/docking/bash_scripts/PDBbind dir."
  exit 1
fi

# looping through all the pdbcodes in the PDBbind dir select everything except index and readme dir
dirs=$(find "$PDBbind_dir" -mindepth 1 -maxdepth 1 -type d -not -name "index" -not -name "readme")

# getting count of dirs
total=$(echo "$dirs" | wc -l)

count=0

# loop through each pdbcodes
for dir in $dirs; do
    code=$(basename "$dir")
    ligand="${dir}/${code}_ligand.sdf"
    protein="${dir}/${code}_protein.pdb"
    pocket="${dir}/${code}_pocket.pdb"

    echo -e "Processing $code \t: $((++count)) / $total"

    # running obabel to convert ligand to pdbqt
    obabel -isdf $ligand --title $code -opdbqt -O "${dir}/${code}_ligand.pdbqt"

    #checking error code
    if [ $? -ne 0 ]; then
        echo "obabel failed to convert ligand to pdbqt for $code"
        exit 1
    fi

    # running prepare_receptor4.py to convert protein to pdbqt
    "${2}/bin/pythonsh" "${ADT_path}/prepare_receptor4.py" -r $protein -o "${dir}/${code}_protein.pdbqt" -A checkhydrogens -U nphs_lps_waters_nonstdres

    #checking error code
    if [ $? -ne 0 ]; then
        echo "prepare_receptor4.py failed to convert protein to pdbqt for $code"
        exit 1
    fi

    # preparing config file with binding site info
    python ../../prep_conf.py -r $protein -l $ligand -pp $pocket -o "${dir}/${code}_conf.txt"

    #checking error code
    if [ $? -ne 0 ]; then
        echo "prep_conf.py failed to prepare config file for $code"
        exit 1
    fi
done