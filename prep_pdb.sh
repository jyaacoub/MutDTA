#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <path> <pdbcode> [<complex>] <ADT_path>"
    echo -e "\t path - path to the directory containing the pdb file '<pdbcode>.pdb'"
    echo -e "\t pdbcode - pdb code of the structure"
    echo -e "\t complex - (optional) 'a' if you want all structures extracted, 'l' if you only want the largest structure (receptor). Default is 'm', to extract only receptor (largest structure) and its ligand (closest structure to receptor)."
    echo -e "\t ADT_path - path to MGLToolsPckgs/AutoDockTools/Utilities24/"
    exit 1
fi

# Assign the arguments to variables
path=$1
pdbcode=$2
ADT_path=$3

# Check if the 'complex' argument was provided, otherwise set a default value
if [ $# -eq 4 ]; then
    complex=$3
    ADT_path=$4
else
    complex="m"
fi
# ADT_path="/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/"
# path="test_prep/"
# pdbcode="1a1e"

pdb_path="${path}/${pdbcode}.pdb"
prep_path="${path}/prep/${pdbcode}"


### Error checking
# Checking to see that file exists
if [[ ! -f "${pdb_path}" ]]; then
  echo "File ${pdb_path} does not exist"
  exit 1
fi

# creating prep directory if it does not exist
if [[ ! -d "${path}/prep" ]]; then
  mkdir "${path}/prep"
fi

# running prepare_receptor4.py (from AutoDockTools) to clean up pdb file
# Checking if pythonsh command is available
if ! command -v pythonsh >/dev/null 2>&1; then
  echo "pythonsh is not installed or not in the system PATH"
  exit 1
fi

# Checking if prepare_receptor4.py exists
if [[ ! -f "${ADT_path}/prepare_receptor4.py" ]]; then
  echo "prepare_receptor4.py does not exist in the specified location: ${ADT_path}"
  exit 1
fi

### Running prepare_receptor4.py
# note that we do not use -e flag so that we can also 
# extract ligand if it is present
echo -e "Running prepare_receptor4.py\n"
pythonsh ${ADT_path}/prepare_receptor4.py -r "${pdb_path}" -o "${prep_path}".pdbqt -A checkhydrogens -U nphs_lps_waters_nonstdres

# Checking the return code of prepare_receptor4.py
if [[ $? -ne 0 ]]; then
  echo "prepare_receptor4.py failed to run successfully"
  exit 1
fi


### Splitting PDB structures into separate files
# Splitting PDB structures into separate files
echo -e "Splitting PDB structures into separate files\n"

if [[ "${complex}" == "a" ]]; then
  python split_pdb.py -r "${prep_path}".pdbqt -s all
elif [[ "${complex}" == "l" ]]; then
  python split_pdb.py -r "${prep_path}".pdbqt -s largest
else
  python split_pdb.py -r "${prep_path}".pdbqt -s mains

# Checking the return code of split_pdb.py
if [[ $? -ne 0 ]]; then
  echo "split_pdb.py failed to run successfully"
  exit 1
fi

### Running prep_conf.py
# This will create config file for AutoDock Vina it will also get 
# binding site coordinates if PDB file contains ligand.
echo -e "Running prep_conf.py\n"

python prep_conf.py -r "${prep_path}".pdbqt -l "${prep_path}"_ligand.pdbqt
