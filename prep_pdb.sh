#!/bin/bash
#TODO: change ADT_path, pdbcode, path to be arguments
ADT_path="/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/"
path="test_prep/"
pdbcode="1a1e"

pdb_path="${path}/${pdbcode}.pdb"
prep_path="${path}/prep/${pdbcode}"

# Checking to see that file exists
if [[ ! -f "${pdb_path}" ]]; then
  echo "File ${pdb_path} does not exist"
  exit 1
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

# note that we do not use -e flag so that we can also 
# extract ligand if it is present
pythonsh ${ADT_path}/prepare_receptor4.py -r "${pdb_path}" -o "${prep_path}" -A checkhydrogens -U nphs_lps_waters_nonstdres
