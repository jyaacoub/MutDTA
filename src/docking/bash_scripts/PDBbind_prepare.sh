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
#>>>>>>>>>>>>>>>>> ARG PARSING >>>>>>>>>>>>>>>>>>>>>
# Function to display script usage
function usage {
  echo "Usage: $0 path ADT_path template [OPTIONS]"
  echo "       path     - path to PDBbind dir containing pdb for protein to convert to pdbqt (ABSOLUTE PATH)."
  echo "       ADT_path - path to MGL root  (e.g.: '~/mgltools_x86_64Linux2_1.5.7/')"
  echo "       template - path to conf template file (create empty file if you want vina defaults)."
  echo "Options:"
  echo "       -sl --shortlist: path to csv file containing a list of pdbcodes to process."
  echo "              Doesn't matter what the file is as long as the first column contains the pdbcodes."
  echo "       -cd --config-dir: path to store new configurations in."
  echo "              Default is to store it with the prepared receptor as <PDBCode>_conf.txt"
  exit 1
}

# Check the number of arguments
if [[ $# -lt 3 ]]; then
  usage
fi

# Assign the arguments to variables
path="$1"
ADT_path="$2"
template="$3"
shortlist=""
config_dir=""

# Parse the options
while [[ $# -gt 3 ]]; do
  key="$4"
  case $key in
    -sl|--shortlist)
      shortlist="$5"
      shift 2
      ;;
    -cd|--config-dir)
      config_dir="$5"
      shift 2
      ;;
    *)
      echo "Unknown option: $key"
      usage
      ;;
  esac
done

# Print the parsed arguments
echo "Path: $path"
echo "ADT Path: $ADT_path"
echo "Template: $template"
if [[ -n "$shortlist" ]]; then
  echo "Shortlist: $shortlist"
fi
if [[ -n "$config_dir" ]]; then
  echo "Config Dir: $config_dir"
fi
exit 0
#<<<<<<<<<<<<<<<<< ARG PARSING <<<<<<<<<<<<<<<<<<<<<

#<<<<<<<<<<<<<<<<< PRE-RUN CHECKS >>>>>>>>>>>>>>>>>>
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

# Checking if shortlist file exists
#   [not empty] and [not a file]
if [[ ! -z "$shortlist" ]] && [[ ! -f "$shortlist" ]]; then
  echo "shortlist file does not exist: ${shortlist}"
  exit 1
fi

# NOTE: change this if you run from a diff dir Check to see if ../prep_conf.py file exists
if [[ ! -f "../prep_conf.py" ]]; then
  echo "../prep_conf.py does not exist, make sure to run this script from the src/docking/bash_scripts/PDBbind dir."
  exit 1
fi

if [[ ! -z "$shortlist" ]]; then
  # if shortlist is provided, use it
  echo "Using shortlist file: ${shortlist}"
  # getting all the pdbcodes from the shortlist file
  codes=$(awk -F',' 'NR>1 {print $1}' "$shortlist")

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
#<<<<<<<<<<<<<<<<< PRE-RUN CHECKS <<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>> MAIN LOOP >>>>>>>>>>>>>>>>>>>>>
count=0
errors=0
# reset pdb_error.txt
echo "" > pdb_error.txt
for dir in $dirs; do
  code=$(basename "$dir")
  echo -e "Processing $code \t: $((++count)) / $total \t: $((errors)) errors"

  # skipping if already processed
  if [ -f "${dir}/${code}_conf.txt" ]; then
    echo -e "\t Skipping...already processed"
    continue
  fi

  # running prepare_receptor4.py to convert protein to pdbqt
  protein="${dir}/${code}_protein.pdb"
  "${2}/bin/pythonsh" "${ADT_path}/prepare_receptor4.py" -r $protein -o "${dir}/${code}_protein.pdbqt" -A checkhydrogens -U nphs_lps_waters_nonstdres

  #checking error code
  if [ $? -ne 0 ]; then
    echo "prepare_receptor4.py failed to convert protein to pdbqt for $code"
    # saving code to error file
    echo "$code" >> pdb_error.txt
    ((errors++))
    # skip this code
    continue
  fi

  # running obabel to convert ligand to pdbqt
  ligand="${dir}/${code}_ligand.sdf"
  obabel -isdf $ligand --title $code -opdbqt -O "${dir}/${code}_ligand.pdbqt"

  #checking error code
  if [ $? -ne 0 ]; then
    echo "obabel failed to convert ligand to pdbqt for $code"
    exit 1
  fi

  # new protien and lig files
  protein="${dir}/${code}_protein.pdbqt"
  ligand="${dir}/${code}_ligand.pdbqt"

  # preparing config file with binding site info
  pocket="${dir}/${code}_pocket.pdb"
  python ../prep_conf.py -r $protein -l $ligand -pp $pocket -o "${dir}/${code}_conf.txt"

  #checking error code
  if [ $? -ne 0 ]; then
    echo "prep_conf.py failed to prepare config file for $code"
    exit 1
  fi
done
