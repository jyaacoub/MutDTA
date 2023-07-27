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
# Check if the required arguments are provided
function usage {
  echo "Usage: $0 path template [OPTIONS]"
  echo "       path     - path to PDBbind dir containing pdb for protein to convert to pdbqt."
  echo "       template - path to conf template file (create empty file if you want vina defaults)."
  echo "Options:"
  echo "       -sl --shortlist: path to csv file containing a list of pdbcodes to process."
  echo "              Doesn't matter what the file is as long as the first column contains the pdbcodes."
  echo "       -cd --config-dir: path to store new configurations in."
  echo "              Default is to store it with the prepared receptor as <PDBCode>_conf.txt"
  echo "       -f --flex: optional, runs prepare_flexreceptor4.py with all residues in <code>_pocket.pdb"
  echo "              as flexible."
  exit 1
}


if [[ $# -lt 2 ]]; then
  usage
fi
# e.g. use:
# prepare_conf_only.sh /cluster/projects/kumargroup/jean/data/refined-set ./data/vina_conf/run4.conf 
#                       ./data/shortlists/no_err_50/sample.csv ./data/vina_conf/run4/

bash_scripts=$(dirname $(realpath "$0")) # get the directory of this script
# prep_conf file is in the same directory as this script
prep_confpy=${bash_scripts}/../python_helpers/prep_conf.py

PDBbind=$(realpath $1)
template=$2

shortlist=""
config_dir=""
flexible=false # default is not flexible

echo $# ARGS
# Parse the options
while [[ $# -gt 2 ]]; do
  key="$3"
  case $key in
    -sl|--shortlist)
      shortlist="$4"
      shift 2
      ;;
    -cd|--config-dir)
      config_dir="$4"
      shift 2
      ;;
    -f|--flex)
      flexible=true
      shift 1
      ;;
    *)
      echo "Unknown option: $key"
      usage
      ;;
  esac
done

# Print the parsed arguments
echo "$# ARGS"
echo -e "\tPath: $PDBbind"
echo -e "\tTemplate: $template"
echo "OPTIONAL ARGS:"
if [[ -n "$shortlist" ]]; then
  echo -e "\tShortlist: $shortlist"
fi
if [[ -n "$config_dir" ]]; then
  echo -e "\tConfig Dir: $config_dir"
fi
echo -e "\tFlexible: $flexible\n"
#<<<<<<<<<<<<<<<<< ARG PARSING <<<<<<<<<<<<<<<<<<<<<

#<<<<<<<<<<<<<<<<< PRE-RUN CHECKS >>>>>>>>>>>>>>>>>>
# Checking if shortlist file exists
#   [not empty] and [not a file]
if [[ ! -z "$shortlist" ]] && [[ ! -f "$shortlist" ]]; then
  echo "shortlist file does not exist: ${shortlist}"
  exit 1
fi

if [[ ! -z "$shortlist" ]]; then
  # if shortlist is provided, use it
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
    if [[ ! -d "${PDBbind}/${code}" ]]; then
      echo "PDBbind dir (${PDBbind}) does not contain ${code} specified in shortlist file"
      exit 1
    fi
    dirs="${dirs} ${PDBbind}/${code}"
  done
  total=$(echo "$dirs" | wc -w)
else # otherwise use all
  # looping through all the pdbcodes in the PDBbind dir select everything except index and readme dir
  dirs=$(find "$PDBbind" -mindepth 1 -maxdepth 1 -type d -not -name "index" -not -name "readme")
  # getting count of dirs
  total=$(echo "$dirs" | wc -l)
fi
#<<<<<<<<<<<<<<<<< PRE-RUN CHECKS <<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>> MAIN LOOP >>>>>>>>>>>>>>>>>>>>>
count=0
errors=0
for dir in $dirs; do
  code=$(basename "$dir")
  echo -e "Processing $code \t: $((++count)) / $total \t: $((errors)) errors"

  # getting out path for conf file
  if [[ ! -z  $config_dir ]]; then  # if not empty arg
    conf_out="${config_dir}/${code}_conf.txt"
  else
    conf_out="${dir}/${code}_conf.txt"  
  fi

  # skipping if already processed
  if [ -f $conf_out ]; then
    echo -e "\t Skipping...already processed"
    continue
  fi


  # new protein and lig files
  protein="${dir}/${code}_protein.pdbqt"
  ligand="${dir}/${code}_ligand.pdbqt"

  # preparing config file with binding site info
  pocket="${dir}/${code}_pocket.pdb"
  
  # Checking to make sure that the files exist
  if [[ ! -f "$protein" || ! -f "$ligand" || ! -f "$pocket" ]]; then
    echo "Error: One or more prep files not found for $code"
    ((errors++))
    continue
  fi

  if $flexible; then
    # -f flag for flexible receptor
    python $prep_confpy -r $protein -l $ligand -pp $pocket -o $conf_out -f
  else
    python $prep_confpy -r $protein -l $ligand -pp $pocket -o $conf_out
  fi

  #checking error code
  if [ $? -ne 0 ]; then
    echo "prep_conf.py failed to prepare config file for $code"
    exit 1
  fi
done
