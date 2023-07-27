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
  echo "Usage: $0 path ADT template [OPTIONS]"
  echo "       path     - path to PDBbind dir containing pdb for protein to convert to pdbqt."
  echo "       ADT - path to MGL root  (e.g.: '~/mgltools_x86_64Linux2_1.5.7/')"
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

# Check the number of arguments
if [[ $# -lt 3 ]]; then
  usage
fi

bash_scripts=$(dirname $(realpath "$0"))
prep_confpy=${bash_scripts}/../python_helpers/prep_conf.py
get_flexpy=${bash_scripts}/../python_helpers/get_flexible.py


# Assign the arguments to variables
PDBbind=$(realpath $1)
ADT="$2"
prep_receptor="${ADT}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
prep_flexreceptor="${ADT}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_flexreceptor4.py"
template="$3"
shortlist=""
config_dir=""
flexible=false

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
echo "ARGS"
echo -e "\tPath: $PDBbind"
echo -e "\tADT Path: $ADT"
echo -e "\tTemplate: $template"
echo "OPTIONAL ARGS:"
if [[ -n "$shortlist" ]]; then
  echo -e "\tShortlist: $shortlist"
fi
if [[ -n "$config_dir" ]]; then
  echo -e "\tConfig Dir: $config_dir"
fi
echo -e "\tFlexible: $flexible\n"
exit

#<<<<<<<<<<<<<<<<< ARG PARSING <<<<<<<<<<<<<<<<<<<<<

#<<<<<<<<<<<<<<<<< PRE-RUN CHECKS >>>>>>>>>>>>>>>>>>
# Checking if obabel command is available (install from https://openbabel.org/wiki/Category:Installation)
if ! command -v obabel >/dev/null 2>&1; then
  echo "obabel is not installed or not in the system PATH"
  exit 1
fi

# Checking if pythonsh command is available (part of MGLTools)
if [[ ! -f "${ADT}/bin/pythonsh" ]]; then
  echo "pythonsh is not installed or not in the system PATH"
  echo -e "\t ${ADT}/bin/pythonsh"
  exit 1
fi

# Checking if prepare_receptor4.py exists (part of MGLTools)
if [[ ! -f $prep_receptor ]]; then
  echo "prepare_receptor4.py does not exist in the specified location: $prep_receptor"
  exit 1
fi

# Checking if prepare_flexreceptor4.py exists (part of MGLTools)
if [[ ! -f $prep_flexreceptor ]]; then
  echo "prepare_flexreceptor4.py does not exist in the specified location: $prep_flexreceptor"
  exit 1
fi

# NOTE: change this if you run from a diff dir Check to see if ../prep_conf.py file exists
if [[ ! -f $prep_confpy ]]; then
  echo "prep_conf.py does not exist (${prep_confpy}). Make sure to run this at src/docking/bash_scripts/."
  exit 1
fi

if [[ ! -f $get_flexpy ]]; then
  echo "get_flexible.py does not exist (${get_flexpy}). Make sure to run this at src/docking/bash_scripts/."
  exit 1
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
# divider for errors
echo -e "\n------------------------------------------------------------\n" >> prep_pdb_error.txt
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

  # running prepare_receptor4.py to convert protein to pdbqt
  protein_p="${dir}/${code}_protein"

  # Clean PDB from ions, waters etc...
  grep ^ATOM "${protein_p}.pdb" > "${protein_p}_temp.pdb"
  # Adding charges, etc...
  "${ADT}/bin/pythonsh" $prep_receptor -r "${protein_p}_temp.pdb" -o "${protein_p}.pdbqt" -A checkhydrogens -U nphs_lps_waters_nonstdres

  rm -fv "${protein_p}_temp.pdb"

  # Checking error code
  if [ $? -ne 0 ]; then
    echo "prepare_receptor4.py failed to convert protein to pdbqt for $code"
    # saving code to error file
    echo "$code" >> pdb_error_NEW.txt
    ((errors++))
    # skip this code
    exit 1
  fi

  # running flex receptor if toggled
  if $flexible; then
    # getting flexible residues to pass into prepare_flexreceptor4.py as:
    # -s     specification for flex residues
    #             Use underscores to separate residue names:
    #               ARG8_ILE84  
    #             Use commas to separate 'full names' which uniquely identify residues:
    #               hsg1:A:ARG8_ILE84,hsg1:B:THR4 
    #             [syntax is molname:chainid:resname]
    
    flex_res=$(python $get_flexpy -pf "${dir}/${code}_pocket.pdb")
    "${ADT}/bin/pythonsh" $prep_flexreceptor -r "${protein_p}.pdbqt" -g "${protein_p}_rigid.pdbqt" -x "${protein_p}_flex.pdbqt" -s $flex_res 
    # -g and -x are output files for rigid and flexible parts of receptor
    
    # Checking error code
    if [ $? -ne 0 ]; then
      echo "prepare_flexreceptor4.py failed to prepare receptor for $code"
      # saving code to error file
      echo "$code" >> prep_pdb_error.txt
      ((errors++))
      # skip this code
      exit 1
    fi
  fi


  # running obabel to convert ligand sdf to pdbqt
  ligand_p="${dir}/${code}_ligand"
  obabel -isdf "${ligand_p}.sdf" --title $code -opdbqt -O "${ligand_p}.pdbqt"

  #checking error code
  if [ $? -ne 0 ]; then
    echo "obabel failed to convert ligand to pdbqt for $code"
    exit 1
  fi


  # new protein and lig files
  protein="${protein_p}.pdbqt"
  ligand="${ligand_p}.pdbqt"

  # preparing config file with binding site info
  pocket="${dir}/${code}_pocket.pdb"

  # Checking to make sure that the files exist
  if [[ ! -f "$protein" || ! -f "$ligand" || ! -f "$pocket" ]]; then
    echo "Error: One or more prep files not found for $code"
    ((errors++))
    #exit 1
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
