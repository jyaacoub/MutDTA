#!/bin/bash

echo -e "\n### Starting ###\n"
# Check if the required arguments are provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <path> [<complex>] <ADT_path>"
    echo -e "\t path - path to the directory containing the pdb files in directories '<pdbcode>/<pdbcode>.pdb'"
    echo -e "\t complex - (optional) 'a' if you want all structures extracted, 'l' if you only want the largest structure (receptor). Default is 'm', to extract only receptor (largest structure) and its ligand (closest structure to receptor)."
    echo -e "\t ADT_path - path to MGLToolsPckgs/AutoDockTools/Utilities24/"
    exit 1
fi

force=1 # 1 for true

# e.g.:
# ADT_path="/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/"
# ~/lib/mgltools/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/
# path="test_prep/"

# Assign the arguments to variables
path=$1
ADT_path=$2

# Check if the 'complex' argument was provided, otherwise set a default value
if [ $# -eq 4 ]; then
    complex=$3
    ADT_path=$4
else
    complex="m"
fi

# Delete exisiting prep files
# rm -rf ./data/structures/*/prep

# Use the find command to list all directories within the path
directories=$(find "$path" -mindepth 1 -maxdepth 1 -type d)

# Loop through each directory
for dir in $directories; do
    # Process each directory
    pdbcode=$(basename "$dir")
    echo -e "\n\t### ${pdbcode}"

    prep_directories=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -name "prep")
    if [ -n "$prep_directories" ]; then
        echo -e "\t### 'prep' directory already exists. Skipping..."
    else
        bash prep_pdb.sh "${path}/${pdbcode}" "${pdbcode}" "${complex}" "${ADT_path}"
        # Checking the return code of prep_conf.py
        if [[ $? -ne 0 ]]; then
            echo -e "\nERROR: prep_pdb.sh failed to run successfully"
            if [[ $force -ne 1 ]]; then
                exit 1
            fi
        fi
    fi
done

echo -e "\n### DONE ###\n"