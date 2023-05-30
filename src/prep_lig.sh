# this assumes you have already set up openbabel
echo -e "\n### Starting ###\n"
# Check if the required arguments are provided
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <path> [<complex>] <ADT_path>"
    echo -e "\t path - path to csv file containing ligand_name and SMILE string you want to convert to pdbqt"
    echo -e "\t output (optional) - path to output directory to place pdbqt files. Default is same directory as csv file."
    exit 1
fi

# e.g. use: prep_lig.sh data/PDBbind/kd_ki/unique_lig.csv data/structures/ligands/


# csv file contains ligand_name and SMILE string
csv_file=$1
if [ $# -eq 2 ]; then
    output=$2
else
    output=$(dirname "$csv_file")
fi

# Skip the first line (header) and process each subsequent line
# "-n +2" -> start at line 2
# 
tail "$csv_file" | while IFS=',' read -r lig_name smile; do
    # Print the ligand_name and SMILE string
    echo "Ligand Name: $lig_name"
    echo "SMILE: $smile"
    echo "------------------------"
done

# END GOAL IS TO LOOP THROUGH 
# obabel -:"<SMILES>" --gen3d -opdbqt -O <path>/<ligand_name>.pdbqt