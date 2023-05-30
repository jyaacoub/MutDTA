# this assumes you have already set up openbabel
# Check if the required arguments are provided
if [ $# -lt 1 ] || [ $# -gt 1 ]; then
    echo "Usage: $0 <path>"
    echo -e "\t path - path to PDBbind dir containing pdbcodes with sdf files for ligands to convert to pdbqt"
    exit 1
fi

echo -e "\n### Starting ###\n"
# e.g. use: prep_lig.sh data/PDBbind/raw/refined-set


PDBbind_dir=$1

# looping through all the pdbcodes in the PDBbind dir select everything except index and readme dir
dirs=$(find "$PDBbind_dir" -mindepth 1 -maxdepth 1 -type d -not -name "index" -not -name "readme")

# loop through each pdbcodes
for dir in $dirs; do
    code=$(basename "$dir")
    sdf="${dir}/${code}_ligand.sdf"

    obabel -isdf $sdf --title $code -opdbqt -O "${dir}/${code}_ligand.pdbqt"
done