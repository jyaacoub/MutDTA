
# Argument help message
display_help() {
    echo "Usage: ./split_csv.sh [-h] input_file output_dir num_partitions"
    echo "Split a CSV file into multiple partitions named <part#>.csv."
    echo
    echo "Arguments:"
    echo "  -i               Ignore the header row in the input file (optional)."
    echo "  input_file       Path to the input CSV file."
    echo "  output_dir       Directory to store the output partitions."
    echo "  num_partitions   Number of partitions to create."
    echo
    echo "Example:"
    echo "  ./split_csv.sh -h input.csv output_partitions 5"
}

if [[ $# -lt 3 ]] || [[ $# -gt 4 ]]; then
    display_help
    exit 1
fi

# Parse options
ignore_header=false
while getopts "i" opt; do
    case $opt in
        i)
            ignore_header=true
            ;;
        \?)
            display_help
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

input_file=$1
output_path=$2
num_partitions=$3

echo "ignore_header: $ignore_header"
echo "input_file: $input_file"
echo "output_path: $output_path"
echo "num_partitions: $num_partitions"

# Check if the input file exists
if [[ ! -f $input_file ]]; then
    echo "Input file '$input_file' does not exist."
    exit 1
fi


#####
# Calculate the number of lines in the input file
num_lines=$(grep -cve '^\s*$' "$input_file") # -v counts non blank lines

# Calculate the number of lines per partition
lines_per_partition=$((num_lines / num_partitions))

echo "num_lines: $num_lines"
echo "lines_per_partition: $lines_per_partition"

# Create the output directory if it doesn't exist
# mkdir -p "$output_dir"

# Split the input file into partitions
split --suffix-length=2 --additional-suffix='_split_out' -l "$lines_per_partition" $input_file $output_path


#####
# Rename the partitions
partition_index=0
while [ $partition_index -lt $num_partitions ]
do
    # deletes existing file if present
    if [[ -f "${output_path}/$((partition_index)).csv" ]]; then
        rm -vf ${output_path}/$((partition_index)).csv
    fi
    partition_index=$((partition_index + 1))
done

partition_index=0
# only rename files with 2 char names (output from split is 2 char alphabetic: aa, ab, ac, ... 
# followed by '_split_out')
for partition_file in ${output_path}/??_split_out
do
    mv -v "$partition_file" "${output_path}/$((partition_index++)).csv"
done

if $ignore_header; then
    # Remove the header from first partition
    tail -n +2 "${output_path}/0.csv" > "${output_path}/0.tmp" && mv "${output_path}/0.tmp" "${output_path}/0.csv"
fi