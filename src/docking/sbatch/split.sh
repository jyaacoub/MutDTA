
# Argument help message
display_help() {
    echo "Usage: ./split_csv.sh [-h] input_file output_dir num_partitions"
    echo "Split a CSV file into multiple partitions."
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

# Calculate the number of lines in the input file
num_lines=$(wc -l < "$input_file")

# Calculate the number of lines per partition
lines_per_partition=$((num_lines / num_partitions))

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Split the input file into partitions
split -l "$lines_per_partition" $input_file $output_path

# Rename the partitions
partition_index=0
for partition_file in ${output_path}/*
do
    echo -e "${partition_file} \t--> ${output_path}/$((partition_index++)).csv"
    mv "$partition_file" "${output_path}/$((partition_index++)).csv"
done

if $ignore_header; then
    # Remove the header from first partition
    tail -n +2 "${output_path}/0.csv" > "${output_path}/0.temp" && mv "${output_path}/0.tmp" "${output_path}/0.csv"
fi