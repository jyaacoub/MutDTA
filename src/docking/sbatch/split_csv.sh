#!/bin/bash
#!/bin/bash

# Argument help message
display_help() {
    echo "Usage: ./split_csv.sh [-h] input_file output_dir num_partitions"
    echo "Split a CSV file into multiple partitions."
    echo
    echo "Arguments:"
    echo "  -h               Ignore the header row in the input file (optional)."
    echo "  input_file       Path to the input CSV file."
    echo "  output_dir       Directory to store the output partitions."
    echo "  num_partitions   Number of partitions to create."
    echo
    echo "Example:"
    echo "  ./split_csv.sh -h input.csv output_partitions 5"
}

# Check the number of arguments
if [[ $# -lt 3 ]]; then
    display_help
    exit 1
fi

# Parse options
ignore_header=false
while getopts "h" opt; do
    case $opt in
        h)
            ignore_header=true
            ;;
        \?)
            display_help
            exit 1
            ;;
    esac
done

# Shift the option arguments
shift $((OPTIND - 1))

# Retrieve the remaining arguments
input_file=$1
output_dir=$2
num_partitions=$3

# Check if the input file exists
if [[ ! -f $input_file ]]; then
    echo "Input file '$input_file' does not exist."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Determine the starting line for partitioning
start_line=1
if $ignore_header; then
    start_line=2
fi


###### SPLITTING
# Calculate the number of lines in the input file
num_lines=$(wc -l < "$input_file")

# Calculate the number of lines per partition
lines_per_partition=$((num_lines / num_partitions))

# Split the input file into partitions
split --numeric-suffixes=1 -l "$lines_per_partition" -a 1 -d --additional-suffix=.csv --skip= "$input_file" "$output_dir/partition"

# Rename the partitions
partition_index=1
for partition_file in "$output_dir/partition"*
do
    mv "$partition_file" "$output_dir/partition_${partition_index}.csv"
    partition_index=$((partition_index + 1))
done

echo "CSV file split into $num_partitions partitions in the '$output_dir' directory."
