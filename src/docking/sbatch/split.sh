# Retrieve args
input_file=$1
output_path=$2
num_partitions=$3

# Calculate the number of lines in the input file
num_lines=$(wc -l < "$input_file")

# Calculate the number of lines per partition
lines_per_partition=$((num_lines / num_partitions))

# Split the input file into partitions
split -l "$lines_per_partition" $input_file $output_path
