#!/bin/bash

ROOT_DIR="/lustre06/project/6069023"
OUT_DIR="${ROOT_DIR}/MutDTA_outputs/"
SBATCH_OUTS="${OUT_DIR}/SBATCH_OUTS/"
proj_dir="/lustre06/project/6069023/jyaacoub/" # contains the relevant alphaflow/ and MutDTA/ repos with their venvs


N_CONFORMATIONS=10 # number of conformations to generate by alphaflow
MSA_DIRECTORY_PATH=""
# MSA is needed for input so that we can run it through alphaflow to get our predictions
# MSA input format must match the following depending on its extension:
#   1. ".aln" - purely just protein sequences (each line is a protein sequence) with the first line being the target.
#   2. ".a3m" - Protein sequences on even line numbers (normal msa format with '>' preceeding each protein sequence) 
#       with the second line being the target sequence - no blank lines allowed
#
# NOTE: this will run all msas (files ending in .a3m or .aln) in this directory through alphaflow
ALPHAFLOW_TIME="3:00:00" # for 1 protein it should not take longer than an hour or so

mkdir -p ${SBATCH_OUTS}

######
# TODO: check if alphaflow has already generated these files and skip alphaflow prepare/submission step
######
io_dir="${OUT_DIR}/alphaflow_io/" #PATH TO INPUT_CSVs DIR

echo "loading up python venv"
cd "${proj_dir}/alphaflow"
source ".venv-ds/bin/activate"

echo "preparing msa inputs"
python -u prepare_input.py $MSA_DIRECTORY_PATH $io_dir

if [[ $? -ne 0 ]]; then
    echo -e "\nERROR: non-zero exit for prepare_input.py"
fi;


JOB_ID=$(sbatch << EOF | awk '{print $4}'
#!/bin/bash
#SBATCH -t ${ALPHAFLOW_TIME}
#SBATCH --job-name=AlphaFlow

#SBATCH --gpus-per-node=a100:1 # If running into CUDA MEM errors increase this to 2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=${SBATCH_OUTS}/alphaflow/%x_%a.out

cd "${proj_dir}/alphaflow"
source ".venv-ds/bin/activate"

#                                   For LMA use only, defaults are:
export DEFAULT_LMA_Q_CHUNK_SIZE=512     # 1024
export DEFAULT_LMA_KV_CHUNK_SIZE=2048   # 4096

export CUDA_LAUNCH_BLOCKING=1

# Determine the number of GPUs available
world_size=\$(nvidia-smi --list-gpus | wc -l)

# runs deepspeed if there are two or more GPUs requested
run_command() {
    if [[ \$world_size -ge 2 ]]; then
        deepspeed --num_gpus \$world_size predict_deepspeed.py --world_size \$world_size \$@
    else
        python predict.py \$@
    fi
}

run_command --mode alphafold \
            --input_csv "${io_dir}/input.csv" \
            --msa_dir "${io_dir}/msa_dir" \
            --weights "weights/alphaflow_md_distilled_202402.pt" \
            --outpdb "${io_dir}/out_pdb" \
            --samples ${N_CONFORMATIONS} \
            --no_overwrite \
            --no_diffusion \
            --noisy_first
EOF
)

echo Submitted job ${JOB_ID} for alphaflow conformations


echo Will now submit second job for running through model of choice
# TODO: create inference.py file for running given a alphaflow file and a ligand.
# alphaflow files should come from "${io_dir}/out_pdb" and protein ids from the file name
# protein sequences can be gotten from that same pdb or using the msa at ${io_dir}/msa_dir
