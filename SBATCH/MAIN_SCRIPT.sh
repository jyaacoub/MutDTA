#!/bin/bash
#SBATCH -t 15
#SBATCH --job-name=AlphaFlow_inference
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=./%x_%A.out
# THIS SCRIPT RUNS MSA INPUTS THROUGH ALPHAFLOW AND INFERENCE ON PRETRAINED MODELS
# REQUIRES:
#   - MSA dir full of relevant .a3m files for alphaflow 
#   - ligand sdf file

ROOT_DIR="/lustre06/project/6069023"
proj_dir="${ROOT_DIR}/jyaacoub/" # contains the relevant alphaflow/ and MutDTA/ repos with their venvs
MSA_DIRECTORY_PATH="${proj_dir}/MutDTASBATCH/samples/input/alphaflow/msa_input"
# MSA is needed for input so that we can run it through alphaflow to get our predictions
# MSA input format must match the following depending on its extension:
#   1. ".aln" - purely just protein sequences (each line is a protein sequence) with the first line being the target.
#   2. ".a3m" - Protein sequences on even line numbers (normal msa format with '>' preceeding each protein sequence) 
#       with the second line being the target sequence - no blank lines allowed
#
# NOTE: this will run all msas (files ending in .a3m or .aln) in this directory through alphaflow

OUT_DIR="${ROOT_DIR}/MutDTA_outputs/" # will populate alphaflow_io and results csv file
SBATCH_OUTS="${OUT_DIR}/SBATCH_OUTS/"
LIGAND_SDF_FILE="${proj_dir}/MutDTA/SBATCH/samples/input/inference/VWW_ideal.sdf" # SDF file is needed for GVPL features
MODEL_opt="PDBbind_gvpl_aflow"
# select from one of the following:
#   davis_DG, davis_gvpl, davis_esm, 
#   kiba_DG, kiba_esm, kiba_gvpl, 
#   PDBbind_DG, PDBbind_esm, PDBbind_gvpl, PDBbind_gvpl_aflow

N_CONFORMATIONS=10 # number of conformations to generate by alphaflow
ALPHAFLOW_TIME="3:00:00" # for 1 protein it should not take longer than an hour or so

mkdir -p ${SBATCH_OUTS}

###########################################
# ALPHAFLOW STEP:
###########################################
io_dir="${OUT_DIR}/alphaflow_io/" #PATH TO INPUT_CSVs DIR

# Check if Alphaflow output already exists
if [ -d "${io_dir}/out_pdb" ] && [ "$(ls -A ${io_dir}/out_pdb)" ]; then
    echo "Alphaflow output already exists, skipping Alphaflow preparation and submission steps."
    skip_alphaflow=true
else
    skip_alphaflow=false
fi

if [ "$skip_alphaflow" = false ]; then
    echo "Loading up Python virtual environment for Alphaflow prepare input."
    cd "${proj_dir}/alphaflow"
    source ".venv-ds/bin/activate"

    echo "preparing msa inputs"
    python -u prepare_input.py $MSA_DIRECTORY_PATH $io_dir

    if [[ $? -ne 0 ]]; then
        echo -e "\nERROR: non-zero exit for prepare_input.py"
    fi;

    alphaflow_job_id=$(sbatch << EOF | awk '{print $4}'
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

else
    echo "Proceeding directly to the inference step."
    alphaflow_job_id=""
fi


###########################################
# INFERENCE STEP:
###########################################

echo "Submitting second job for running through the selected model."

if [ -z "$alphaflow_job_id" ]; then
    dependency_line=""
else
    dependency_line="#SBATCH --dependency=afterok:${alphaflow_job_id}"
fi

SBATCH << EOF
#!/bin/bash
#SBATCH -t 30:00
#SBATCH --job-name=run_mutagenesis_davis_esm
#SBATCH --mem=10G

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4

#SBATCH --output=${SBATCH_OUTS}/%x_%a.out
#SBATCH --array=0-4
${dependency_line}

CSV_OUT="${OUT_DIR}/inference_test.csv" # this is used for outputs

module load StdEnv/2020 && module load gcc/9.3.0 && module load arrow/12.0.1
cd ${proj_dir}/MutDTA
source .venv/bin/activate

python -u inference.py \
            --ligand_sdf ${LIGAND_SDF_FILE} \
            --pdb_files ${io_dir}/out_pdb/*.pdb \
            --csv_out ${CSV_OUT} \
            --model_opt ${MODEL_OPT} \
            --fold \${SLURM_ARRAY_TASK_ID} \
            --ligand_id "1a30_ligand" \
            --batch_size 8 
EOF