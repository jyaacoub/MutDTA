# %%
import os
import submitit

from src.utils import config # sets up env vars
from src.utils.arg_parse import parse_train_test_args
from src.train_test import dtrain

#EDIM_PDBbindD_nomsaF_binaryE_20B_0.0001LR_0.4D_2000E
#DGM_kibaD_shannonF_binaryE_64B_0.0001LR_0.4D_2000E
args = parse_train_test_args(verbose=True, distributed=True,
            jyp_args=' -odir ./slurm_tests/edge_weights/%j'+ \
                ' -m EDI -d kiba -f nomsa -e binary -lr 0.0001 -bs 10 -do 0.4'+ \
                ' -s_t 4320 -s_m 10GB -s_nn 1 -s_ng 4') # 3days == 4320 mins

# args = parse_train_test_args(verbose=True, distributed=True,
#             jyp_args=' -odir ./slurm_tests/edge_weights/%j'+ \
#                 ' -m DG -d davis -f nomsa -e binary -lr 0.0001 -bs 32 --protein_overlap'+ \
#                 ' -s_t 4320 -s_m 10GB -s_nn 1 -s_ng 2') # 3days == 4320 mins
#-odir ./slurm_tests/edge_weights/%j -m EDI -d PDBbind -f nomsa -e anm -lr 0.0001 -bs 16 -s_t 4320 -s_m 10GB -s_nn 1 -s_ng 4
# %% PARSE ARGS

os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

# Model name and dataset cannot be added since we can provide a list of them
args.output_dir += f'_{"-".join(args.model_opt)}_{"-".join(args.data_opt)}_'+\
                   f'{"-".join(args.edge_opt)}_{args.learning_rate}_{args.batch_size*args.slurm_nnodes*args.slurm_ngpus}'
print("out_dir:", args.output_dir)

# %% SETUP SLURM EXECUTOR
executor = submitit.AutoExecutor(folder=args.output_dir, 
                                 slurm_max_num_timeout=0) # no requeuing if TIMEOUT or preemption

executor.update_parameters(
    cpus_per_task=args.slurm_cpus_per_task, # needs to be at least 2 for dataloader.workers == 2
    timeout_min=args.slurm_time, # SBATCH "-time" param
    
    nodes=args.slurm_nnodes,
    slurm_gpus_per_node=f'v100:{args.slurm_ngpus}',
    tasks_per_node=args.slurm_ngpus,
    
    slurm_job_name=f'DDP-{args.model_opt[0]}_{args.data_opt[0]}_{args.slurm_ngpus*args.slurm_nnodes}v100',
    # slurm_partition="gpu",
    # slurm_account='kumargroup_gpu',
    slurm_mem=args.slurm_mem,
    # Might need this since ESM takes up a lot of memory
    # slurm_constraint='gpu16g', # using small batch size will be sufficient for now
    # slurm_constraint='gpu32g', # using small batch size will be sufficient for now
    # v100-34G can handle batch size of 15 -> v100-16G == 7?
)

if args.slurm_nodelist: # Rescrict to certain nodes
    executor.update_parameters(
        slurm_additional_parameters={"nodelist": f'{args.slurm_nodelist}' })

#%% submit job:
job = executor.submit(dtrain, args)
print(f"Submitted job_id: {job.job_id}")

# %%
