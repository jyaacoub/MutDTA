# %%
import os
import submitit

from src.utils import config as cfg  # sets up env vars
from src.utils.arg_parse import parse_train_test_args
from src.train_test import dtrain

args, unknown_args = parse_train_test_args(verbose=True, distributed=True,
                             jyp_args=' -odir ./slurm_out_DDP/%j' +
                             ' -m EDI -d PDBbind -f nomsa -e anm' +
                             ' -lr 0.00001 -bs 16 -do 0.2 --train' +
                             ' -s_t 4320 -s_m 18GB -s_nn 1 -s_ng 4 -s_cp 4')
# 3days == 4320 mins
# %% PARSE ARGS

os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

# Model name and dataset cannot be added since we can provide a list of them
args.output_dir += f'_{"-".join(args.model_opt)}_{"-".join(args.data_opt)}{args.fold_selection or ""}_'+\
                   f'{"-".join(args.edge_opt)}_{args.learning_rate}_{args.batch_size*args.slurm_nnodes*args.slurm_ngpus}'
print("out_dir:", args.output_dir)

# %% SETUP SLURM EXECUTOR
executor = submitit.AutoExecutor(folder=args.output_dir, 
                                 slurm_max_num_timeout=0) # no requeuing if TIMEOUT or preemption

additional_params = { # slurm params not accepted by submitit
        'mail-type': 'ALL',
        'mail-user': 'j.yaacoub@mail.utoronto.ca',
        # 'dependency': 'afterok:10275728',
    }
if args.slurm_nodelist: # Rescrict to certain nodes
    additional_params["nodelist"] = args.slurm_nodelist

executor.update_parameters(
    cpus_per_task=args.slurm_cpus_per_task, # needs to be at least 2 for dataloader.workers == 2
    timeout_min=args.slurm_time, # SBATCH "-time" param
    
    nodes=args.slurm_nnodes,
    slurm_gpus_per_node=f'{cfg.SLURM_GPU_NAME}:{args.slurm_ngpus}',
    tasks_per_node=args.slurm_ngpus,
    
    slurm_job_name=f'DDP-{args.model_opt[0]}_{args.data_opt[0]}{args.fold_selection or ""}' +
                    f'_{args.slurm_ngpus*args.slurm_nnodes}{cfg.SLURM_GPU_NAME}',
    slurm_partition=cfg.SLURM_PARTITION,         
    slurm_account=cfg.SLURM_ACCOUNT,
    slurm_mem=args.slurm_mem,
    
    # Might need this since ESM takes up a lot of memory
    #    slurm_constraint=cfg.SLURM_CONSTRAINT, 
    # For CC docs see: (https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm)
    
    slurm_additional_parameters=additional_params
)

#%% submit job:
job = executor.submit(dtrain, args, unknown_args)
print(f"Submitted job_id: {job.job_id}")

# %%
