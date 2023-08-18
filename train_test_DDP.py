# %%
import os
import submitit

from src.utils import config # sets up env vars
from src.utils.arg_parse import parse_train_test_args
from src.train_test import dtrain

args = parse_train_test_args(verbose=True, distributed=True,
                             # 16 * 4 = 64 batch size
                             jyp_args='-m DG -d davis -f nomsa -e simple -bs 16 ' + \
                                      '-s_t 1200 -s_m 10GB -s_nn 1 -s_ng 4') #slurm args
# %% PARSE ARGS

os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

# %% SETUP SLURM EXECUTOR
executor = submitit.AutoExecutor(folder=args.output_dir, 
                                 slurm_max_num_timeout=0) # no requeuing if TIMEOUT or preemption

executor.update_parameters(
    cpus_per_task=args.slurm_cpus_per_task, # needs to be at least 2 for dataloader.workers == 2
    timeout_min=args.slurm_time, # SBATCH "-time" param
    
    nodes=args.slurm_nnodes,
    slurm_gpus_per_node=f'v100:{args.slurm_ngpus}',
    tasks_per_node=args.slurm_ngpus,
    
    slurm_job_name=f'Test_gpu_{args.slurm_ngpus}',
    slurm_partition="gpu",
    slurm_account='kumargroup_gpu',
    slurm_mem=args.slurm_mem,
    # Might need this since ESM takes up a lot of memory
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
