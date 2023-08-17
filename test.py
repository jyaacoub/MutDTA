# %%
import random, argparse, os
import submitit

import config
from src.distributed_train.utils import dtrain

#example call:
# python train.py -slurm_nnodes 2 -slurm_ngpus 8 
# %% PARSE ARGS
args = {
    'port': random.randint(49152,65535),
    'output_dir': "/cluster/home/t122995uhn/projects/MutDTA/slurm_tests/DDP/%j",
    'slurm_ngpus': 3,
    'slurm_nnodes': 1,
    'slurm_nodelist': None,
    'slurm_time': 2800,
    'slurm_cpus_per_task': 2
}
n_gpu = args['slurm_ngpus']
os.makedirs(os.path.dirname(args['output_dir']), exist_ok=True)

# %% SETUP SLURM EXECUTOR
executor = submitit.AutoExecutor(folder=args['output_dir'], 
                                 slurm_max_num_timeout=0) # no requeuing if TIMEOUT or preemption

executor.update_parameters(
    cpus_per_task=args['slurm_cpus_per_task'], # needs to be at least 2 for dataloader.workers == 2
    timeout_min=args['slurm_time'], # SBATCH "-time" param
    
    nodes=args['slurm_nnodes'],
    slurm_gpus_per_node=f'v100:{n_gpu}',
    tasks_per_node=n_gpu,
    
    slurm_job_name=f'Test_gpu_{n_gpu}',
    slurm_partition="gpu",
    slurm_account='kumargroup_gpu',
    slurm_mem='15GB',
    # Might need this since ESM takes up a lot of memory
    # slurm_constraint='gpu32g', # using small batch size will be sufficient for now
    # v100-34G can handle batch size of 15 -> v100-16G == 7?
)

if args['slurm_nodelist']: # Rescrict to certain nodes
    executor.update_parameters(
        slurm_additional_parameters={"nodelist": f'{args["slurm_nodelist"]}' })

#%% submit job:
job = executor.submit(dtrain, args)
print(f"Submitted job_id: {job.job_id}")

# %%
