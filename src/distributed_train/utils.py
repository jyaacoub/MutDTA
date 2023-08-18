import signal, os, subprocess

import submitit
import numpy as np

import torch
import torch.distributed as dist


def handle_sigusr1(signum, frame):
    # requeues the job
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()    

def init_node(args):    
    args.ngpus_per_node = torch.cuda.device_count()

    # requeue job on SLURM preemption
    signal.signal(signal.SIGUSR1, handle_sigusr1)

    # find the common host name on all nodes
    cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0] # first node is the host
    args.dist_url = f'tcp://{host_name}:{args.port}'

    # distributed parameters
    args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
    args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
    
def init_dist_gpu(args):
    job_env = submitit.JobEnvironment()
    args.gpu = job_env.local_rank
    args.rank = job_env.global_rank

    # PyTorch calls to setup gpus for distributed training
    dist.init_process_group(backend='gloo', init_method=args.dist_url, 
                            world_size=args.world_size, rank=args.rank)
    
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    torch.cuda.set_device(args.gpu)
    # cudnn.benchmark = True # not needed since we include dropout layers
    dist.barrier()

    # # disabling printing if not master process:
    # import builtins as __builtin__
    # builtin_print = __builtin__.print

    # def print(*args, **kwargs):
    #     force = kwargs.pop('force', False)
    #     if (args.rank == 0) or force:
    #         builtin_print(*args, **kwargs)

    # __builtin__.print = print
