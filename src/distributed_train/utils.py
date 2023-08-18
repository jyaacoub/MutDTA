import signal, os, subprocess, time
from pathlib import Path

import submitit
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from src.models.mut_dta import EsmDTA
from src.models.prior_work import DGraphDTA
from src.data_processing.datasets import DavisKibaDataset
from src.models.training import train
from src.models import print_device_info


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
    

# distributed training fn
def dtrain(args):
    # ==== initialize the node ====
    init_node(args)
    
    
    # ==== Set up distributed training environment ====
    init_dist_gpu(args)
    
    DATA = args.data_opt[0] # only one data option for now
    FEATURE = args.feature_opt[0] # only one feature option for now
    EDGEW = args.edge_opt[0] # only one edge option for now
    MODEL = args.model_opt[0] # only one model option for now
    
    BATCH_SIZE = args.batch_size
    DROPOUT = args.dropout
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.num_epochs
    
    print(os.getcwd())
    print(f"----------------- HYPERPARAMETERS -----------------")
    print(f"         Local Batch size: {BATCH_SIZE}")
    print(f"        Global Batch size: {BATCH_SIZE*args.world_size}")
    print(f"            Learning rate: {LEARNING_RATE}")
    print(f"                  Dropout: {DROPOUT}")
    print(f"               Num epochs: {EPOCHS}")
    print(f"              Edge option: {EDGEW}")
    
    print(f'----------------- GPU INFO ------------------------')
    print_device_info(args.gpu)
    
    # ==== Load up training dataset ====
    dataset = DavisKibaDataset(
        save_root=f'../data/DavisKibaDataset/{DATA}_{FEATURE}/',
        data_root=f'../data/{DATA}/',
        aln_dir  =f'../data/{DATA}/aln/',
        cmap_threshold=-0.5, 
        feature_opt=FEATURE
        )


    #TODO: replace this with my version of train_test split to account for prot overlap
    sampler = DistributedSampler(dataset, shuffle=True, 
                                 num_replicas=args.world_size,
                                 rank=args.rank, seed=0)
    train_loader = DataLoader(dataset=dataset, 
                            sampler = sampler,
                            batch_size=BATCH_SIZE, # batch size per gpu (https://stackoverflow.com/questions/73899097/distributed-data-parallel-ddp-batch-size)
                            num_workers = 2, # number of subproc used for data loading
                            pin_memory = True,
                            drop_last = True
                            )
    print(f"Data loaded")
    
    # ==== Load model ====
    num_feat_pro = 54 if 'msa' in FEATURE else 34
    if MODEL == 'DG':
        model = DGraphDTA(num_features_pro=num_feat_pro, 
                          dropout=DROPOUT, edge_weight_opt=EDGEW)
    elif MODEL == 'ED':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320, # only esm features
                        pro_emb_dim=54, # inital embedding size after first GCN layer
                        dropout=DROPOUT,
                        esm_only=True,
                        edge_weight_opt=EDGEW)
    
    # args.gpu is the local rank for this process
    model.cuda(args.gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    
    # ==== train (this was modified from `training.py`):  ====
    CRITERION = torch.nn.MSELoss()
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("starting training:")
    #TODO: add validation data.
    for epoch in range(1, EPOCHS+1):
        START_T = time.time()
        
        # Training loop
        model.train()
        train_loss = 0.0
        for data in train_loader:
            batch_pro = data['protein'].cuda(args.gpu) 
            batch_mol = data['ligand'].cuda(args.gpu) 
            labels = data['y'].reshape(-1,1).cuda(args.gpu) 
            
            # Forward pass
            predictions = model(batch_pro, batch_mol)

            # Compute loss
            loss = CRITERION(predictions, labels)
            train_loss += loss.item()

            # Backward pass and optimization
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

        # Compute average training loss for the epoch
        train_loss /= len(train_loader)
        
        # Print training and validation loss for the epoch
        print(f"Epoch {epoch}/{EPOCHS}: Train Loss: {train_loss:.4f}, Time elapsed: {time.time()-START_T}")
