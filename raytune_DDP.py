# This is a simple tuning script for the raytune library.
# no support for distributed training in this file.

import random, os, socket, time
import torch

from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import ray
from ray import tune, train
from ray.air import session # this session just comes from train._internal.session._session
from ray.train.torch import TorchCheckpoint
from ray.tune.search.optuna import OptunaSearch


from src.utils.loader import Loader
from src.train_test.simple import simple_train, simple_eval
from src.utils import config as cfg

def main(rank, world_size, config): # define this inside objective??
    # ============= Set up DDP environment =====================
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = config['port']
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    device = torch.device(rank)
    
    # ============ Load up distributed training data ==============
    p_grp = session.get_trial_resources() # returns placement group object from TrialInfo
    # first item is resource list, in our case resources are the same across all trials 
    # so it is safe to just take the first from the list to get our gpu count
    trial_resources = p_grp._bound.args[0][0]
    ncpus = trial_resources['cpu']
    
    local_bs = config['global_batch_size']/world_size
    if not local_bs.is_integer():
        print(f'WARNING: batch size is not divisible by world size. Local batch size is {local_bs}.')
    
    local_bs = int(local_bs)
    
    loaders = Loader.load_distributed_DataLoaders(
        num_replicas=world_size, rank=rank, seed=42, # DDP specific params
        
        data=config['dataset'], 
        pro_feature=config['feature_opt'], 
        edge_opt=config['edge_opt'], 
        batch_train=local_bs,   # global_bs/world_size
        datasets=['train', 'val'], 
        training_fold=config['fold_selection'],
        num_workers=ncpus, # number of subproc used for data loading
    )
    
    # ============ Init Model ==============
    model = Loader.init_model(model=config["model"], pro_feature=config["feature_opt"],
                            pro_edge=config["edge_opt"], dropout=config["dropout"]
                            ).to(device)
        
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    torch.distributed.barrier() # Sync params across GPUs
    
    # ============ Train Model for n epochs ============
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    save_checkpoint = config.get("save_checkpoint", False)
    
    for _ in range(config["epochs"]):
        torch.distributed.barrier()
        simple_train(model, optimizer, loaders['train'], 
                    device=device, 
                    epochs=1)  # one epoch
        
        torch.distributed.barrier()
        loss = simple_eval(model, loaders['val'], device)  # Compute validation accuracy
        
        checkpoint = None
        if save_checkpoint and rank == 0:
            checkpoint = TorchCheckpoint.from_state_dict(model.state_dict())
            
        # Report metrics (and possibly a checkpoint) to Tune
        session.report({"mean_loss": loss}, checkpoint=checkpoint)
        
    destroy_process_group()


def objective_DDP(config): # NO inter-node distribution due to communication difficulties
    world_size = torch.cuda.device_count()
    # device is also inserted as the first arg to main()
    print(f'World size: {world_size}')
    mp.spawn(main, args=(world_size, config,), nprocs=world_size)
    
    
if __name__ == "__main__":    
    search_space = {
        # constants:
        "epochs": 10, # 15 epochs
        "model": "DG",
        "dataset": "davis",
        "feature_opt": "nomsa",
        "edge_opt": "binary",
        "fold_selection": 0,
        "save_checkpoint": False,
        
        # DDP specific constants:
        "port": random.randint(49152,65535),
        
        # hyperparameters to tune:
        "lr": tune.loguniform(1e-4, 1e-2),
        "dropout": tune.uniform(0, 0.5),
        "embedding_dim": tune.choice([64, 128, 256]),
        
        "global_batch_size": tune.choice([16, 32, 48]), # global batch size is divided by ngpus/world_size
    }

    ray.init(num_gpus=1, num_cpus=8, ignore_reinit_error=True)

tuner = tune.Tuner(
    tune.with_resources(objective_DDP, resources={"cpu": 6, "gpu": 2}),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=50,
    ),
)

results = tuner.fit()
