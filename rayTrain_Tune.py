# This is a simple tuning script for the raytune library.
# no support for distributed training in this file.

import random
import os
import tempfile

import torch

import ray
from ray.air import session # this session just comes from train._internal.session._session
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchCheckpoint, TorchTrainer
from ray.tune.search.optuna import OptunaSearch


from src.utils.loader import Loader
from src.train_test.simple import simple_train, simple_eval
from src.utils import config as cfg

def train_func(config):
    # ============ Init Model ==============
    model = Loader.init_model(model=config["model"], pro_feature=config["feature_opt"],
                            pro_edge=config["edge_opt"],
                            # additional kwargs send to model class to handle
                            dropout=config["dropout"], 
                            dropout_prot=config["dropout_prot"],
                            pro_emb_dim=config["pro_emb_dim"],
                            )
    
    # prepare model with rayTrain (moves it to correct device and wraps it in DDP)
    model = ray.train.torch.prepare_model(model)
    
    # ============ Load dataset ==============
    print("Loading Dataset")
    loaders = Loader.load_DataLoaders(data=config['dataset'], pro_feature=config['feature_opt'], 
                                      edge_opt=config['edge_opt'], 
                                      path=cfg.DATA_ROOT, 
                                      batch_train=config['batch_size'],
                                      datasets=['train', 'val'],
                                      training_fold=config['fold_selection'])
    
    # prepare dataloaders with rayTrain (adds DistributedSampler and moves to correct device)
    for k in loaders.keys():
        loaders[k] = ray.train.torch.prepare_data_loader(loaders[k])
    
    
    # ============= Simple training and eval loop =====================
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    save_checkpoint = config.get("save_checkpoint", False)
    
    for _ in range(config['epochs']):
        
        # NOTE: no need to pass in device, rayTrain will handle that for us
        simple_train(model, optimizer, loaders['train'], epochs=1)  # Train the model
        loss = simple_eval(model, loaders['val'])  # Compute test accuracy
        
            
        # Report metrics (and possibly a checkpoint) to ray        
        checkpoint = None
        if save_checkpoint:
            checkpoint_dir = tempfile.gettempdir()
            checkpoint_path = checkpoint_dir + "/model.checkpoint"
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            
        ray.train.report({"loss": loss},   checkpoint=checkpoint)
    
    
if __name__ == "__main__":
    print("DATA_ROOT:", cfg.DATA_ROOT)
    print("os.environ['TRANSFORMERS_CACHE']", os.environ['TRANSFORMERS_CACHE'])
    print("Cuda support:", torch.cuda.is_available(),":", 
                            torch.cuda.device_count(), "devices")
    print("CUDA VERSION:", torch.__version__)
#    ray.init(num_gpus=1, num_cpus=8, ignore_reinit_error=True)
    
    search_space = {
        ## constants:
        "epochs": 20,
        "model": "RNG",
        "dataset": "PDBbind",
        "feature_opt": "nomsa", # NOTE: SPD requires foldseek features!!!
        "edge_opt": "ring3",
        "fold_selection": 0,
        "save_checkpoint": False,
                
        ## hyperparameters to tune:
        "lr": ray.tune.loguniform(1e-5, 1e-3),
        "batch_size": ray.tune.choice([8, 16, 24]),        # batch size is per GPU! #NOTE: multiply this by num_workers
        
        # model architecture hyperparams
        "dropout": ray.tune.uniform(0.0, 0.5), # for fc layers
        "dropout_prot": ray.tune.uniform(0.0, 0.5),
        "pro_emb_dim": ray.tune.choice([128, 256, 512]), # input from SaProt is 480 dims
    }
    
    # each worker is a node from the ray cluster.
    # WARNING: SBATCH GPU directive should match num_workers*GPU_per_worker
    # same for cpu-per-task directive
    scaling_config = ScalingConfig(num_workers=2, # number of ray actors to launch to distribute compute across
                                   use_gpu=True,  # default is for each worker to have 1 GPU (overrided by resources per worker)
                                   resources_per_worker={"CPU": 2, "GPU": 1},
                                   # trainer_resources={"CPU": 2, "GPU": 1},
                                   # placement_strategy="PACK", # place workers on same node
                                   )
    
    print('init Tuner')     
    tuner = ray.tune.Tuner(
        TorchTrainer(train_func),
        param_space={
            "train_loop_config": search_space,
            "scaling_config": scaling_config
            },
        tune_config=ray.tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=OptunaSearch(), # using ray.tune.search.Repeater() could be useful to get multiple trials per set of params
                                       # would be even better if we could set trial-wise dependencies for a certain fold.
                                       # https://github.com/ray-project/ray/issues/33677
            num_samples=200,
        ),
    )

    results = tuner.fit()
