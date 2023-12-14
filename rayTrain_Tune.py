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
                            pro_edge=config["edge_opt"], dropout=config["dropout"])
    
    # prepare model with rayTrain (moves it to correct device and wraps it in DDP)
    model = ray.train.torch.prepare_model(model)
    
    # ============ Load dataset ==============
    loaders = Loader.load_DataLoaders(data=config['dataset'], pro_feature=config['feature_opt'], 
                                      edge_opt=config['edge_opt'], 
                                      path=cfg.DATA_ROOT, 
                                      batch_train=config['batch_size'],
                                      datasets=['train', 'val'],
                                      training_fold=config['fold_selection'])
    
    # prepare dataloaders with rayTrain (adds DistributedSampler and moves to correct device)
    for k in loaders.keys():
        loaders[k] = ray.train.torch.prepare_dataloader(loaders[k])
    
    
    # ============= Simple training and eval loop =====================
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    save_checkpoint = config.get("save_checkpoint", False)
    
    for _ in range(15):
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
            
        ray.train.report({"loss": loss.item()},   checkpoint=checkpoint)
    
    
if __name__ == "__main__":    

    ray.init(num_gpus=1, num_cpus=8, ignore_reinit_error=True)
    
    search_space = {
        # constants:
        "epochs": 10,
        "model": "DG",
        "dataset": "davis",
        "feature_opt": "nomsa",
        "edge_opt": "binary",
        "fold_selection": 0,
        "save_checkpoint": False,
                
        # hyperparameters to tune:
        "lr": ray.tune.loguniform(1e-4, 1e-2),
        "dropout": ray.tune.uniform(0, 0.5),
        "embedding_dim": ray.tune.choice([64, 128, 256]),
        "batch_size": ray.tune.choice([16, 32, 48]),
    }
    
    scaling_config = ScalingConfig(num_workers=2, # number of ray actors to launch
                                   use_gpu=False,
                                   trainer_resources={"CPU": 4, "GPU": 1},
                                #    placement_strategy="PACK", # place workers on same node
                                   )
        
    tuner = ray.tune.Tuner(
        TorchTrainer(train_func),
        param_space={
            "train_loop_config": search_space,
            "scaling_config": scaling_config
            },
        tune_config=ray.tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=OptunaSearch(),
            num_samples=50,
        ),
    )

    results = tuner.fit()