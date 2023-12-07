# This is a simple tuning script for the raytune library.
# no support for distributed training in this file.

import torch
from ray.air import session
from ray.train.torch import TorchCheckpoint
from ray import tune
from ray.tune.search.optuna import OptunaSearch


from src.utils.loader import Loader
from src.train_test.tune import simple_train, simple_test
from src.utils import config as cfg


def objective(config):
    save_checkpoint = config.get("save_checkpoint", False)
    loaders = Loader.load_DataLoaders(data=config['dataset'], pro_feature=config['feature_opt'], 
                                      edge_opt=config['edge_opt'], 
                                      path=cfg.DATA_ROOT, 
                                      batch_train=config['batch_size'],
                                      datasets=['train', 'val'],
                                      training_fold=config['fold_selection'])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = Loader.init_model(model=config["model"], pro_feature=config["feature_opt"],
                              pro_edge=config["edge_opt"], dropout=config["dropout"]
                              # WARNING: no ligand features for now
                              ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    for _ in range(config["epochs"]):
        simple_train(model, optimizer, loaders['train'], 
                     device=device, 
                     epochs=1)  # Train the model
        loss = simple_test(model, loaders['val'], 
                           device=device)  # Compute test accuracy
        
        checkpoint = None
        if save_checkpoint:
            checkpoint = TorchCheckpoint.from_state_dict(model.state_dict())
            
        # Report metrics (and possibly a checkpoint) to Tune
        session.report({"mean_loss": loss}, checkpoint=checkpoint)

algo = OptunaSearch()
# algo = ConcurrencyLimiter(algo, max_concurrent=4)
search_space = {
    # constants:
    "epochs": 15, # 15 epochs
    "model": "DG",
    "dataset": "davis",
    "feature_opt": "nomsa",
    "edge_opt": "binary",
    "fold_selection": 0,
    "save_checkpoint": False,
    
    # hyperparameters to tune:
    "lr": tune.loguniform(1e-4, 1e-2),
    "dropout": tune.uniform(0, 0.5),
    "batch_size": tune.choice([16, 32, 64, 128]),
}

tuner = tune.Tuner(
    tune.with_resources(objective, resources={"cpu": 6, "gpu": 1}), # NOTE: must match SBATCH directives
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=50,
    ),
    param_space=search_space,
)

results = tuner.fit()
