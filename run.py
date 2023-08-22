#%%
import ray
from ray import tune
from src.train_test.training import train_tune
from src.models.prior_work import DGraphDTA
from src.models.mut_dta import EsmDTA, EsmAttentionDTA
from src.utils.loader import Loader
ray.init(num_cpus=16, num_gpus=1)


    

#%%
config = {
    "model_cls": tune.grid_search([DGraphDTA]),
    
    "lr": tune.grid_search([0.0001, 0.001, 0.01]),
    "batch_size": tune.grid_search([32, 64, 128]),
    "dropout": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
}

train_dataset = Loader.load_dataset('davis', 'nomsa', subset='train',
                                    path='/home/jyaacoub/projects/data') # WARNING: HARD CODED PATH
val_dataset = Loader.load_dataset('davis', 'nomsa', subset='val',
                                    path='/home/jyaacoub/projects/data')

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(train_tune, train_dataset=train_dataset, 
                             val_dataset=val_dataset),
        resources={"cpu": 8, "gpu": .5}
    ),
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
    ),
    param_space=config,
)
# %%
results = tuner.fit()

# %%
