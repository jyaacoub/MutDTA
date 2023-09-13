#%%
from src.train_test.training import train_tune
from src.models.prior_work import DGraphDTA
from src.models.mut_dta import EsmDTA, EsmAttentionDTA
from src.utils.loader import Loader
from src.utils import config
import pickle


#%%
import ray
from ray import tune
ray.init(num_cpus=6, num_gpus=3)


# %%
MODEL = 'EDI'
PRO_FEATURE = 'nomsa'
DATA = 'davis'
EDGE = 'simple'
data_root = "/cluster/home/t122995uhn/projects/data"# WARNING: HARD CODED PATH

config = {
    "edge": tune.grid_search(['weighted', 'binary']),
    "dropout": tune.grid_search([0.1, 0.2, 0.4, 0.5]),
    
    "lr": tune.grid_search([1e-4, 1e-3, 1e-2]),
    "batch_size": tune.grid_search([4, 5, 6, 10, 12]),
}

#%%
train_dataset = Loader.load_dataset(DATA, PRO_FEATURE, EDGE, subset='train', path=data_root) 
val_dataset = Loader.load_dataset(DATA, PRO_FEATURE, EDGE, subset='val', path=data_root)

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(train_tune, model=MODEL, pro_feature=PRO_FEATURE,
                             train_dataset=train_dataset, 
                             val_dataset=val_dataset),
        resources={"cpu": 2, "gpu": 1}
    ),
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
    ),
    param_space=config,
)
# %%
results = tuner.fit()
print(results)
print(results.get_best_result())

with open(f'{MODEL}_{PRO_FEATURE}_{DATA}-rayTuneResults.pickle', 'wb') as handle:
    pickle.dump(results, handle)