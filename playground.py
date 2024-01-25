# %%
from src.utils.loader import Loader
from src.utils import config as cfg

model = Loader.init_test_model()

loaders = Loader.load_DataLoaders(data='davis', pro_feature='nomsa', edge_opt='binary', path=cfg.DATA_ROOT, 
                                        ligand_feature=None, ligand_edge=None,
                                        batch_train=1,
                                        datasets=['test'])
# %%
for b in loaders['test']: break
# %%
model(b['protein'], b['ligand'])
# %%
