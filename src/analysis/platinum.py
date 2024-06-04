import torch, os
import pandas as pd

from src import cfg
from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.train_test.training import test
device = torch.cuda.device(0) if torch.cuda.is_available() else torch.device('cpu')

CONFIG = TUNED_MODEL_CONFIGS['davis_gvpl_aflow']
#%% load up model:
cp_dir = cfg.CHECKPOINT_SAVE_DIR

MODEL_KEY = lambda fold: Loader.get_model_key(CONFIG['model'], CONFIG['dataset'], CONFIG['feature_opt'], CONFIG['edge_opt'], 
                                 CONFIG['batch_size'], CONFIG['lr'], CONFIG['architecture_kwargs']['dropout'],
                                 n_epochs=2000, fold=fold, 
                                 ligand_feature=CONFIG['lig_feat_opt'], ligand_edge=CONFIG['lig_edge_opt'])
cp = lambda fold: f"{cp_dir}/{MODEL_KEY(fold)}.model"

out_dir = f'{cfg.MEDIA_SAVE_DIR}/test_set_pred/'
os.makedirs(out_dir, exist_ok=True)

model = Loader.init_model(model=CONFIG["model"], pro_feature=CONFIG["feature_opt"],
                            pro_edge=CONFIG["edge_opt"],**CONFIG['architecture_kwargs'])

#%%
# load up platinum test db
loaders = Loader.load_DataLoaders(cfg.DATA_OPT.platinum,
                               pro_feature    = CONFIG['feature_opt'], 
                               edge_opt       = CONFIG['edge_opt'],
                               ligand_feature = CONFIG['lig_feat_opt'], 
                               ligand_edge    = CONFIG['lig_edge_opt'],
                               datasets=['test'])

for i in range(5):
    model.safe_load_state_dict(torch.load(cp(i), map_location=device))
    model.to(device)
    model.eval()

    loss, pred, actual = test(model, loaders['test'], device, verbose=True)
    
    # saving as csv with columns code, pred, actual
    # get codes from test loader
    codes, pid = [b['code'][0] for b in loaders['test']], [b['prot_id'][0] for b in loaders['test']]
    df = pd.DataFrame({'prot_id': pid, 'pred': pred, 'actual': actual}, index=codes)
    df.index.name = 'code'
    df.to_csv(f'{out_dir}/{MODEL_KEY(i)}_PLATINUM.csv')