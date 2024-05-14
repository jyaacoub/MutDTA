#%% Building davis GVPL_aflow dataset
from src.data_prep.init_dataset import create_datasets
from src import config as cfg

create_datasets(cfg.DATA_OPT.davis, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                ligand_features=cfg.LIG_FEAT_OPT.gvp, ligand_edges=cfg.LIG_EDGE_OPT.binary,
                k_folds=5)

# %%
from src.utils.loader import Loader

l = Loader.load_DataLoaders(cfg.DATA_OPT.davis, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                            ligand_feature=cfg.LIG_FEAT_OPT.gvp, ligand_edge=cfg.LIG_EDGE_OPT.binary,
                            training_fold=0, batch_train=5)


#%%
d = {'11314340': 'already downloaded', '24889392': 'already downloaded', '11409972': 'already downloaded', '11338033': 'already downloaded', '10184653': 'already downloaded', '5287969': 'already downloaded', '6450551': 'already downloaded', '11364421': 'already downloaded', '9926054': 'downloaded', '16007391': 'already downloaded', '5328940': 'already downloaded', '11234052': 'already downloaded', '11656518': 'already downloaded', '6918454': 'already downloaded', '156414': 'already downloaded', '9933475': 'already downloaded', '11626560': 'already downloaded', '3062316': 'already downloaded', '156422': 'already downloaded', '44150621': 'downloaded', '176167': 'already downloaded', '176870': 'already downloaded', '42642645': 'already downloaded', '11717001': 'already downloaded', '16725726': 'already downloaded', '11617559': 'already downloaded', '123631': 'already downloaded', '5291': 'already downloaded', '4908365': 'already downloaded', '11427553': 'already downloaded', '208908': 'already downloaded', '126565': 'already downloaded', '11485656': 'already downloaded', '9929127': 'already downloaded', '11712649': 'already downloaded', '10074640': 'already downloaded', '51004351': 'already downloaded', '11667893': 'already downloaded', '9915743': 'already downloaded', '644241': 'already downloaded', '447077': 'already downloaded', '10461815': 'already downloaded', '9884685': 'already downloaded', '24180719': 'already downloaded', '25243800': 'already downloaded', '10113978': 'already downloaded', '17755052': 'already downloaded', '11984591': 'downloaded', '153999': 'already downloaded', '25127112': 'downloaded', '176155': 'already downloaded', '24779724': 'already downloaded', '3025986': 'already downloaded', '10138260': 'already downloaded', '10127622': 'already downloaded', '216239': 'already downloaded', '44259': 'already downloaded', '5329102': 'already downloaded', '16038120': 'already downloaded', '10427712': 'already downloaded', '16722836': 'already downloaded', '3038522': 'already downloaded', '9926791': 'already downloaded', '5494449': 'already downloaded', '3038525': 'already downloaded', '3081361': 'already downloaded', '9809715': 'already downloaded', '151194': 'already downloaded'}

for k,v in d.items():
    if v != 'already downloaded':
        print(k, v)
        
        
# %%
