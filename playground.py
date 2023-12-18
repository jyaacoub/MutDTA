#%%
import torch

from src.utils.loader import Loader
from src.train_test.utils import debug
from src.utils import config as cfg
from torch_geometric.utils import dropout_edge, dropout_node

device = torch.device('cuda:0')#'cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL, DATA,  = 'SPD', 'davis'
FEATURE, EDGEW = 'foldseek', 'binary'
ligand_feature, ligand_edge = None, None
BATCH_SIZE = 20
fold = 0
pro_overlap = False
DROPOUT = 0.2


# ==== LOAD DATA ====
loaders = Loader.load_DataLoaders(data=DATA, pro_feature=FEATURE, edge_opt=EDGEW, path=cfg.DATA_ROOT, 
                                    ligand_feature=ligand_feature, ligand_edge=ligand_edge,
                                    batch_train=BATCH_SIZE,
                                    datasets=['train'],
                                    training_fold=fold,
                                    protein_overlap=pro_overlap)


# ==== LOAD MODEL ====
print(f'#Device: {device}')
model = Loader.init_model(model=MODEL, pro_feature=FEATURE, pro_edge=EDGEW, dropout=DROPOUT,
                            ligand_feature=ligand_feature, ligand_edge=ligand_edge).to(device)


# %%
train, eval = debug(model, loaders['train'], device=device)
# %%
