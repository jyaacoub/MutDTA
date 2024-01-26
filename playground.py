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
from src.utils.af_clust import AF_Clust
dir_p = f"/cluster/home/t122995uhn/projects/colabfold"

# %% EGFR
pid = "EGFR"
msa = f"{dir_p}/in_a3m_misc/{pid}/{pid}.a3m"
af = AF_Clust(keyword="test", input_msa=msa, output_dir=dir_p+ f"/in_a3m/{pid}/", verbose=True)

#%% davis 
msa = f"{dir_p}/davis_a3m/part1/WEE1.a3m"
af = AF_Clust(keyword="test", input_msa=msa, output_dir=f"{dir_p}/davis_a3m/test/", verbose=True)

# %% PDBBind
msa = f"{dir_p}/pdbbind_a3m/1a1e.msa.a3m"
af = AF_Clust(keyword="test", input_msa=msa, output_dir=f"{dir_p}/test_af_clust/", verbose=True)

# %%
