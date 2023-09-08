#%%
import pandas as pd
code = "3eql"
df_old = pd.read_csv('/cluster/home/t122995uhn/projects/data/PDBbindDataset/old_nomsa/full/XY.csv', index_col=0)
df = pd.read_csv('/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa/full/XY.csv', index_col=0)

df_old['prot_seq'].eq(df['prot_seq'])

#%% checking to see if current seq match with MSA generated alignments
from src.data_processing.datasets import PDBbindDataset
FEATURE='nomsa' # msa not working due to refactoring that caused change in sequences used
dataset = PDBbindDataset(save_root=f'../data/PDBbindDataset/{FEATURE}',
                    data_root=f'../data/v2020-other-PL/',
                    aln_dir=f'../data/PDBbind_aln',
                    cmap_threshold=8.0,
                    edge_opt='anm',
                    feature_opt=FEATURE,
                    overwrite=True
                    )

#%%
from src.feature_extraction.process_msa import create_pfm_np_files

create_pfm_np_files('../data/PDBbind_aln/', processes=4)

#%%
from src.utils.loader import Loader
from src.utils import config
import pickle

import numpy as np
import pandas as pd
from src.data_processing.processors import Processor
from prody import parsePDB, calcANM

# delNonstdAminoacid('SEP')


td = Loader.load_dataset('PDBbind', 'nomsa',
                                    path="/cluster/home/t122995uhn/projects/data") 

hv = parsePDB('../data/v2020-other-PL/5swg/5swg_protein.pdb', subset='calpha').getHierView()
for c in hv: print(c)

#%%
ch = Processor.pdb_get_chain('../data/v2020-other-PL/5swg/5swg_protein.pdb', model=1)
seq_mine = ch.getSequence()
seq_real = hv['A'].getSequence()
for i, c in enumerate(seq_real):
    if seq_real[i] != seq_mine[i]:
        print(f'{i}: MISMATCH')
        print(f'\t{seq_real[i-3:i+3]}')
        print(f'\t{seq_mine[i-3:i+3]}')
        break


#%%
from prody import parsePDB, calcANM
from src.feature_extraction.protein import calcCrossCorr

idx=1
pdb_fp = td.pdb_p(td[idx]['code'])
target_seq = td[idx]['protein'].pro_seq
n_modes=10

pdb = parsePDB(pdb_fp, subset='calpha').getHierView()
    
anm = None
for chain in pdb:
    if chain.getSequence() == target_seq:
        anm, _ = calcANM(chain, selstr='calpha', n_modes=n_modes)
        break
    
if anm is None:
    raise ValueError(f"No matching chain found in pdb file ({pdb_fp})")
else:
    # norm=True normalizes it from -1.0 to 1.0
    cc = calcCrossCorr(anm[:n_modes], n_cpu=2, norm=True)







###########################################################################################
#%% selecting chain A to run ANM on:
from prody import parsePDB, calcANM, calcCrossCorr
import matplotlib.pyplot as plt
import numpy as np

pdb = "10gs"
pdb_fp = f"/cluster/home/t122995uhn/projects/data/v2020-other-PL/{pdb}/{pdb}_protein.pdb"

pdb = parsePDB(pdb_fp, subset='calpha').getHierView()['A']
anm, atoms = calcANM(pdb, selstr='calpha', n_modes=20)

# %%
cc = calcCrossCorr(anm[:2], n_cpu=1, norm=True)

plt.matshow(cc)#(cc-np.min(cc))/(np.max(cc)-np.min(cc)))
plt.title('Cross Correlation Matrix:')
plt.show()

lap_m = anm.getKirchhoff()
adj = lap_m - np.eye(lap_m.shape[0])*lap_m
plt.title('Overlapped with adjacency matrix')
plt.imshow(adj, alpha=0.5)
plt.imshow(cc, alpha=0.6)#(cc-np.min(cc))/(np.max(cc)-np.min(cc)), alpha=0.6)

# %% TM SCORE:
# ================================================================================================
from prody import parsePDB, matchAlign, showProtein
from pylab import legend
import numpy as np

# %%
# tm score for A&B is 0.9835 (https://seq2fun.dcmb.med.umich.edu//TM-score/tmp/110056.html)
src_model = '/cluster/home/t122995uhn/projects/data/v2020-other-PL/1a1e/1a1e_protein.pdb'
pred_model = '/cluster/home/t122995uhn/projects/colabfold/out/1a1e.msa_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb'

sm = parsePDB(src_model, model=1, subset="ca", chain="A")
sm.setTitle('experimental')
pm = parsePDB(pred_model, model=1, subset="ca", chain="A")
pm.setTitle('alphafold')

showProtein(sm,pm)
legend()

#%% Performing alignment before TM-score
result = matchAlign(pm, sm)

showProtein(sm,pm)
legend()

# %%
def tm_score(xyz0, xyz1): #Check if TM-align use all atoms!    
    L = len(xyz0)
    # d0 is less than 0.5 for L < 22 
    # and nan for L < 15 (root of a negative number)
    d0 = 1.24 * np.power(L - 15, 1/3) - 1.8
    d0 = max(0.5, d0) 

    # compute the distance for each pair of atoms
    di = np.sum((xyz0 - xyz1) ** 2, 1) # sum along first axis
    return np.sum(1 / (1 + (di / d0) ** 2)) / L

# TM score for predicted model 1 with chain A of src should be 0.9681 (https://seq2fun.dcmb.med.umich.edu//TM-score/tmp/987232.html)
tm_score(sm.getCoords(), 
         pm.getCoords())
