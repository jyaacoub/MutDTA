# #%%
# from src.data_processing.datasets import PlatinumDataset
# from src.utils.residue import Chain

# import pandas as pd
# import numpy as np
# import os.path as osp
# import os
# data = 'PDBbind'
# FEATURE = 'nomsa'
# data_root_dir = '/cluster/home/t122995uhn/projects/data/kiba/'
# df_p = '../data/PlatinumDataset/raw/platinum_flat_file.csv'
# df_pro = '../data/PlatinumDataset/nomsa_binary/full/XY.csv'
# # data_root_dir = '/home/jyaacoub/projects/data/'
# DATA_ROOT = f'{data_root_dir}/'

# df_raw = pd.read_csv(df_p)

# # get test sample 
# row = df_raw.iloc[778]
# mut = row['mutation']
# pdb_wt = row['mut.wt_pdb']
# pdb_mt = row['mut.mt_pdb'] 
# t_chain = row['affin.chain']


# #%%
# from src.data_processing.downloaders import Downloader

# # download missing pdbs:
# # only missing pdbs will be those that are not wildtypes since that is the default
# def filter(row):
#     mt = row['mut.mt_pdb']
#     return mt != 'NO' and not osp.isfile(f'../data/PlatinumDataset/raw/platinum_pdb/{mt}.pdb')

# Downloader.download_PDBs(df_raw[df_raw.apply(filter, axis=1)]['mut.mt_pdb'])

#%%
from src.data_processing.datasets import PlatinumDataset
print('CREATING PLATINUM DATASET...')
d = PlatinumDataset(
    save_root=f'../data/PlatinumDataset/',
    data_root=f'../data/PlatinumDataset/raw',
    aln_dir=None,
    cmap_threshold=8.0,
    feature_opt='nomsa',
    edge_opt='binary')
# exit()

# dataset = DavisKibaDataset(
#         save_root=f'../data/DavisKibaDataset/kiba/',
#         data_root=DATA_ROOT,
#         aln_dir=f'{DATA_ROOT}/aln/', 
#         cmap_threshold=-0.5, 
#         feature_opt=FEATURE,
#         edge_opt = 'anm',
#         af_conf_dir='../colabfold/pdbbind_out/out0',
# )


#%%
from src.data_processing.downloaders import Downloader
import pandas as pd
import json, shutil, os

root_dir = '/cluster/home/t122995uhn/projects/data/kiba'
save_dir = f'{root_dir}/structures'

unique_prots = json.load(open(f'{root_dir}/proteins.txt', 'r'))
# [...]
# send unique prots to uniprot for structure search

##### Map to PDB structural files
# downloaded from https://www.uniprot.org/id-mapping/bcf1665e2612ea050140888440f39f7df822d780/overview
df = pd.read_csv(f'{root_dir}/kiba_mapping_pdb.tsv', sep='\t')
# getting only first hit for each unique PDB-ID
df = df.loc[df[['From']].drop_duplicates().index]

# getting missing/unmapped prot ids
missing = [prot_id for prot_id in unique_prots.keys() if prot_id not in df['From'].values]

# %%
# ##### download pdb files
# Downloader.download_PDBs(df['To'].values, save_dir=save_dir)

# # retrieve missing structures from AlphaFold:
# Downloader.download_predicted_PDBs(missing, save_dir=save_dir)

# # NOTE: some uniprotIDs map to the same structure and so using the df mapping we will rename the mapped pdb to be uniprot file names
# #%% copying as neccessary

# # copying to new uniprot id file names
# for i, row in df.iterrows():
#     uniprot = row['From']
#     pdb = row['To']
#     # finding pdb file
#     f_in = f'{save_dir}/{pdb}.pdb'
#     f_out = f'{save_dir}/{uniprot}.pdb'
#     if not os.path.isfile(f_in):
#         print('Missing', f_in)
#     elif not os.path.isfile(f_out):
#         shutil.copy(f_in, f_out)

# # removing old pdb files.
# for i, row in df.iterrows():
#     pdb = row['To']
#     f_in = f'{save_dir}/{pdb}.pdb'
#     if os.path.isfile(f_in):
#         os.remove(f_in)


#%% Some downloaded pdbs dont match the provided input sequence
from src.utils.residue import Chain
# mismatch
mismatch = {}
for uniprot, seq in unique_prots.items():
    if uniprot in missing: continue
    f_in = f'{save_dir}/{uniprot}.pdb'
    if not os.path.isfile(f_in):
        print('Missing', f_in)
    else:
        c = Chain(f_in)
        pdb_sequence = c.getSequence()
        if len(seq) != len(pdb_sequence):
            mismatch[uniprot] = (c, seq, mismatch_count)
            print(f'LENGTH MISMATCH FOR {uniprot} - {f_in} (Expected: {len(seq)}, Actual: {len(pdb_sequence)})')
        else:
            # Calculate the number of mismatches
            mismatch_count = sum(1 for i in range(len(seq)) if seq[i] != pdb_sequence[i])
            
            if mismatch_count > 0:
                mismatch[uniprot] = (c, seq, mismatch_count)
                print(f'MISMATCH FOR {uniprot} - {f_in} (Mismatches: {mismatch_count})')
            
        # try:
        #     hv = parsePDB(f_in, subset='ca').getHierView()
        #     # s.getSequence()
        #     # c = Chain(f_in)
        #     match = None
        #     for c in hv.iterChains():
        #         if c.getSequence() == seq:
        #             match = c
        #             break
        #     # if c.getSequence() != seq:
        #     if match is None:
        #         print(f'MISMATCH FOR {uniprot} - {f_in}')
        # except Exception as e:
        #     raise Exception(f'{f_in}') from e

#%%
from src.data_processing.datasets import PDBbindDataset
FEATURE='nomsa'
dataset = PDBbindDataset(save_root=f'../data/PDBbindDataset/',
        data_root=f'../data/v2020-other-PL/',
        aln_dir=f'../data/PDBbind_aln',
        cmap_threshold=8.0,
        edge_opt='af2',
        feature_opt=FEATURE,
        overwrite=False, # overwrite old cmap.npy files
        af_conf_dir='/cluster/home/t122995uhn/projects/colabfold/pdbbind_out/out0'
        )
#%%
from src.utils.residue import Chain
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm

# Create empty lists to store protein sequence lengths and average TM_scores
sequence_lengths = []
average_tm_scores = []

# Read the CSV file
df = pd.read_csv("/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_anm/full/XY.csv", 
                 index_col=0)

af_dir = '/cluster/home/t122995uhn/projects/data/PDBbind_afConf'

for code in tqdm(df[['prot_id']].drop_duplicates().index):
    t_p = f'/cluster/home/t122995uhn/projects/data/v2020-other-PL/{code}/{code}_protein.pdb'
    template = Chain(t_p)

    af_confs = glob(f'{af_dir}/{code}*.pdb')

    chains = [Chain(p) for p in af_confs]
    if len(chains) == 0: continue
    tm_avg = np.max([c.TM_score(template) for c in chains])

    # Get the protein sequence length (you should implement this based on your data)
    sequence_length = len(template)  # Replace with your code to get sequence length

    sequence_lengths.append(sequence_length)
    average_tm_scores.append(tm_avg)

#%% Create a bar graph
plt.figure(figsize=(10, 6))
plt.scatter(sequence_lengths, average_tm_scores, alpha=0.3)
plt.xlabel('Protein Sequence Length')
plt.ylabel('Max TM_score')
plt.title('Max TM_score vs. Protein Sequence Length')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
from src.utils.residue import Chain
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# IN: list of chains
# OUT: edge weights and updated edge index?
df = pd.read_csv("/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_anm/full/XY.csv", 
                 index_col=0)


af_dir = '/cluster/home/t122995uhn/projects/data/PDBbind_afConf'

for code in df[['prot_id']].drop_duplicates().index:
    t_p = f'/cluster/home/t122995uhn/projects/data/v2020-other-PL/{code}/{code}_protein.pdb'
    template = Chain(t_p)

    af_confs = glob(f'{af_dir}/{code}*.pdb')

    chains = [Chain(p) for p in af_confs]
    tm_avg = np.mean([c.TM_score(template) for c in chains])
    print(tm_avg)

# chains[0].getCoords()
#%% Get adjacency matrix

M = np.array([c.get_contact_map() for c in chains])
ew = np.sum(M < 8.0, axis=0)/len(M)







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
t_p = f'/cluster/home/t122995uhn/projects/data/v2020-other-PL/{code}/{code}_protein.pdb'
af_confs = glob(f'{af_dir}/{code}*.pdb')
src_model ='/cluster/home/t122995uhn/projects/data/v2020-other-PL/1a1e/1a1e_protein.pdb'
pred_model = '/cluster/home/t122995uhn/projects/colabfold/out/1a1e.msa_unrelaxed_rank_002_alphafold2_ptm_model_1_seed_000.pdb'

sm = parsePDB(src_model, model=1, subset="ca", chain="A")
sm.setTitle('experimental')
pm = parsePDB(pred_model, model=1, subset="ca", chain="A")
pm.setTitle('alphafold')

# showProtein(sm,pm)
# legend()
import numpy as np

# Assuming you have two sets of 3D coordinates, c1 and c2, as NumPy arrays
c1, c2 = sm.getCoords(), pm.getCoords()

# Calculate the centroid (center of mass) of each set of coordinates
centroid1 = np.mean(c1, axis=0)
centroid2 = np.mean(c2, axis=0)

# Translate both sets of coordinates to their respective centroids
c1_centered = c1 - centroid1
c2_centered = c2 - centroid2

# Calculate the covariance matrix
covariance_matrix = np.dot(c2_centered.T, c1_centered)

# Use singular value decomposition (SVD) to find the optimal rotation matrix
u, _, vt = np.linalg.svd(covariance_matrix)
rotation_matrix = np.dot(u, vt)

# Apply the calculated rotation matrix to c2_centered
c2_aligned = np.dot(c2_centered, rotation_matrix)

# Calculate the root mean square deviation (RMSD) to measure structural similarity
rmsd = np.sqrt(np.mean((c1_centered - c2_aligned) ** 2))
print(f"RMSD: {rmsd}")
sm.setCoords(c1_centered)
pm.setCoords(c2_aligned)
showProtein(sm,pm)
legend()

#%% Performing alignment before TM-score

from prody import confProDy
confProDy(verbosity='debug') # stop printouts from prody
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

# %%
