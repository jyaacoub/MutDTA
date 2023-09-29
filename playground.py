# %% Rename old models so that they match with new format
import os
from os import path as osp
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv = 'results/model_media/model_stats.csv'

#%%
df = pd.read_csv(csv)

# create data, feat, and overlap columns for easier filtering.
df['data'] = df['run'].str.extract(r'_(davis|kiba|PDBbind)', expand=False)
df['feat'] = df['run'].str.extract(r'_(nomsa|msa|shannon)F_', expand=False)
df['edge'] = df['run'].str.extract(r'_(binary|simple|anm|af2)E_', expand=False)
df['ddp'] = df['run'].str.contains('DDP-')
df['improved'] = df['run'].str.contains('IM_') # trail of model name will include I if "improved"
df['batch_size'] = df['run'].str.extract(r'_(\d+)B_', expand=False)

df.loc[df['run'].str.contains('EDIM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
df.loc[df['run'].str.contains('EDAIM'), 'feat'] += '-ESM'

df['overlap'] = df['run'].str.contains('overlap')

df[['run', 'data', 'feat', 'edge', 'batch_size', 'overlap']]

###########################################
#%% Figure 1 - Protein overlap cindex difference (nomsa)

def fig1_pro_overlap(df, sel_col='cindex'):
    grouped_df = df[(df['feat'] == 'nomsa') 
                    & (df['batch_size'] == '64') 
                    & (df['edge'] == 'binary')
                    & (~df['ddp'])              
                    & (~df['improved'])].groupby(['data'])
    
    # each group is a dataset with 2 bars (overlap and no overlap)
    for group_name, group_data in grouped_df:
        print(f"\nGroup Name: {group_name}")
        print(group_data[['cindex', 'mse', 'overlap']])

    # these groups are spaced by the data type, physically grouping bars of the same dataset together.
    # Initialize lists to store cindex values for each dataset type
    t_overlap = []
    f_overlap = []
    dataset_types = []

    for dataset, group in grouped_df:
        print('')
        print(group[['cindex', 'mse', 'overlap', 'data']])
        # overlap
        t_overlap_vals = group[group['overlap']][sel_col].values
        if len(t_overlap_vals) == 0:
            t_overlap.append(0)
        elif len(t_overlap_vals) == 1:
            t_overlap.append(t_overlap_vals[0])
        else:
            raise IndexError('Too many overlap values, filter is too broad.')
        
        # no overlap
        f_overlap_vals = group[~group['overlap']][sel_col].values
        print(f_overlap_vals)
        if len(f_overlap_vals) == 0:
            f_overlap.append(0)
        elif len(f_overlap_vals) == 1:
            f_overlap.append(f_overlap_vals[0])
        else:
            raise IndexError('Too many overlap values, filter is too broad.')
        dataset_types.append(dataset)

    # Create an array of x positions for the bars
    x = np.arange(len(dataset_types))

    # Set the width of the bars
    width = 0.35

    # Create a bar plot with two bars for each dataset type
    fig, ax = plt.subplots()
    bar2 = ax.bar(x - width/2, t_overlap, width, label='With Overlap')
    bar1 = ax.bar(x + width/2, f_overlap, width, label='No Overlap')

    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_types)

    # Set the y-axis label
    ax.set_ylabel('cindex')
    ax.set_ylim([0.5, 1]) # 0.5 is the worst cindex value

    # Set the title and legend
    ax.set_title('Protein Overlap cindex Difference (nomsa)')
    ax.legend()

    # Show the plot
    plt.show()

fig1_pro_overlap(df)

#%% Figure 2 - Feature type cindex difference
def fig2_pro_feat(df):
    # comparing nomsa, msa, shannon, and esm
    # group by data type
    
    # this will capture multiple models per dataset (different LR, batch size, etc)
    #   Taking the max cindex value for each dataset will give us the best model for each dataset
    grouped_df = df[(df['edge'] == 'binary')
                    & (~df['overlap'])].groupby(['data'])
    
    # each group is a dataset with 4 bars (nomsa, msa, shannon, esm)
    for group_name, group_data in grouped_df:
        print(f"\nGroup Name: {group_name}")
        print(group_data[['cindex', 'mse', 'feat']])

    # these groups are spaced by the data type, physically grouping bars of the same dataset together.
    # Initialize lists to store cindex values for each dataset type
    nomsa = []
    msa = []
    shannon = []
    esm = []
    dataset_types = []
    
    for dataset, group in grouped_df:
        print('')
        print(group[['cindex', 'mse', 'overlap', 'data']])
        
        nomsa_v = group[group['feat'] == 'nomsa']['cindex'].max() # NOTE: take min for mse
        msa_v = group[group['feat'] == 'msa']['cindex'].max()
        shannon_v = group[group['feat'] == 'shannon']['cindex'].max()
        ESM_v = group[group['feat'] == 'ESM']['cindex'].max()
        
        # appending if not nan else 0
        nomsa.append(nomsa_v if not np.isnan(nomsa_v) else 0)
        msa.append(msa_v if not np.isnan(msa_v) else 0)
        shannon.append(shannon_v if not np.isnan(shannon_v) else 0)
        esm.append(ESM_v if not np.isnan(ESM_v) else 0)
        dataset_types.append(dataset)
        
    # Create an array of x positions for the bars
    x = np.arange(len(dataset_types))
    
    # Set the width of the bars
    width = 0.2
    
    # Create a bar plot with 4 bars for each dataset type
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width, nomsa, width, label='nomsa')
    bar2 = ax.bar(x, msa, width, label='msa')
    bar3 = ax.bar(x + width, shannon, width, label='shannon')
    bar4 = ax.bar(x + width*2, esm, width, label='esm')
    
    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_types)
    
    # Set the y-axis label
    ax.set_ylabel('cindex')
    ax.set_ylim([0.5, 1])
    
    # Set the title and legend
    ax.set_title('Feature Type cindex Difference')
    ax.legend()
    
    # Show the plot
    plt.show()

fig2_pro_feat(df)

#%% Figure 3 - Edge type cindex difference


#%%
exit()
import json, os
import pandas as pd
from src.utils.residue import Chain
from tqdm import tqdm
from prody import parsePDB

root_dir = '/cluster/home/t122995uhn/projects/data/kiba_tmp'
save_dir = f'{root_dir}/structures'
pdb_fp = lambda x: f'{save_dir}/{x}.pdb'

# Contains protein sequences mapped to uniprotIDs
unique_prots = json.load(open(f'{root_dir}/proteins.txt', 'r'))
# [...] send to https://www.uniprot.org/id-mapping to get associated pdb files
# this returns a tsv containing all matching pdbs for each unique uniprotID
df = pd.read_csv(f'{root_dir}/kiba_mapping_pdb.tsv', sep='\t')

#%%
from thefuzz import fuzz
# fuzzy matching:
matches = {} # tracks matching sequences to pdb structures
fails = []
for i, row in tqdm(df.iterrows(), desc='Matching pdbs', total=len(df)):
    uniprot = row['From']
    pdb = row['To']
    if uniprot in matches: continue # already matched
    try:
        pdb_s = parsePDB(pdb_fp(pdb), subset='ca')
        hv = pdb_s.getHierView()
        seq = pdb_s.getSequence() # includes all chains 
    except Exception as e:
        fails.append((pdb, e))
        # raise Exception(f'Error on {i}, ({uniprot}, {pdb}).') from e
    
    curr_seq = unique_prots[uniprot] 
    
    for c in hv:
        chain_seq = c.getSequence()        
        if (len(chain_seq) == len(curr_seq) and chain_seq == curr_seq) or \
            (len(chain_seq) > len(curr_seq) and curr_seq in chain_seq):
            # (len(chain_seq) < len(curr_seq) and chain_seq in curr_seq): # this would cause small chains to match with reference
            sim_score = fuzz.ratio(chain_seq, curr_seq)
            matches[uniprot] = (c, sim_score)
            
    # if (len(seq) == len(curr_seq) and seq == curr_seq) or \
    #     (len(seq) > len(curr_seq) and curr_seq in seq)or \
    #     (len(seq) < len(curr_seq) and seq in curr_seq):
    #     matches[uniprot] = pdb

#%% Fails are due to empty pdb files
for c, _ in fails:
    os.remove(pdb_fp(c))



#%% Download missing
from src.data_processing.downloaders import Downloader
import pickle
with open('temp.pkl', 'rb') as f:
    ids = pickle.load(f)
    
Downloader.download_PDBs(ids, save_dir=save_dir)


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
