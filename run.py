# %%
from src.models.helpers.contact_map import get_contact
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

PDBbind = '/cluster/projects/kumargroup/jean/data/refined-set'
path = lambda c: f'{PDBbind}/{c}/{c}_protein.pdb'

# %% main loop to create and save contact maps

# index error "CA" on 5aa9 (no CA for that residue??)
r,c = 10,10
f, ax = plt.subplots(r,c, figsize=(15, 15))
i=0
for code in tqdm(os.listdir(PDBbind)[:100]):
    if os.path.isdir(os.path.join(PDBbind, code)) and code not in ["index", "readme"]:
        cmap = get_contact(path(code))
        
        ax[i//c][i%c].imshow(cmap)
        ax[i//c][i%c].set_title(code)
        ax[i//c][i%c].axis('off')
        i+=1
        

#%%


# %%
