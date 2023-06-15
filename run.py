# %%
from src.models.helpers.contact_map import get_contact
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

PDBbind = '/cluster/projects/kumargroup/jean/data/refined-set'
path = lambda c: f'{PDBbind}/{c}/{c}_protein.pdb'

# %% main loop to create and save contact maps
cmaps = {}
for code in tqdm(os.listdir(PDBbind)[:100]):
    if os.path.isdir(os.path.join(PDBbind, code)) and code not in ["index", "readme"]:
        cmap = get_contact(path(code))
        cmaps[code] = cmap
        

#%% Displaying:
r,c = 10,10
f, ax = plt.subplots(r,c, figsize=(15, 15))
i=0
threshold = 8.0
for i, code in enumerate(cmaps.keys()):
    cmap = cmaps[code] < 12.0
    ax[i//c][i%c].imshow(cmap)
    ax[i//c][i%c].set_title(code)
    ax[i//c][i%c].axis('off')
    i+=1

# %%
