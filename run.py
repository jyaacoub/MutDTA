# %%
from src.models.helpers.contact_map import get_contact
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

PDBbind = '/cluster/projects/kumargroup/jean/data/refined-set'
path = lambda c: f'{PDBbind}/{c}/{c}_protein.pdb'
cmap_p = lambda c: f'{PDBbind}/{c}/{c}_contact_.npy'

# %% main loop to create and save contact maps
cmaps = {}
for code in tqdm(os.listdir(PDBbind)[32:100]):
    if os.path.isdir(os.path.join(PDBbind, code)) and code not in ["index", "readme"]:
        try:
            cmap = get_contact(path(code), CA_only=False)
        except Exception as e:
            print(code)
            raise e
            
        cmaps[code] = cmap
        np.save(cmap_p(code), cmap)
        

#%% Displaying:
r,c = 10,10
f, ax = plt.subplots(r,c, figsize=(15, 15))
i=0
threshold = None
for i, code in enumerate(cmaps.keys()):
    cmap = cmaps[code] if threshold is None else cmaps[code] < threshold
    ax[i//c][i%c].imshow(cmap)
    ax[i//c][i%c].set_title(code)
    ax[i//c][i%c].axis('off')
    i+=1

 # %% Saving ALL contact maps
