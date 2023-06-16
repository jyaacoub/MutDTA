#%%
import matplotlib.pyplot as plt
import numpy as np

# %%sample dgraphdta contact map:
path = "/cluster/home/t122995uhn/projects/DGraphDTA/data/kiba/pconsc4/O00141.npy"
cmap = np.load(path)
plt.matshow(cmap)
plt.show()
# %%
