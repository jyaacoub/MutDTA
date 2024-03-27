# %%
from prody import fetchPDB

fetchPDB('10gs', compressed=False)

# %%
from src.utils.residue import Chain
c = Chain('10gs.pdb', grep_atoms={'CA', 'N', 'C'})
# %%
import logging
logging.getLogger().setLevel(logging.DEBUG)

c.getCoords(get_all=True).shape # (N, 3)

# %%
from src.data_prep.feature_extraction.gvp import GVPFeatures

gvp_f = GVPFeatures()

# %%
f = gvp_f.featurize_as_graph('10gs', c.getCoords(get_all=True), c.sequence)
# %%
