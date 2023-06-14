#%% from src.models.helpers.contact_map import get_contact
# code = '1a1e'
# path = f'/cluster/projects/kumargroup/jean/data/refined-set/{code}/{code}_protein.pdb'
# get_contact(path, display=True)

from src.data_analysis.display import plot_all
conf = '/cluster/projects/kumargroup/jean/data/vina_conf/run8/1a1e_conf.txt'

# %% plotting
fig, prot, lig = plot_all(conf, show=True)

#%%

