#%% from src.models.helpers.contact_map import get_contact
# code = '1a1e'
# path = f'/cluster/projects/kumargroup/jean/data/refined-set/{code}/{code}_protein.pdb'
# get_contact(path, display=True)

from src.data_analysis.display import plot_all
code='1a1e'
conf = f'/cluster/projects/kumargroup/jean/data/vina_conf/run8/{code}_conf.txt'
pocket_p = f'/cluster/projects/kumargroup/jean/data/refined-set/{code}/{code}_pocket.pdb'

# %% plotting
# fig, prot, lig = plot_all(conf, show=True)
fig, prot, lig = plot_all(conf, show=False, pocket=pocket_p)
# fig, prot, lig = plot_all(conf, show=True, fig=fig,
#                         pocket=pocket_p)

#%%

