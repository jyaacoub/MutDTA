# %% Creating Data folds for Davis
from src.data_processing.init_dataset import create_datasets

create_datasets(
    data_opt=['davis'],
    feat_opt=['nomsa'],
    edge_opt=['binary'],
    pro_overlap=False,
    k_folds=5   
)

# %%
