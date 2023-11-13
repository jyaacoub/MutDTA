# %%
from src.data_processing.init_dataset import create_datasets

create_datasets(
    data_opt=['davis'],
    feat_opt=['foldseek'],
    edge_opt=['binary']
)
# %%