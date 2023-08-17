# %%
from src.data_processing.datasets import DavisKibaDataset

dataset = DavisKibaDataset('/cluster/home/t122995uhn/projects/data/DavisKibaDataset/davis_nomsa',
                           subset='full')

# %%
from src.data_processing.utils import train_val_test_split

train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                        train_split=0.8, val_split=0.1,
                        shuffle_dataset=True, random_seed=31, 
                        batch_train=10, use_refined=False,
                        split_by_prot=True)
# %%
dataset.save_subset(train_loader, 'train')

# %%
