import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader


# Creating data indices for training and validation splits:
def train_val_test_split(dataset, train_split=.8, val_split=.1, 
                         shuffle_dataset=True, random_seed=None,
                         batch_size=128, use_refined=True):
      
      if random_seed is not None: 
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
      
      dataset_size = len(dataset)
      indices = list(range(dataset_size))
      if shuffle_dataset:
            np.random.shuffle(indices)

      if use_refined:
            # test set will contain only refined complexes so that we can compare with vina
            vina_df = pd.read_csv('./results/PDBbind/vina_out/run10.csv', index_col=0)
            # cols are: PDBCode,vina_deltaG(kcal/mol),vina_kd(uM)
            test_indices = []
            train_val_indices = []
            for i in range(dataset_size):
                  code = dataset[i][0].code
                  if code in vina_df.index:
                        test_indices.append(i)
                  else:
                        train_val_indices.append(i)
                  # break once we have enough test indices to meet the size requirement
                  if len(test_indices) >= (1-(train_split+val_split))*dataset_size:
                        train_val_indices += indices[i+1:]
                        break
      else:
            # split into train_val and test
            tv_size = int(np.floor((train_split+val_split) * dataset_size))
            train_val_indices, test_indices = indices[:tv_size], indices[tv_size:]

      # split train_val into train and val
      t_size = int(np.floor(train_split * dataset_size))
      train_indices, val_indices = train_val_indices[:t_size], train_val_indices[t_size:]

      # Creating PT data samplers
      train_sampler = SubsetRandomSampler(train_indices)
      val_sampler = SubsetRandomSampler(val_indices)
      test_sampler = SubsetRandomSampler(test_indices)
      
      t_count = len(train_sampler)
      v_count = len(val_sampler)
      te_count = len(test_sampler)
      
      print(f'  Train size: {t_count}')
      print(f'    Val size: {v_count}')
      print(f'   Test size: {te_count}')
      print(f'            = {t_count+v_count+te_count}')
      print(f'Dataset size: {dataset_size}')
      
      # create dataloaders
      train_loader = DataLoader(dataset, batch_size=batch_size, 
                                sampler=train_sampler)
      val_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=val_sampler)
      test_loader = DataLoader(dataset, batch_size=batch_size,
                               sampler=test_sampler)
      
      return train_loader, val_loader, test_loader