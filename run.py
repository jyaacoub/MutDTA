#%%
import os
from src.feature_extraction.protein import create_save_cmaps, get_sequence
from src.data_processing import PDBbindProcessor, Downloader
from data_processing.datasets import PDBbindDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd

pdb_raw_dir = '../data/v2020-other-PL/'
pdb_processed_dir = '../data/pytorch_PDBbind/'

#%%
pdb_dataset = PDBbindDataset(pdb_processed_dir, pdb_raw_dir)
loader = DataLoader(pdb_dataset, batch_size=32, shuffle=True)