#%%
import os, random, itertools, math

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing import PDBbindDataset, DavisKibaDataset, train_val_test_split
from src.models import train, test, CheckpointSaver
from src.models.utils import print_device_info
from src.data_analysis import get_metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print_device_info(device)

#/randomW_kiba_32B_0.001LR_0.2D_200E_msaF_DGraphDTAImproved.model
#randomW_davis_64B_0.0001LR_0.4D_200E_msaF_DGraphDTA.model

#%% Testing esm model 
# based on https://github.com/facebookresearch/esm#main-models-you-should-use-
# I should be using ESM-MSA-1b (esm1b_t33_650M_UR50S)
# from https://github.com/facebookresearch/esm/tree/main#pre-trained-models-
# Luke used esm2_t6_8M_UR50D for his experiments

# https://huggingface.co/facebook/esm2_t33_650M_UR50D is <10GB
# https://huggingface.co/facebook/esm2_t36_3B_UR50D is 11GB
from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM, EsmModel, EsmTokenizer

config = EsmConfig.from_pretrained('facebook/esm2_t6_8M_UR50D')
print(config)
tok = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
# this will raise a warning since lm head is missing but that is okay since we are not using it:
esm2 = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D')



# %% test:
sample_seq = 'MMPPKLM'
sample_tok = tok(sample_seq, return_tensors='pt')
# NOTE: tokenizer adds special tokens to the beginning and end of the sequence
# the first token is <cls> and the last token is <sep>
# these are for classification tasks and are not needed for our purposes
sample_out = esm2(**sample_tok)
print(sample_seq)
print(sample_tok.input_ids.shape)
print(list(sample_seq[0:5]), 'corresponds to', sample_tok.input_ids[0][1:6])

# print(sample_out)
print('Hidden state shape:', sample_out.last_hidden_state.shape)


