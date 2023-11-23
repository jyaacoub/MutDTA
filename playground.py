# %%
import torch
from src.utils import config as cfg
from src.utils.loader import Loader

model = Loader.init_model('SPD', pro_feature='nomsa', pro_edge='binary', dropout=0.4)

# %%
d = Loader.load_dataset(data='davis',
                    pro_feature='foldseek',
                    edge_opt='binary',
                    subset='val0')

dl = Loader.load_DataLoaders(loaded_datasets={'val0':d}, batch_train=2)['val0']
# %% ESM emb ####
# cls and sep tokens are added to the sequence by the tokenizer
data = next(iter(dl))

out = model(data['protein'], data['ligand'])
# seq_tok = model.esm_tok(data.pro_seq, 
#                         return_tensors='pt', 
#                         padding=True) # [B, L_max+2]
# seq_tok['input_ids'] = seq_tok['input_ids'].to(data.x.device)
# seq_tok['attention_mask'] = seq_tok['attention_mask'].to(data.x.device)

# esm_emb = model.esm_mdl(**seq_tok).last_hidden_state # [B, L_max+2, emb_dim]

# # mask tokens dont make it through to the final output 
# # thus the final output is the same length as if we were to run it through the original ESM

# #%% removing <cls> token
# esm_emb = esm_emb[:,1:,:] # [B, L_max+1, emb_dim]

# # %% removing <sep>/<eos> and <pad> token by applying mask
# # for saProt token 2 == <eos>
# L_max = esm_emb.shape[1] # L_max+1
# mask = torch.arange(L_max)[None, :] < torch.tensor([len(seq)/2 #NOTE: this is the main difference from normal ESM since the input sequence includes SA tokens
#     for seq in data.pro_seq])[:, None]
# mask = mask.flatten(0,1) # [B*L_max+1]

# #%% flatten from [B, L_max+1, emb_dim] 
# esm_emb = esm_emb.flatten(0,1) # to [B*L_max+1, emb_seqdim]
# esm_emb = esm_emb[mask] # [B*L, emb_dim]

# #%%
# if model.esm_only:
#     target_x = esm_emb # [B*L, emb_dim]
# else:
#     # append esm embeddings to protein input
#     target_x = torch.cat((esm_emb, data.x), axis=1)
#     #  ->> [B*L, emb_dim+feat_dim]
# %%
