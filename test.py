#%%
from src.utils import config
import torch
import os
from torch_geometric.loader import DataLoader

from src.data_analysis.metrics import get_metrics
from src.train_test.training import test
from src.utils.loader import Loader
from src.utils.arg_parse import parse_train_test_args
args = parse_train_test_args(verbose=True,
                             jyp_args=' -m EDI -d PDBbind -f nomsa -e anm -lr 0.0001 -bs 20 -do 0.4 -ne 2000')
#%%

MODEL = args.model_opt[0]
DATA = args.data_opt[0]
FEATURE = args.feature_opt[0]
EDGE = args.edge_opt[0]

# Model Hyperparameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DROPOUT = args.dropout
EPOCHS = args.num_epochs

checkpoint_p_tmp = lambda x: f'results/model_checkpoints/ours/{x}.model_tmp'
checkpoint_p = lambda x: f'results/model_checkpoints/ours/{x}.model'

media_save_dir = 'results/model_media/'
media_save_p = f'{media_save_dir}/{DATA}/'
MODEL_STATS_CSV = 'results/model_media/model_stats.csv'

MODEL_KEY = Loader.get_model_key(MODEL,DATA,FEATURE,EDGE,
                                     BATCH_SIZE,LEARNING_RATE,DROPOUT,EPOCHS,
                                     pro_overlap=args.protein_overlap)
# MODEL_KEY = 'DDP-' + MODEL_KEY # distributed model
model_p = checkpoint_p(MODEL_KEY)
model_p = model_p if os.path.isfile(model_p) else checkpoint_p_tmp(MODEL_KEY)
assert os.path.isfile(model_p), f"MISSING MODEL CHECKPOINT {model_p}"

print(model_p)

# %%
subset = 'test-overlap' if args.protein_overlap else 'test'
test_dataset = Loader.load_dataset(DATA, FEATURE, EDGE, subset=subset, path='../data')
test_loader = DataLoader(test_dataset, 1, shuffle=False)


#%%
model = Loader.load_model(MODEL, FEATURE, EDGE, DROPOUT)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mdl_dict = torch.load(model_p, map_location=device)
try:
    model.load_state_dict(mdl_dict)
except RuntimeError as e:
    # if model was distributed then it will have extra "module." prefix
    # due to https://discuss.pytorch.org/t/check-if-model-is-wrapped-in-nn-dataparallel/67957
    # print("Error(s) in loading state_dict for EsmDTA")
    mdl_dict = {(k[7:] if 'module.' == k[:7] else k):v for k,v in mdl_dict.items()}
    model.load_state_dict(mdl_dict)
    
model.to(device)

#%%
loss, pred, actual = test(model, test_loader, device)
print(f'# Test loss: {loss}')
get_metrics(actual, pred,
            save_results=True,
            save_path=media_save_p,
            model_key=MODEL_KEY,
            csv_file=MODEL_STATS_CSV,
            show=False,
            )
# %%
# renaming checkpoint to remove _tmp specification
model_p = checkpoint_p(MODEL_KEY)
model_p_tmp = checkpoint_p_tmp(MODEL_KEY)

if (not os.path.isfile(model_p) and  # ensuring no overwrite
    os.path.isfile(model_p_tmp)):
    os.rename(model_p_tmp, model_p)