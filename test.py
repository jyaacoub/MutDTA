#%%

#%%
from src.utils import config
import torch
from torch_geometric.loader import DataLoader

from src.data_analysis.metrics import get_metrics
from src.train_test.training import train_tune, test
from src.models.prior_work import DGraphDTA
from src.models.mut_dta import EsmDTA, EsmAttentionDTA
from src.utils.loader import Loader


MODEL = 'EDI'
DATA = 'davis'
FEATURE = 'nomsa'
EDGE = 'binary'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT = 0.4
EPOCHS = 2000
checkpoint_p = lambda x: f'results/model_checkpoints/ours/{x}.model_tmp'

media_save_dir = 'results/model_media/'
media_save_p = f'{media_save_dir}/{DATA}/'
MODEL_STATS_CSV = 'results/model_media/model_stats.csv'

MODEL_KEY = Loader.get_model_key(MODEL,DATA,FEATURE,EDGE,
                                     BATCH_SIZE,LEARNING_RATE,DROPOUT,EPOCHS)
MODEL_KEY = 'DDP-' + MODEL_KEY

print(checkpoint_p(MODEL_KEY))

# %%
test_dataset = Loader.load_dataset(DATA, FEATURE, subset='test', path='../data')
test_loader = DataLoader(test_dataset, 1, shuffle=False)


#%%
model = Loader.load_model(MODEL, FEATURE, EDGE, DROPOUT)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# due to https://discuss.pytorch.org/t/check-if-model-is-wrapped-in-nn-dataparallel/67957
mdl_dict = torch.load(checkpoint_p(MODEL_KEY), map_location=device)
new_dict = {(k[7:] if 'module.' == k[:7] else k):v for k,v in mdl_dict.items()}

model.load_state_dict(new_dict)
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
