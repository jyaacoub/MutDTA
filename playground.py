# %%
from src.data_analysis.figures import prepare_df, fig3_edge_feat
from src.utils import config

from transformers import AutoTokenizer, AutoModel


df = prepare_df('results/model_media/model_stats.csv')

# %%
fig3_edge_feat(df, show=True, exclude=[])

# %%
print('test')

#### ChemGPT ####

tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")
model = AutoModel.from_pretrained("ncfrey/ChemGPT-4.7M")