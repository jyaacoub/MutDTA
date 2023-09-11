# TODO: create a trainer class for modularity
from functools import wraps
from src.models.mut_dta import EsmDTA, EsmAttentionDTA
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing.datasets import PDBbindDataset, DavisKibaDataset
from src.utils import config # sets up os env for HF

def validate_args(valid_options):
    def decorator(func):
        @wraps(func) # to maintain original fn metadata
        def wrapper(*args, **kwargs):
            for arg, value in kwargs.items():
                if arg in valid_options and value not in valid_options[arg]:
                    raise ValueError(f'Invalid {arg} option: {value}.')
            return func(*args, **kwargs)
        return wrapper
    return decorator

class Loader():
    model_opt = ['DG', 'DGI', 'ED', 'EDA', 'EDI', 'EDAI', 'EAT']
    edge_opt = ['simple', 'binary', 'anm']
    data_opt = ['davis', 'kiba', 'PDBbind']
    pro_feature_opt = ['nomsa', 'msa', 'shannon']
    
    @staticmethod
    @validate_args({'model': model_opt, 'data':data_opt, 'edge': edge_opt, 'pro_feature': pro_feature_opt})
    def get_model_key(model:str, data:str, pro_feature:str, edge:str, 
                      batch_size:int, lr:float, dropout:float, n_epochs:int):
        if model in ['EAT']: # no edgew or features for this model type
            print('WARNING: edge weight and feature opt is not supported with the specified model.')
            return f'{model}M_{data}D_{batch_size}B_{lr}LR_{dropout}D_{n_epochs}E'
        else:
            return f'{model}M_{data}D_{pro_feature}F_{edge}E_{batch_size}B_{lr}LR_{dropout}D_{n_epochs}E'
        
    @staticmethod
    @validate_args({'model': model_opt, 'edge': edge_opt, 'pro_feature': pro_feature_opt})
    def load_model(model:str, pro_feature:str, edge:str, dropout:float):
        num_feat_pro = 54 if 'msa' in pro_feature else 34
        
        if model == 'DG':
            model = DGraphDTA(num_features_pro=num_feat_pro, 
                            dropout=dropout, edge_weight_opt=edge)
        elif model == 'DGI':
            model = DGraphDTAImproved(num_features_pro=num_feat_pro, output_dim=128, # 128 is the same as the original model
                                    dropout=dropout, edge_weight_opt=edge)
        elif model == 'ED':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320, # only esm features
                        pro_emb_dim=54, # inital embedding size after first GCN layer
                        dropout=dropout,
                        esm_only=True,
                        edge_weight_opt=edge)
        elif model == 'EDA':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320+num_feat_pro, # esm features + other features
                        pro_emb_dim=54, # inital embedding size after first GCN layer
                        dropout=dropout,
                        esm_only=False, # false to include all feats
                        edge_weight_opt=edge)
        elif model == 'EDI':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320,
                        pro_emb_dim=512, # increase embedding size
                        dropout=dropout,
                        esm_only=True,
                        edge_weight_opt=edge)
        elif model == 'EDAI':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320 + num_feat_pro,
                        pro_emb_dim=512,
                        dropout=dropout,
                        esm_only=False,
                        edge_weight_opt=edge)
        elif model == 'EAT':
            # this model only needs protein sequence, no additional features.
            model = EsmAttentionDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                                    dropout=dropout)
            
        return model
    
    @staticmethod
    @validate_args({'data': data_opt, 'pro_feature': pro_feature_opt})
    def load_dataset(data:str, pro_feature:str, subset:str=None, path:str='../data/'):
        if data == 'PDBbind':
            dataset = PDBbindDataset(save_root=f'{path}/PDBbindDataset/{pro_feature}',
                    data_root=f'{path}/v2020-other-PL/',
                    aln_dir=f'{path}/PDBbind_aln', 
                    cmap_threshold=8.0,
                    feature_opt=pro_feature,
                    subset=subset
                    )
        elif data in ['davis', 'kiba']:
            dataset = DavisKibaDataset(
                    save_root=f'{path}/DavisKibaDataset/{data}_{pro_feature}/',
                    data_root=f'{path}/{data}/',
                    aln_dir  =f'{path}/{data}/aln/',
                    cmap_threshold=-0.5, 
                    feature_opt=pro_feature,
                    subset=subset
                    )
        else:
            raise Exception(f'Invalid data option, pick from {Loader.data_opt}')
            
        return dataset