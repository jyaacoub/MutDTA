# TODO: create a trainer class for modularity
from functools import wraps
from src.models.mut_dta import EsmDTA, EsmAttentionDTA
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing.datasets import PDBbindDataset, DavisKibaDataset

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
    edge_opt = ['simple', 'binary']
    data_opt = ['davis', 'kiba', 'PDBbind']
    pro_feature_opt = ['nomsa', 'msa', 'shannon']
    
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
    def load_dataset(data:str, pro_feature:str):
        if data == 'PDBbind':
            dataset = PDBbindDataset(save_root=f'../data/PDBbindDataset/{pro_feature}',
                    data_root='../data/v2020-other-PL/',
                    aln_dir='../data/PDBbind_aln', 
                    cmap_threshold=8.0,
                    feature_opt=pro_feature
                    )
        else:
            dataset = DavisKibaDataset(
                    save_root=f'../data/DavisKibaDataset/{data}_{pro_feature}/',
                    data_root=f'../data/{data}/',
                    aln_dir  =f'../data/{data}/aln/',
                    cmap_threshold=-0.5, 
                    feature_opt=pro_feature
                    )
            
        return dataset