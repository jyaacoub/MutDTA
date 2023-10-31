# TODO: create a trainer class for modularity
from functools import wraps
from typing import Iterable
from torch_geometric.loader import DataLoader

from src.models.lig_mod import DGraphDTALigand
from src.models.pro_mod import EsmDTA, EsmAttentionDTA
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_processing.datasets import PDBbindDataset, DavisKibaDataset
from src.utils import config  as cfg # sets up os env for HF

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
    model_opt = cfg.MODEL_OPT
    edge_opt = cfg.EDGE_OPT
    data_opt = cfg.DATA_OPT
    pro_feature_opt = cfg.PRO_FEAT_OPT
    
    @staticmethod
    @validate_args({'model': model_opt, 'data':data_opt, 'edge': edge_opt, 'pro_feature': pro_feature_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def get_model_key(model:str, data:str, pro_feature:str, edge:str, ligand_feature:str, ligand_edge:str,
                      batch_size:int, lr:float, dropout:float, n_epochs:int, pro_overlap:bool=False):
        data += '-overlap' if pro_overlap else ''
        
        if model in ['EAT']: # no edgew or features for this model type
            print('WARNING: edge weight and feature opt is not supported with the specified model.')
            model_key = f'{model}M_{data}D_{batch_size}B_{lr}LR_{dropout}D_{n_epochs}E'
        else:
            model_key = f'{model}M_{data}D_{pro_feature}F_{edge}E_{batch_size}B_{lr}LR_{dropout}D_{n_epochs}E'
        
        return model_key + f'_{ligand_feature}LF_{ligand_edge}LE'
        
    @staticmethod
    @validate_args({'model': model_opt, 'edge': edge_opt, 'pro_feature': pro_feature_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def init_model(model:str, pro_feature:str, edge:str, dropout:float, ligand_feature:str=None, ligand_edge:str=None):
        num_feat_pro = 54 if 'msa' in pro_feature else 34
        
        if (ligand_feature is not None and ligand_feature != 'original') or \
            (ligand_edge is not None and ligand_edge != 'binary'):
            print('WARNING: currently no support for combining pro and lig modifications, using original pro features.')
            #TODO: add support for above. 
            return DGraphDTALigand(ligand_feature, ligand_edge)
        
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
                        pro_feat='esm_only',
                        edge_weight_opt=edge)
        elif model == 'EDA':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320+num_feat_pro, # esm features + other features
                        pro_emb_dim=54, # inital embedding size after first GCN layer
                        dropout=dropout,
                        pro_feat='all', # to include all feats
                        edge_weight_opt=edge)
        elif model == 'EDI':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320,
                        pro_emb_dim=512, # increase embedding size
                        dropout=dropout,
                        pro_feat='esm_only',
                        edge_weight_opt=edge)
        elif model == 'EDAI':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320 + num_feat_pro,
                        pro_emb_dim=512,
                        dropout=dropout,
                        pro_feat='all',
                        edge_weight_opt=edge)
        elif model == 'EAT':
            # this model only needs protein sequence, no additional features.
            model = EsmAttentionDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                                    dropout=dropout)
            
        return model
    
    @staticmethod
    @validate_args({'data': data_opt, 'pro_feature': pro_feature_opt, 'edge_opt': edge_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def load_dataset(data:str, pro_feature:str, edge_opt:str, subset:str=None, path:str=cfg.DATA_ROOT, 
                     ligand_feature:str='original', ligand_edge:str='binary'):
        # subset is used for train/val/test split.
        # can also be used to specify the cross-val fold used by train1, train2, etc.
        if data == 'PDBbind':
            dataset = PDBbindDataset(save_root=f'{path}/PDBbindDataset',
                    data_root=f'{path}/v2020-other-PL/',
                    aln_dir=f'{path}/PDBbind_a3m', 
                    cmap_threshold=8.0,
                    feature_opt=pro_feature,
                    edge_opt=edge_opt,
                    subset=subset,
                    af_conf_dir='../colabfold/pdbbind_af2_out/out0',
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge,
                    max_seq_len=1500
                    )
        elif data in ['davis', 'kiba']:
            dataset = DavisKibaDataset(
                    save_root=f'{path}/DavisKibaDataset/{data}/',
                    data_root=f'{path}/{data}/',
                    aln_dir  =f'{path}/{data}/aln/',
                    cmap_threshold=-0.5, 
                    feature_opt=pro_feature,
                    edge_opt=edge_opt,
                    subset=subset,
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge,
                    max_seq_len=1500
                    )
        else:
            raise Exception(f'Invalid data option, pick from {Loader.data_opt}')
            
        return dataset
    
    @staticmethod
    @validate_args({'data': data_opt, 'pro_feature': pro_feature_opt, 'edge_opt': edge_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def load_DataLoaders(data:str, pro_feature:str, edge_opt:str, path:str=cfg.DATA_ROOT, 
                      batch_train:int=64, datasets:Iterable[str]=['train', 'test', 'val'],
                      protein_overlap:bool=False, 
                     ligand_feature:str=None, ligand_edge:str=None):
        loaders = {}
        
        # different list for subset so that loader keys are the same name as input
        if protein_overlap:
            subsets = [d+'-overlap' for d in datasets]
        else:
            subsets = datasets
            
        for d, s in zip(datasets, subsets):
            dataset = Loader.load_dataset(data, pro_feature, edge_opt, 
                                          subset=s, path=path, 
                                          ligand_feature=ligand_feature, 
                                          ligand_edge=ligand_edge)                
                                            
            bs = 1 if d == 'test' else batch_train
            loader = DataLoader(dataset=dataset, batch_size=bs, 
                                shuffle=False)
            loaders[d] = loader
            
        return loaders
        