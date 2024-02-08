# TODO: create a trainer class for modularity
from functools import wraps
from typing import Iterable
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from src.models.utils import BaseModel
from src.models.lig_mod import ChemDTA, ChemEsmDTA
from src.models.esm_models import EsmDTA, SaProtDTA
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.data_prep.datasets import PDBbindDataset, DavisKibaDataset
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
    edge_opt = cfg.PRO_EDGE_OPT
    data_opt = cfg.DATA_OPT
    pro_feature_opt = cfg.PRO_FEAT_OPT
    
    @staticmethod
    @validate_args({'model': model_opt, 'data':data_opt, 'edge': edge_opt, 'pro_feature': pro_feature_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def get_model_key(model:str, data:str, pro_feature:str, edge:str,
                      batch_size:int, lr:float, dropout:float, n_epochs:int, pro_overlap:bool=False,
                      fold:int=None, ligand_feature:str=None, ligand_edge:str=None):
        data += f'{fold}' if fold is not None else '' # for cross-val
        data += '-overlap' if pro_overlap else ''
        
        if model in ['EAT']: # no edgew or features for this model type
            print('WARNING: edge weight and feature opt is not supported with the specified model.')
            model_key = f'{model}M_{data}D_{batch_size}B_{lr}LR_{dropout}D_{n_epochs}E'
        else:
            model_key = f'{model}M_{data}D_{pro_feature}F_{edge}E_{batch_size}B_{lr}LR_{dropout}D_{n_epochs}E'
        
        # add ligand modifications if specified
        if ligand_feature is not None:
            model_key += f'_{ligand_feature}LF'
        if ligand_edge is not None:
            model_key += f'_{ligand_edge}LE'
        return model_key
    
    def init_test_model():
        """ Loads original DGraphDTA model for testing """
        
        return Loader.init_model("DG", "nomsa", "binary", 0.5)
        
    @staticmethod
    @validate_args({'model': model_opt, 'edge': edge_opt, 'pro_feature': pro_feature_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def init_model(model:str, pro_feature:str, pro_edge:str, dropout:float, 
                   ligand_feature:str=None, ligand_edge:str=None, **kwargs) -> BaseModel:
        """
        kwargs are used to pass additional arguments to the model constructor 
        (e.g.: pro_emb_dim, extra_profc_layer, dropout_prot_p for EsmDTA)

        Parameters
        ----------
        `model` : str
            _description_
        `pro_feature` : str
            _description_
        `pro_edge` : str
            _description_
        `dropout` : float
            _description_
        `ligand_feature` : str, optional
            _description_, by default None
        `ligand_edge` : str, optional
            _description_, by default None

        Returns
        -------
        BaseModel
            _description_
        """
        # node and edge features that dont change architecture are changed at the dataset level and not model level (e.g.: nomsa)
        # here they are only used to set the input dimensions:
        num_feat_pro = 34 if pro_feature == 'shannon' else 54
                
        if model == 'DG':
            model = DGraphDTA(num_features_pro=num_feat_pro, 
                            dropout=dropout, edge_weight_opt=pro_edge)
        elif model == 'DGI':
            model = DGraphDTAImproved(num_features_pro=num_feat_pro, output_dim=128, # 128 is the same as the original model
                                    dropout=dropout, edge_weight_opt=pro_edge)
        elif model == 'ED':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320, # only esm features
                        # pro_emb_dim=54, # inital embedding size after first GCN layer # this is the default
                        dropout=dropout,
                        pro_feat='esm_only',
                        edge_weight_opt=pro_edge,
                        **kwargs)
        elif model == 'EDA':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320+num_feat_pro, # esm features + other features
                        pro_emb_dim=54, # inital embedding size after first GCN layer
                        dropout=dropout,
                        pro_feat='all', # to include all feats (esm + 52 from DGraphDTA)
                        edge_weight_opt=pro_edge)
        elif model == 'EDI':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320,
                        pro_emb_dim=512, # increase embedding size
                        dropout=dropout,
                        pro_feat='esm_only',
                        edge_weight_opt=pro_edge,
                        **kwargs)
        elif model == 'SPD':
            assert pro_feature == 'foldseek', 'Please load up the correct dataset! SaProt only supports foldseek features.'
            #SaProt
            model = SaProtDTA(esm_head='westlake-repl/SaProt_35M_AF2',
                        num_features_pro=480,
                        dropout=dropout,
                        pro_feat='esm_only',
                        edge_weight_opt=pro_edge,
                        **kwargs)
        elif model == 'EDAI':
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320 + num_feat_pro,
                        pro_emb_dim=512,
                        dropout=dropout,
                        pro_feat='all',
                        edge_weight_opt=pro_edge)
        elif model == 'CD':
            # this model only needs sequence, no additional features.
            model = ChemDTA(dropout=dropout)
        elif model == 'CED':
            model = ChemEsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                num_features_pro=320,
                pro_emb_dim=512, # increase embedding size
                dropout=dropout,
                pro_feat='esm_only',
                edge_weight_opt=pro_edge)
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
    def load_datasets(data:str, pro_feature:str, edge_opt:str, path:str=cfg.DATA_ROOT,
                      subsets:Iterable[str]=['train', 'test', 'val'],
                      training_fold:int=None, # for cross-val. None for no cross-val
                      protein_overlap:bool=False, 
                      ligand_feature:str=None, ligand_edge:str=None):
        # no overlap or cross-val
        subsets_cv = subsets
        
        # training folds are identified by train1, train2, etc. 
        # (see model_key fn above)
        if training_fold is not None:
            subsets_cv = [d+str(training_fold) for d in subsets_cv]
            try:
                # making sure test set is not renamed
                subsets_cv[subsets.index('test')] = 'test'
            except ValueError:
                pass
            
        # Overlap is identified by adding '-overlap' to the subset name (after cross-val)
        if protein_overlap:
            subsets_cv = [d+'-overlap' for d in subsets_cv]
        
        loaded_datasets = {}
        for k, s in zip(subsets, subsets_cv):
            dataset = Loader.load_dataset(data, pro_feature, edge_opt, 
                                          subset=s, path=path, 
                                          ligand_feature=ligand_feature, 
                                          ligand_edge=ligand_edge)                
            loaded_datasets[k] = dataset
        return loaded_datasets
    
    @staticmethod
    @validate_args({'data': data_opt, 'pro_feature': pro_feature_opt, 'edge_opt': edge_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def load_DataLoaders(data:str=None, pro_feature:str=None, edge_opt:str=None, path:str=cfg.DATA_ROOT, 
                      datasets:Iterable[str]=['train', 'test', 'val'],
                      training_fold:int=None, # for cross-val. None for no cross-val
                      protein_overlap:bool=False, 
                      ligand_feature:str=None, ligand_edge:str=None,
                      # NOTE:  if loaded_dataset is provided batch_train is the only real argument
                      loaded_datasets:dict=None,
                      batch_train:int=64):
        # loaded_datasets is used to avoid loading the same dataset multiple times when we just want 
        # to create a new dataloader (e.g.: for testing with different batch size)
        if loaded_datasets is None:
            loaded_datasets = Loader.load_datasets(data=data, pro_feature=pro_feature, edge_opt=edge_opt, 
                                               path=path, subsets=datasets, training_fold=training_fold, 
                                               protein_overlap=protein_overlap, ligand_feature=ligand_feature, 
                                               ligand_edge=ligand_edge)
        
        loaders = {}
        for d in loaded_datasets:
            bs = 1 if d == 'test' else batch_train
            loader = DataLoader(dataset=loaded_datasets[d], 
                                batch_size=bs, 
                                shuffle=False)
            loaders[d] = loader
            
        return loaders
    
    @staticmethod
    @validate_args({'data': data_opt, 'pro_feature': pro_feature_opt, 'edge_opt': edge_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def load_distributed_DataLoaders(num_replicas:int, rank:int, seed:int, data:str, # additional args for distributed
                                     
                                     pro_feature:str, edge_opt:str, path:str=cfg.DATA_ROOT,
                                     batch_train:int=64, datasets:Iterable[str]=['train', 'test', 'val'],
                                     training_fold:int=None, # for cross-val. None for no cross-val
                                     protein_overlap:bool=False, 
                                     
                                     ligand_feature:str=None, ligand_edge:str=None,
                                     
                                     num_workers:int=4):
        
        loaded_datasets = Loader.load_datasets(data=data, pro_feature=pro_feature, edge_opt=edge_opt, 
                                               path=path, subsets=datasets, training_fold=training_fold, 
                                               protein_overlap=protein_overlap, ligand_feature=ligand_feature, 
                                               ligand_edge=ligand_edge)
        
        loaders = {}
        for d in loaded_datasets:
            dataset = loaded_datasets[d]
            sampler = DistributedSampler(dataset, shuffle=True,
                                            num_replicas=num_replicas,
                                            rank=rank, seed=seed)
                                            
            bs = 1 if d == 'test' else batch_train
            loader = DataLoader(dataset=dataset, 
                                sampler=sampler,
                                batch_size=bs, # should be per gpu batch size (local batch size)
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True) # drop last batch if not divisible by batch size
            loaders[d] = loader
            
        return loaders
        