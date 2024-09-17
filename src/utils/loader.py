import os
import logging
from functools import wraps
from typing import Iterable
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from src.models.utils import BaseModel
from src.models.lig_mod import ChemDTA, ChemEsmDTA
from src.models.esm_models import EsmDTA, SaProtDTA
from src.models.prior_work import DGraphDTA, DGraphDTAImproved
from src.models.ring3 import Ring3DTA
from src.models.gvp_models import GVPModel, GVPLigand_DGPro, GVPLigand_RNG3, GVPL_ESM
from src.data_prep.datasets import PDBbindDataset, DavisKibaDataset, PlatinumDataset, BaseDataset
from src.utils import config  as cfg # sets up os env for HF
from src import TUNED_MODEL_CONFIGS
from glob import glob


def validate_args(valid_options):
    def decorator(func):
        @wraps(func) # to maintain original fn metadata
        def wrapper(*args, **kwargs):
            for arg, value in kwargs.items():
                if value is not None and arg in valid_options and value not in valid_options[arg]:
                    raise ValueError(f'Invalid {arg} option: {value}.')
            return func(*args, **kwargs)
        return wrapper
    return decorator

##########################################################
################## Class Method ##########################
##########################################################
class Loader():
    model_opt = cfg.MODEL_OPT
    edge_opt = cfg.PRO_EDGE_OPT
    data_opt = cfg.DATA_OPT
    pro_feature_opt = cfg.PRO_FEAT_OPT
    
    @staticmethod
    @validate_args({'model': model_opt, 'data':data_opt, 'edge': edge_opt, 'pro_feature': pro_feature_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def get_model_key(model:str, data:str, pro_feature:str, edge:str,
                      batch_size:int, lr:float, dropout:float, n_epochs:int=2000, pro_overlap:bool=False,
                      fold:int=None, ligand_feature:str='original', ligand_edge:str='binary', **kwargs):
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
    @validate_args({'tuned_model':TUNED_MODEL_CONFIGS.keys()})
    def load_tuned_model(tuned_model='davis_DG', fold=0):
        MODEL_TUNED_PARAMS = TUNED_MODEL_CONFIGS[tuned_model]

        def reformat_kwargs(model_kwargs):
            return {
                'model': model_kwargs['model'],
                'data': model_kwargs['dataset'],
                'pro_feature': model_kwargs['feature_opt'],
                'edge': model_kwargs['edge_opt'],
                'batch_size': model_kwargs['batch_size'],
                'lr': model_kwargs['lr'],
                'dropout': model_kwargs['architecture_kwargs']['dropout'],
                'n_epochs': model_kwargs.get('n_epochs', 2000),  # Assuming a default value for n_epochs
                'pro_overlap': model_kwargs.get('pro_overlap', False),  # Assuming a default or None
                'fold': model_kwargs.get('fold', fold),  # Assuming a default or None
                'ligand_feature': model_kwargs['lig_feat_opt'],
                'ligand_edge': model_kwargs['lig_edge_opt']
            }
        model_kwargs = reformat_kwargs(MODEL_TUNED_PARAMS)
        MODEL_KEY = Loader.get_model_key(**model_kwargs)
        
        model_p = f'{cfg.MODEL_SAVE_DIR}/{MODEL_KEY}.model'
        glob_p = glob(model_p+'*')

        if len(glob_p) == 0:
            glob_p = glob(model_p.replace('_originalLF_binaryLE', '')+'*')

        assert len(glob_p) < 2, f"TOO MANY MODEL CHECKPOINTS FOR {model_p} - {glob_p}"
        assert len(glob_p) == 1, f"MISSING MODEL CHECKPOINT FOR {model_p}"

        # if there was a mismatch we want to update this to keep things simple
        if model_p != glob_p[0]:
            logging.warning(f'\n\t{glob_p[0]} -> \n\t{model_p}')
            os.rename(glob_p[0], model_p)
        model_p = glob_p[0]

        logging.debug(f'loading: {model_p}')
        model = Loader.init_model(model=model_kwargs['model'], pro_feature=model_kwargs['pro_feature'], 
                                pro_edge=model_kwargs['edge'], **MODEL_TUNED_PARAMS['architecture_kwargs'])
        return model, model_kwargs
    
    @staticmethod
    @validate_args({'model': model_opt, 'edge': edge_opt, 'pro_feature': pro_feature_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def init_model(model:str, pro_feature:str, pro_edge:str, dropout:float=0.2, **kwargs) -> BaseModel:
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
            pro_emb_dim = 512 # increase embedding size
            if "pro_emb_dim" in kwargs:
                if pro_emb_dim != kwargs['pro_emb_dim']:
                    logging.warning(f'pro_emb_dim changed from default of {512} to {pro_emb_dim} for model EDI')
                pro_emb_dim = kwargs['pro_emb_dim']
                del kwargs['pro_emb_dim']
                
            model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320,
                        pro_emb_dim=pro_emb_dim,
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
        elif model == 'RNG':
            model = Ring3DTA(num_features_pro=54,
                             dropout=dropout)
        elif model == cfg.MODEL_OPT.GVP:
            model = GVPModel(num_features_mol=78, **kwargs)
            
        elif model == cfg.MODEL_OPT.GVPL:
            model = GVPLigand_DGPro(num_features_pro=num_feat_pro,
                                    dropout=dropout, 
                                    edge_weight_opt=pro_edge,
                                    **kwargs)
        elif model == cfg.MODEL_OPT.GVPL_RNG:
            model = GVPLigand_RNG3(dropout=dropout, **kwargs)
        elif model == cfg.MODEL_OPT.GVPL_ESM:
            model = GVPL_ESM(pro_num_feat=320, 
                             edge_weight_opt=pro_edge, 
                             **kwargs)
        return model
        
    @staticmethod
    @validate_args({'data': data_opt, 'pro_feature': pro_feature_opt, 'edge_opt': edge_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def load_dataset(data:str, pro_feature:str=None, edge_opt:str=None, subset:str=None, 
                     path:str=cfg.DATA_ROOT,
                     ligand_feature:str='original', ligand_edge:str='binary',
                     
                     max_seq_len:int=1500):
        # subset is used for train/val/test split.
        # can also be used to specify the cross-val fold used by train1, train2, etc.
        if data == 'PDBbind':
            dataset = PDBbindDataset(save_root=f'{path}/PDBbindDataset',
                    data_root=f'{path}/v2020-other-PL/',
                    aln_dir=f'{path}/pdbbind/PDBbind_a3m', 
                    cmap_threshold=8.0,
                    feature_opt=pro_feature,
                    edge_opt=edge_opt,
                    subset=subset,
                    af_conf_dir=f'{path}/pdbbind/pdbbind_af2_out/all_ln/',
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge,
                    max_seq_len=max_seq_len
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
                    af_conf_dir='../colabfold/davis_af2_out/',
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge,
                    max_seq_len=max_seq_len
                    )
        elif data == 'platinum':
            dataset = PlatinumDataset(
                    save_root=f'{path}/PlatinumDataset/',
                    data_root=f'{path}/PlatinumDataset/raw',
                    af_conf_dir=f'{path}/PlatinumDataset/raw/alphaflow_io/out_pdb_MD-distilled/',
                    aln_dir=None,
                    cmap_threshold=8.0,
                    
                    feature_opt=pro_feature,
                    edge_opt=edge_opt,
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge,
                    subset=subset,
                )
        else:
            # Check if dataset is a string (file path) and it exists
            if isinstance(data, str) and os.path.exists(data):
                kwargs = Loader.parse_db_kwargs(data)
                return Loader.load_dataset(**kwargs, max_seq_len=max_seq_len)
            raise Exception(f'Invalid data option, pick from {Loader.data_opt}')
            
        return dataset
    
    @staticmethod
    @validate_args({'data': data_opt, 'pro_feature': pro_feature_opt, 'edge_opt': edge_opt,
                    'ligand_feature':cfg.LIG_FEAT_OPT, 'ligand_edge':cfg.LIG_EDGE_OPT})
    def load_datasets(data:str, pro_feature:str, edge_opt:str, path:str=cfg.DATA_ROOT,
                      subsets:Iterable[str]=['train', 'test', 'val'],
                      training_fold:int=None, # for cross-val. None for no cross-val
                      protein_overlap:bool=False, 
                      ligand_feature:str='original', ligand_edge:str='binary'):
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
                      ligand_feature:str='original', ligand_edge:str='binary',
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
                                     
                                     ligand_feature:str='original', ligand_edge:str='binary',
                                     
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
    
    @staticmethod
    def parse_db_kwargs(db_path):
        """
        Parses parameters given a path string to a db you want to load up. 
        If subset folder is not included then we default to 'full' for the subset
        """
        kwargs = {
            'data': None,
            'subset': 'full',
            }
        # get db class/type 
        db_path_s = [x for x in db_path.split('/') if x]
        if 'PDBbindDataset' in db_path_s:
            idx_cls = db_path_s.index('PDBbindDataset')
            kwargs['data'] = cfg.DATA_OPT.PDBbind
            if len(db_path_s) > idx_cls+2: # +2 to skip over db_params
                kwargs['subset'] = db_path_s[idx_cls+2]
                # remove from string
                db_path = '/'.join(db_path_s[:idx_cls+2])
        elif 'DavisKibaDataset' in db_path_s:
            idx_cls = db_path_s.index('DavisKibaDataset')
            kwargs['data'] = cfg.DATA_OPT.davis if db_path_s[idx_cls+1] == 'davis' else cfg.DATA_OPT.kiba
            if len(db_path_s) > idx_cls+3:
                kwargs['subset'] = db_path_s[idx_cls+3]
                db_path = '/'.join(db_path_s[:idx_cls+3])
        else:
            raise ValueError(f"Invalid path string, couldn't find db class info - {db_path_s}")
        
        # get db parameters:
        kwargs_p = {
            'pro_feature': cfg.PRO_FEAT_OPT, 
            'edge_opt': cfg.PRO_EDGE_OPT, 
            'ligand_feature': cfg.LIG_FEAT_OPT, 
            'ligand_edge': cfg.LIG_EDGE_OPT,
        }
        db_params = os.path.basename(db_path.strip('/')).split('_')
        for k, params in kwargs_p.items():
            double = "_".join(db_params[:2])
            
            if double in params:
                kwargs_p[k] = double
                db_params = db_params[2:]
            elif db_params[0] in params:
                kwargs_p[k] = db_params[0]
                db_params = db_params[1:]
            else:
                raise ValueError(f'Invalid option, did not find {double} or {db_params[0]} in {params}')
        assert len(db_params) == 0, f"still some unparsed params - {db_params}"
        
        return {**kwargs, **kwargs_p}


# decorator to allow for input to simply be the path to the dataset directory.
def init_dataset_object(strict=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get dataset argument from args or kwargs
            dataset = kwargs.get('dataset', args[0] if args else None)
            
            # Check if dataset is a string (file path) or an actual DB object
            if isinstance(dataset, str):
                if strict and not os.path.exists(dataset):
                    raise FileNotFoundError(f'Dataset does not exist - {dataset}')
                
                # Parse and build dataset
                db_kwargs = Loader.parse_db_kwargs(dataset)
                logging.info(f'Loading dataset with {db_kwargs}')
                built = Loader.load_dataset(**db_kwargs)
            elif isinstance(dataset, BaseDataset):
                built = dataset
            elif dataset is None:
                raise ValueError('Missing Dataset in args/kwargs')
            else:
                raise TypeError('Invalid format for dataset')
            
            # Add built dataset to args/kwargs
            if 'dataset' in kwargs:
                kwargs['dataset'] = built
            else:
                args = (built, *args[1:])
            
            # Return the function call output
            return func(*args, **kwargs)
        return wrapper
    return decorator