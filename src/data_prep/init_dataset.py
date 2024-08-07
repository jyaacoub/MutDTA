import os
import sys
import itertools
import pandas as pd

# Add the project root directory to Python path so imports work if file is run
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.utils import config as cfg
from src.data_prep.feature_extraction.protein_nodes import create_pfm_np_files
from src.data_prep.datasets import DavisKibaDataset, PDBbindDataset, PlatinumDataset
from src.train_test.splitting import train_val_test_split, balanced_kfold_split, resplit

def create_datasets(data_opt:list[str]|str, feat_opt:list[str]|str, edge_opt:list[str]|str,
                    ligand_features:list[str]=['original'],
                    ligand_edges:list[str]=['binary'],
                    pro_overlap:bool=False, data_root:str=cfg.DATA_ROOT, 
                    k_folds:int=None,
                    random_seed:int=0,
                    train_split:float=0.8,
                    val_split:float=0.1,
                    overwrite=True,
                    test_prots_csv:str=None,
                    val_prots_csv:list[str]=None,
                    **kwargs) -> None:
    """
    Creates the datasets for the given data, feature, and edge options.

    Parameters
    ----------
    `data_opt` : list[str]|str
        The datasets to create.
    `feat_opt` : list[str]|str
        The protein feature options to use.
    `edge_opt` : list[str]|str
        The protein edge weight options to use.
    `pro_overlap` : bool
        Whether or not to create datasets with overlapping proteins, by default 
        False
    `data_root` : str, optional
        The root directory for the datasets, by default cfg.DATA_ROOT
    `ligand_features` : list[str], optional
        Ligand features to use, by default ['original']
    `ligand_edges` : list[str], optional
        Ligand edges to use, by default 'binary'
    `k_folds` : int, optional
        If not None, the number of folds to split the final training set into for 
        cross validation, by default None
    `test_prots_csv` : str, optional
        If not None, the path to a csv file containing the test proteins to use, 
        by default None. The csv file should have a 'prot_id' column.

    """
    if isinstance(data_opt, str): data_opt = [data_opt]
    if isinstance(feat_opt, str): feat_opt = [feat_opt]
    if isinstance(edge_opt, str): edge_opt = [edge_opt]
    if isinstance(ligand_features, str): ligand_features = [ligand_features]
    if isinstance(ligand_edges, str): ligand_edges = [ligand_edges]
    
    # Loop through all combinations of data, feature, and edge options
    for data,     FEATURE,      EDGE, ligand_feature, ligand_edge in itertools.product(
        data_opt, feat_opt, edge_opt, ligand_features, ligand_edges):
        
        print('\n', data, FEATURE, EDGE, ligand_feature, ligand_edge)
        if data in ['davis', 'kiba']:
            if FEATURE == 'msa':
                # position frequency matrix creation -> important for msa feature
                create_pfm_np_files(f'{data_root}/{data}/aln', processes=4)
            if 'af_conf_dir' not in kwargs:
                if EDGE in cfg.OPT_REQUIRES_AFLOW_CONF:
                    kwargs['af_conf_dir'] = f'{data_root}/{data}/alphaflow_io/out_pdb_MD-distilled/'
                else:
                    kwargs['af_conf_dir'] = f'../colabfold/{data}_af2_out/'
                
                
            dataset = DavisKibaDataset(
                    save_root=f'{data_root}/DavisKibaDataset/{data}/',
                    data_root=f'{data_root}/{data}/',
                    aln_dir=f'{data_root}/{data}/aln/', 
                    cmap_threshold=-0.5, 
                    feature_opt=FEATURE,
                    overwrite=overwrite,
                    edge_opt=EDGE,
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge,
                    **kwargs
            )
        elif data == 'PDBbind':
            if 'af_conf_dir' not in kwargs:
                if EDGE in cfg.OPT_REQUIRES_AFLOW_CONF:
                    kwargs['af_conf_dir'] = f'{data_root}/pdbbind/alphaflow_io/out_pid_ln/'
                else:
                    kwargs['af_conf_dir'] = f'{data_root}/pdbbind/pdbbind_af2_out/all_ln/'
            dataset = PDBbindDataset(
                    save_root=f'{data_root}/PDBbindDataset/',
                    data_root=f'{data_root}/pdbbind/v2020-other-PL/',
                    aln_dir=f'{data_root}/pdbbind/PDBbind_aln/',
                    cmap_threshold=8.0,
                    overwrite=overwrite, # overwrite old cmap.npy files
                    feature_opt=FEATURE,
                    edge_opt=EDGE,
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge,
                    **kwargs
                    )
        elif data == 'platinum':
            if 'af_conf_dir' not in kwargs:
                kwargs['af_conf_dir'] = f'{data_root}/PlatinumDataset/raw/alphaflow_io/out_pdb_MD-distilled/'
            dataset = PlatinumDataset(
                save_root=f'{data_root}/PlatinumDataset/',
                data_root=f'{data_root}/PlatinumDataset/raw',
                aln_dir=None,
                cmap_threshold=8.0,
                overwrite=overwrite,
                feature_opt=FEATURE,
                edge_opt=EDGE,
                ligand_feature=ligand_feature,
                ligand_edge=ligand_edge,
                **kwargs
                )
        else:
            raise ValueError(f"Invalid data type {data}, pick from {cfg.DATA_OPT.list()}.")
        
        # saving training, validation, and test sets
        test_split = 1 - train_split - val_split
        if val_prots_csv:
            assert k_folds is None or len(val_prots_csv) == k_folds, "Mismatch between number of val_prot_csvs provided and k_folds selected."
            
            split_files = {os.path.basename(v).split('.')[0]: v for v in val_prots_csv}
            split_files['test'] = test_prots_csv
            dataset = resplit(dataset, split_files=split_files)
        else:
            if k_folds is None:
                train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                                        train_split=train_split, val_split=val_split, 
                                        random_seed=random_seed, split_by_prot=not pro_overlap)
            else:
                assert test_split > 0, f"Invalid train/val/test split: {train_split}/{val_split}/{test_split}"
                assert not pro_overlap, f"No support for overlapping proteins with k-folds rn."
                if test_prots_csv is not None:
                    df = pd.read_csv(test_prots_csv)
                    test_prots = set(df['prot_id'].tolist())
                else:
                    test_prots = None
                
                train_loader, val_loader, test_loader = balanced_kfold_split(dataset, 
                                        k_folds=k_folds, test_split=test_split, test_prots=test_prots,
                                        random_seed=random_seed) # only non-overlapping splits for k-folds
                
            subset_names = ['train', 'val', 'test']
            if pro_overlap:
                subset_names = [s+'-overlap' for s in subset_names]
            
            if test_split < 1: # for datasets that are purely for testing we skip this section
                if k_folds is None:
                    dataset.save_subset(train_loader, subset_names[0])
                    dataset.save_subset(val_loader, subset_names[1])
                else:
                    # loops through all k folds and saves as train1, train2, etc.
                    dataset.save_subset_folds(train_loader, subset_names[0])
                    dataset.save_subset_folds(val_loader, subset_names[1])
            
            dataset.save_subset(test_loader, subset_names[2])
        
        del dataset # free up memory
