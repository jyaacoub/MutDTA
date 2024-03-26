import os
import sys
import itertools
from typing import Iterable
from src.utils import config as cfg

# Add the project root directory to Python path so imports work if file is run
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.data_prep.feature_extraction.protein_nodes import create_pfm_np_files
from src.data_prep.datasets import DavisKibaDataset, PDBbindDataset, PlatinumDataset
from src.train_test.splitting import train_val_test_split, balanced_kfold_split

def create_datasets(data_opt:Iterable[str], feat_opt:Iterable[str], edge_opt:Iterable[str],
                    pro_overlap:bool=False, data_root:str=cfg.DATA_ROOT, 
                    ligand_features:Iterable[str]=['original'],
                    ligand_edges:Iterable[str]=['binary'],
                    k_folds:int=None,
                    random_seed:int=0,
                    train_split:float=0.8,
                    val_split:float=0.1,
                    overwrite=True, 
                    **kwargs) -> None:
    """
    Creates the datasets for the given data, feature, and edge options.

    Parameters
    ----------
    `data_opt` : Iterable[str]
        The datasets to create.
    `feat_opt` : Iterable[str]
        The protein feature options to use.
    `edge_opt` : Iterable[str]
        The protein edge weight options to use.
    `pro_overlap` : bool
        Whether or not to create datasets with overlapping proteins, by default 
        False
    `data_root` : str, optional
        The root directory for the datasets, by default cfg.DATA_ROOT
    `ligand_features` : Iterable[str], optional
        Ligand features to use, by default ['original']
    `ligand_edges` : Iterable[str], optional
        Ligand edges to use, by default 'binary'
    `k_folds` : int, optional
        If not None, the number of folds to split the final training set into for 
        cross validation, by default None
    """
    
    # Loop through all combinations of data, feature, and edge options
    for data,     FEATURE,      EDGE, ligand_feature, ligand_edge in itertools.product(
        data_opt, feat_opt, edge_opt, ligand_features, ligand_edges):
        
        print('\n', data, FEATURE, EDGE)
        if data in ['davis', 'kiba']:
            if FEATURE == 'msa':
                # position frequency matrix creation -> important for msa feature
                create_pfm_np_files(f'{data_root}/{data}/aln', processes=4)
            if 'af_conf_dir' not in kwargs:
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
        elif data == 'Platinum':
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
        if k_folds is None:
            train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                                    train_split=train_split, val_split=val_split, 
                                    random_seed=random_seed, split_by_prot=not pro_overlap)
        else:
            test_split = 1 - train_split - val_split
            assert test_split > 0, f"Invalid train/val/test split: {train_split}/{val_split}/{test_split}"
            assert not pro_overlap, f"No support for overlapping proteins with k-folds rn."
            train_loader, val_loader, test_loader = balanced_kfold_split(dataset, 
                                    k_folds=k_folds, test_split=test_split,
                                    random_seed=random_seed) # only non-overlapping splits for k-folds
            
        subset_names = ['train', 'val', 'test']
        if pro_overlap:
            subset_names = [s+'-overlap' for s in subset_names]
        
        if k_folds is None:
            dataset.save_subset(train_loader, subset_names[0])
            dataset.save_subset(val_loader, subset_names[1])
        else:
            # loops through all k folds and saves as train1, train2, etc.
            dataset.save_subset_folds(train_loader, subset_names[0])
            dataset.save_subset_folds(val_loader, subset_names[1])
            
        dataset.save_subset(test_loader, subset_names[2])
            
        del dataset # free up memory

if __name__ == "__main__":
    create_datasets(data_opt=['davis'], # 'PDBbind' 'kiba' davis
                feat_opt=['nomsa'],    # nomsa 'msa' 'shannon']
                edge_opt=['af2-anm'], # for anm and af2 we need structures! (see colabfold-highQ)
                pro_overlap=False,
                #/home/jyaacoub/projects/data/
                #'/cluster/home/t122995uhn/projects/data/'
                data_root='/cluster/home/t122995uhn/projects/data/')