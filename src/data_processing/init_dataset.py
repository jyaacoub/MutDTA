import os
import sys
import itertools
from typing import Iterable

# Add the project root directory to Python path so imports work if file is run
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.feature_extraction.protein_nodes import create_pfm_np_files
from src.data_processing.datasets import DavisKibaDataset, PDBbindDataset, PlatinumDataset
from src.train_test.utils import train_val_test_split

def create_datasets(data_opt:Iterable[str], feat_opt:Iterable[str], edge_opt:Iterable[str],
                    pro_overlap:bool, data_root_dir:str, 
                    ligand_features:Iterable[str]=['original'],
                    ligand_edges:Iterable[str]='binary') -> None:
    for data,     FEATURE,      EDGE, ligand_feature, ligand_edge in itertools.product(
        data_opt, feat_opt, edge_opt, ligand_features, ligand_edges):
        
        print('\n', data, FEATURE, EDGE)
        if data in ['davis', 'kiba']:
            if FEATURE == 'msa':
                # position frequency matrix creation -> important for msa feature
                create_pfm_np_files(f'{data_root_dir}/{data}/aln', processes=4)
                
            dataset = DavisKibaDataset(
                    save_root=f'{data_root_dir}/DavisKibaDataset/{data}/',
                    data_root=f'{data_root_dir}/{data}/',
                    aln_dir=f'{data_root_dir}/{data}/aln/', 
                    cmap_threshold=-0.5, 
                    feature_opt=FEATURE,
                    af_conf_dir=f'../colabfold/{data}_af2_out/', # colabfold not needed if no structure required methods are used (see config)
                    edge_opt=EDGE,
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge
            )
        elif data == 'PDBbind':
            dataset = PDBbindDataset(
                    save_root=f'{data_root_dir}/PDBbindDataset/',
                    data_root=f'{data_root_dir}/v2020-other-PL/',
                    aln_dir=f'{data_root_dir}/PDBbind_a3m',
                    cmap_threshold=8.0,
                    overwrite=False, # overwrite old cmap.npy files
                    af_conf_dir='../colabfold/pdbbind_af2_out/',
                    feature_opt=FEATURE,
                    edge_opt=EDGE,
                    ligand_feature=ligand_feature,
                    ligand_edge=ligand_edge
                    )
        elif data == 'Platinum':
            dataset = PlatinumDataset(
                save_root=f'{data_root_dir}/PlatinumDataset/',
                data_root=f'{data_root_dir}/PlatinumDataset/raw',
                aln_dir=None,
                cmap_threshold=8.0,
                feature_opt=FEATURE,
                edge_opt=EDGE,
                ligand_feature=ligand_feature,
                ligand_edge=ligand_edge
                )
        
        # saving training, validation, and test sets
        train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                                train_split=0.8, val_split=0.1, random_seed=0,
                                split_by_prot=not pro_overlap)
        if pro_overlap:
            dataset.save_subset(train_loader, 'train-overlap')
            dataset.save_subset(val_loader, 'val-overlap')
            dataset.save_subset(test_loader, 'test-overlap')
        else:
            dataset.save_subset(train_loader, 'train')
            dataset.save_subset(val_loader, 'val')
            dataset.save_subset(test_loader, 'test')
            
        del dataset # free up memory

if __name__ == "__main__":
    create_datasets(data_opt=['davis'], # 'PDBbind' 'kiba' davis
                feat_opt=['nomsa'],    # nomsa 'msa' 'shannon']
                edge_opt=['af2-anm'], # for anm and af2 we need structures! (see colabfold-highQ)
                pro_overlap=False,
                #/home/jyaacoub/projects/data/
                #'/cluster/home/t122995uhn/projects/data/'
                data_root_dir='/cluster/home/t122995uhn/projects/data/')