import os
import sys
import itertools
from typing import Iterable

# Add the project root directory to Python path so imports work if file is run
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from src.feature_extraction.process_msa import create_pfm_np_files
from src.data_processing.datasets import DavisKibaDataset, PDBbindDataset, PlatinumDataset
from src.train_test.utils import train_val_test_split

def create_datasets(data_opt:Iterable[str], feat_opt:Iterable[str], edge_opt:Iterable[str], 
                    pro_overlap:bool, data_root_dir:str) -> None:
    for data, FEATURE, EDGE in itertools.product(data_opt, feat_opt, edge_opt):
        print('\n', data, FEATURE)
        if data in ['davis', 'kiba']:
            DATA_ROOT = f'{data_root_dir}/{data}/'
            create_pfm_np_files(DATA_ROOT+'/aln/', processes=4) # position frequency matrix creation -> important for msa feature
            dataset = DavisKibaDataset(
                    save_root=f'../data/DavisKibaDataset/{data}/',
                    data_root=DATA_ROOT,
                    aln_dir=f'{DATA_ROOT}/aln/', 
                    cmap_threshold=-0.5, 
                    feature_opt=FEATURE,
                    af_conf_dir=None, #TODO: create af_configs for daviskiba
            )
        elif data == 'PDBbind':
            # create_pfm_np_files('../data/PDBbind_aln/', processes=4)
            dataset = PDBbindDataset(
                    save_root=f'../data/PDBbindDataset/',
                    data_root=f'../data/v2020-other-PL/',
                    aln_dir=f'../data/PDBbind_aln',
                    cmap_threshold=8.0,
                    overwrite=False, # overwrite old cmap.npy files
                    af_conf_dir='../colabfold/pdbbind_out/out0',
                    feature_opt=FEATURE,
                    edge_opt=EDGE,
                    )
        elif data == 'Platinum':
            dataset = PlatinumDataset(
                save_root=f'../data/PlatinumDataset/',
                data_root=f'../data/PlatinumDataset/raw',
                aln_dir=None,
                cmap_threshold=8.0,
                feature_opt=FEATURE,
                edge_opt=EDGE
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
    create_datasets(data_opt=['Platinum'], # 'PDBbind' 'kiba' davis
                    feat_opt=['nomsa'],    # nomsa 'msa' 'shannon']
                    edge_opt=['binary'],
                    pro_overlap=True, 
                    #/home/jyaacoub/projects/data/
                    #'/cluster/home/t122995uhn/projects/data/'
                    data_root_dir='/cluster/home/t122995uhn/projects/data/')
