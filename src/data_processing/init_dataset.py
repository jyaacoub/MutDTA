import os
import sys
import itertools

# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)


from src.feature_extraction.process_msa import create_pfm_np_files
from src.data_processing.datasets import DavisKibaDataset
from src.train_test.utils import train_val_test_split

if __name__ == "__main__":
    datas = ['davis']#, 'kiba']
    FEATUREs = ['nomsa']#, 'msa', 'shannon']
    # data_root_dir = '/cluster/home/t122995uhn/projects/data/'
    data_root_dir = '/home/jyaacoub/projects/data/'

    for data, FEATURE in itertools.product(datas, FEATUREs):
        DATA_ROOT = f'{data_root_dir}/{data}/'
        print('\n', data, FEATURE)
        create_pfm_np_files(DATA_ROOT+'/aln/', processes=4)
        if data in ['davis', 'kiba']:
            dataset = DavisKibaDataset(
                    save_root=f'../data/DavisKibaDataset/{data}_{FEATURE}/',
                    data_root=DATA_ROOT,
                    aln_dir=f'{DATA_ROOT}/aln/', 
                    cmap_threshold=-0.5, 
                    feature_opt=FEATURE,
            )
    
        # saving training, validation, and test sets
        train_loader, val_loader, test_loader = train_val_test_split(dataset, 
                                train_split=0.8, val_split=0.1, random_seed=0,
                                split_by_prot=True)
        dataset.save_subset(train_loader, 'train')
        dataset.save_subset(val_loader, 'val')
        dataset.save_subset(test_loader, 'test')
            
        del dataset # free up memory
                        