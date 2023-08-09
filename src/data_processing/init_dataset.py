import os
import sys
import itertools

# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)


from src.feature_extraction.process_msa import create_pfm_np_files
from src.data_processing.datasets import DavisKibaDataset

if __name__ == "__main__":
    datas = ['davis', 'kiba']
    FEATUREs = ['nomsa', 'msa', 'shannon']

    for data, FEATURE in itertools.product(datas, FEATUREs):
        # DATA_ROOT = f'/home/jyaacoub/projects/data/davis_kiba/{data}/'
        DATA_ROOT = f'/cluster/home/t122995uhn/projects/data/{data}/'
        print('\n', data, FEATURE)
        create_pfm_np_files(DATA_ROOT+'/aln/', processes=4)
        if FEATURE == 'nomsa':
            dataset = DavisKibaDataset(
                    save_root=f'../data/DavisKibaDataset/{data}_{FEATURE}/',
                    data_root=DATA_ROOT,
                    aln_dir=None, # set to none == no msa provided
                    cmap_threshold=-0.5, 
                    shannon=False)
        else:
            dataset = DavisKibaDataset(
                    save_root=f'../data/DavisKibaDataset/{data}_{FEATURE}/',
                    data_root=DATA_ROOT,
                    aln_dir=f'{DATA_ROOT}/aln/', 
                    cmap_threshold=-0.5, 
                    shannon=FEATURE=='shannon')
            
        del dataset # free up memory
                        