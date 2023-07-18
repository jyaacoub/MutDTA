import os
import sys

# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)


from src.feature_extraction.process_msa import create_pfm_np_files
from src.data_processing import DavisKibaDataset

if __name__ == "__main__":
        data = ['kiba', 'davis'][0]
        FEATUREs = ['shannon', 'msa']
        DATA_ROOT = f'/cluster/home/t122995uhn/projects/data/{data}/'
        create_pfm_np_files(DATA_ROOT+'/aln/', processes=4)

        for FEATURE in FEATUREs:
                print(data, FEATURE)
                dataset = DavisKibaDataset(
                        save_root=f'../data/DavisKibaDataset/{data}_{FEATURE}/',
                        data_root=DATA_ROOT,
                        aln_dir=f'{DATA_ROOT}/aln/',
                        cmap_threshold=-0.5, shannon=FEATURE=='shannon')
