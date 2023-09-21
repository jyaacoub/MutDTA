import os
import sys
import itertools

# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)


from src.feature_extraction.process_msa import create_pfm_np_files
from src.data_processing.datasets import DavisKibaDataset, PDBbindDataset, PlatinumDataset
from src.train_test.utils import train_val_test_split

# for multiprocess create cmaps run the following (davis and kiba already have cmaps from pscons4; but we can create cmaps from kiba using providied uniprotid)
# from src.feature_extraction.protein import multi_save_cmaps
# import os
# data_root = f'../data/v2020-other-PL/'

# pdb_codes = os.listdir(data_root)
# # filter out readme and index folders
# pdb_codes = [p for p in pdb_codes if p != 'index' and p != 'readme']
# pdb_p = lambda x: os.path.join(data_root, x, f'{x}_protein.pdb')
# cmap_p = lambda x: os.path.join(data_root, x, f'{x}.npy')

# multi_save_cmaps(pdb_codes, pdb_p, cmap_p, processes=4)

if __name__ == "__main__":
    datas = ['Platinum']#['PDBbind']#, 'kiba']
    feat_opt = ['nomsa']#, 'msa', 'shannon']
    edge_opt = ['binary']
    # data_root_dir = '/cluster/home/t122995uhn/projects/data/'
    data_root_dir = '/home/jyaacoub/projects/data/'

    for data, FEATURE, EDGE in itertools.product(datas, feat_opt, edge_opt):
        print('\n', data, FEATURE)
        if data in ['davis', 'kiba']:
            DATA_ROOT = f'{data_root_dir}/'
            create_pfm_np_files(DATA_ROOT+'/aln/', processes=4) # position frequency matrix creation -> important for msa feature
            dataset = DavisKibaDataset(
                    save_root=f'../data/DavisKibaDataset/{data}_{FEATURE}/',
                    data_root=DATA_ROOT,
                    aln_dir=f'{DATA_ROOT}/aln/', 
                    cmap_threshold=-0.5, 
                    feature_opt=FEATURE,
                    af_conf_dir='', #TODO:
            )
        elif data == 'PDBbind':
            # create_pfm_np_files('../data/PDBbind_aln/', processes=4)
            dataset = PDBbindDataset(save_root=f'../data/PDBbindDataset/',
                    data_root=f'../data/v2020-other-PL/',
                    aln_dir=f'../data/PDBbind_aln',
                    cmap_threshold=8.0,
                    overwrite=False, # overwrite old cmap.npy files
                    af_conf_dir='../colabfold/pdbbind_out/out0'
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
                                split_by_prot=True)
        dataset.save_subset(train_loader, 'train')
        dataset.save_subset(val_loader, 'val')
        dataset.save_subset(test_loader, 'test')
            
        del dataset # free up memory
                        