import time, os
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from src.models.mut_dta import EsmDTA
from src.models.prior_work import DGraphDTA

from src.data_processing.datasets import DavisKibaDataset

from src.train_test.training import train
from src.train_test.utils import init_node, init_dist_gpu, print_device_info

# distributed training fn
def dtrain(args):
    # ==== initialize the node ====
    init_node(args)
    
    
    # ==== Set up distributed training environment ====
    init_dist_gpu(args)
    
    DATA = args.data_opt[0] # only one data option for now
    FEATURE = args.feature_opt[0] # only one feature option for now
    EDGEW = args.edge_opt[0] # only one edge option for now
    MODEL = args.model_opt[0] # only one model option for now
    
    BATCH_SIZE = args.batch_size
    DROPOUT = args.dropout
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.num_epochs
    
    print(os.getcwd())
    print(f"----------------- HYPERPARAMETERS -----------------")
    print(f"         Local Batch size: {BATCH_SIZE}")
    print(f"        Global Batch size: {BATCH_SIZE*args.world_size}")
    print(f"            Learning rate: {LEARNING_RATE}")
    print(f"                  Dropout: {DROPOUT}")
    print(f"               Num epochs: {EPOCHS}")
    print(f"              Edge option: {EDGEW}")
    
    print(f'----------------- GPU INFO ------------------------')
    print_device_info(args.gpu)
    
    # ==== Load up training dataset ====
    dataset = DavisKibaDataset(
        save_root=f'../data/DavisKibaDataset/{DATA}_{FEATURE}/',
        data_root=f'../data/{DATA}/',
        aln_dir  =f'../data/{DATA}/aln/',
        cmap_threshold=-0.5, 
        feature_opt=FEATURE,
        subset='train'
        )


    #TODO: replace this with my version of train_test split to account for prot overlap
    sampler = DistributedSampler(dataset, shuffle=True, 
                                 num_replicas=args.world_size,
                                 rank=args.rank, seed=0)
    
    train_loader = DataLoader(dataset=dataset, 
                            sampler = sampler,
                            batch_size=BATCH_SIZE, # batch size per gpu (https://stackoverflow.com/questions/73899097/distributed-data-parallel-ddp-batch-size)
                            num_workers = 2, # number of subproc used for data loading
                            pin_memory = True,
                            drop_last = True
                            )
    print(f"Data loaded")
    
    # ==== Load model ====
    num_feat_pro = 54 if 'msa' in FEATURE else 34
    if MODEL == 'DG':
        model = DGraphDTA(num_features_pro=num_feat_pro, 
                          dropout=DROPOUT, edge_weight_opt=EDGEW)
    elif MODEL == 'ED':
        model = EsmDTA(esm_head='facebook/esm2_t6_8M_UR50D',
                        num_features_pro=320, # only esm features
                        pro_emb_dim=54, # inital embedding size after first GCN layer
                        dropout=DROPOUT,
                        esm_only=True,
                        edge_weight_opt=EDGEW)
    
    # args.gpu is the local rank for this process
    model.cuda(args.gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    
    # ==== train (this was modified from `train_test/training.py`):  ====
    CRITERION = torch.nn.MSELoss()
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("starting training:")
    #TODO: add validation data.
    for epoch in range(1, EPOCHS+1):
        START_T = time.time()
        
        # Training loop
        model.train()
        train_loss = 0.0
        for data in train_loader:
            batch_pro = data['protein'].cuda(args.gpu) 
            batch_mol = data['ligand'].cuda(args.gpu) 
            labels = data['y'].reshape(-1,1).cuda(args.gpu) 
            
            # Forward pass
            predictions = model(batch_pro, batch_mol)

            # Compute loss
            loss = CRITERION(predictions, labels)
            train_loss += loss.item()

            # Backward pass and optimization
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

        # Compute average training loss for the epoch
        train_loss /= len(train_loader)
        
        # Print training and validation loss for the epoch
        print(f"Epoch {epoch}/{EPOCHS}: Train Loss: {train_loss:.4f}, Time elapsed: {time.time()-START_T}")