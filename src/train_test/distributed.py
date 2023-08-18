import time, os
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from src.models.mut_dta import EsmDTA
from src.models.prior_work import DGraphDTA

from src.utils.loader import Loader

from src.train_test.training import train
from src.train_test.utils import init_node, init_dist_gpu, print_device_info

# distributed training fn
def dtrain(args):
    # ==== initialize the node ====
    init_node(args)
    
    
    # ==== Set up distributed training environment ====
    init_dist_gpu(args)
    
    # TODO: update this to loop through all options.
    DATA = args.data_opt[0] # only one data option for now
    FEATURE = args.feature_opt[0] # only one feature option for now
    EDGEW = args.edge_opt[0] # only one edge option for now
    MODEL = args.model_opt[0] # only one model option for now
    
    BATCH_SIZE = args.batch_size
    DROPOUT = args.dropout
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.num_epochs
    
    print(os.getcwd())
    print(f"----------------- DISTRIBUTED ARGS -----------------")
    print(f"         Local Batch size: {BATCH_SIZE}")
    print(f"        Global Batch size: {BATCH_SIZE*args.world_size}")
    print(f"                      GPU: {args.gpu}")
    print(f"                     Rank: {args.rank}")
    print(f"               World Size: {args.world_size}")

    
    print(f'----------------- GPU INFO ------------------------')
    print_device_info(args.gpu)
    
    # ==== Load up training dataset ====
    train_dataset = Loader.load_dataset(DATA, FEATURE, subset='train')
    sampler = DistributedSampler(train_dataset, shuffle=True, 
                                 num_replicas=args.world_size,
                                 rank=args.rank, seed=0)
    
    train_loader = DataLoader(dataset=train_dataset, 
                            sampler = sampler,
                            batch_size=BATCH_SIZE, # batch size per gpu (https://stackoverflow.com/questions/73899097/distributed-data-parallel-ddp-batch-size)
                            num_workers = args.slurm_cpus_per_task, # number of subproc used for data loading
                            pin_memory = True,
                            drop_last = True
                            )
    print(f"Data loaded")
    
    # ==== Load model ====
    # args.gpu is the local rank for this process
    model = Loader.load_model(MODEL,FEATURE, EDGEW, DROPOUT).cuda(args.gpu)
    
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