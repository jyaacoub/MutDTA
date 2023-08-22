import time, os

import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from src.utils.loader import Loader
from src.data_analysis.metrics import get_metrics

from src.train_test.training import train, test
from src.train_test.utils import CheckpointSaver, init_node, init_dist_gpu, print_device_info

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
    
    media_save_p = f'results/model_media/{DATA}/'
    MODEL_STATS_CSV = 'results/model_media/model_stats.csv'
    model_save_dir = 'results/model_checkpoints/ours/'
    MODEL_KEY = Loader.get_model_key(MODEL,DATA,FEATURE,EDGEW,
                                     BATCH_SIZE,LEARNING_RATE,DROPOUT,EPOCHS)
    MODEL_KEY = "DDP-" + MODEL_KEY
    
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
    loaders = {}
    for d in ['train', 'test', 'val']:
        dataset = Loader.load_dataset(DATA, FEATURE, subset=d)
        sampler = DistributedSampler(dataset, shuffle=True, 
                                    num_replicas=args.world_size,
                                    rank=args.rank, seed=0)
        bs = 1 if d == 'test' else BATCH_SIZE
        loader = DataLoader(dataset=dataset, 
                                sampler = sampler,
                                batch_size=bs, # batch size per gpu (https://stackoverflow.com/questions/73899097/distributed-data-parallel-ddp-batch-size)
                                num_workers = args.slurm_cpus_per_task, # number of subproc used for data loading
                                pin_memory = True,
                                drop_last = True
                                )
        loaders[d] = loader
    print(f"Data loaded")
    
    # ==== Load model ====
    # args.gpu is the local rank for this process
    model = Loader.load_model(MODEL,FEATURE, EDGEW, DROPOUT).cuda(args.gpu)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    
    # ==== train ====
    cp_saver = CheckpointSaver(model=model, save_path=f'{model_save_dir}/{MODEL_KEY}.model',
                            train_all=False,
                            patience=10, min_delta=0.1,
                            save_freq=10,
                            dist_rank=args.rank)
    if os.path.exists(cp_saver.save_path):
        print('# Model already trained')
    else:
        print("starting training:")
        train(model, loaders['train'], loaders['val'], args.gpu, EPOCHS, 
              LEARNING_RATE, cp_saver)
        
        cp_saver.save()
    
    # ==== Evaluate ====
    loss, pred, actual = test(model, loaders['test'], args.gpu)
    print("Test loss:", loss)
    if args.rank == 0:
        get_metrics(actual, pred,
                    save_results=True,
                    save_path=media_save_p,
                    model_key=MODEL_KEY,
                    csv_file=MODEL_STATS_CSV,
                    show=False,
                    )