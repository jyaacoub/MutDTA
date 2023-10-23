import time, os

import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from src.utils.loader import Loader
from src.data_analysis.metrics import get_metrics

from src.train_test.training import train, test
from src.train_test.utils import CheckpointSaver, init_node, init_dist_gpu, print_device_info

from src.utils.config import MODEL_STATS_CSV,MEDIA_SAVE_DIR, MODEL_SAVE_DIR

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
    
    media_save_p = f'{MEDIA_SAVE_DIR}/{DATA}/'
    MODEL_KEY = Loader.get_model_key(MODEL,DATA,FEATURE,EDGEW,BATCH_SIZE*args.world_size,
                                     LEARNING_RATE,DROPOUT,EPOCHS,
                                     pro_overlap=args.protein_overlap)
    # MODEL_KEY = "DDP-" + MODEL_KEY
    
    print(os.getcwd())
    print(f"---------------- MODEL OPT ---------------")
    print(f"     Selected og_model_opt: {args.model_opt}")
    print(f"         Selected data_opt: {args.data_opt}")
    print(f" Selected feature_opt list: {args.feature_opt}")
    print(f"    Selected edge_opt list: {args.edge_opt}")
    print(f"           forced training: {args.train}\n")

    print(f"-------------- HYPERPARAMETERS -----------")
    print(f"            Learning rate: {args.learning_rate}")
    print(f"                  Dropout: {args.dropout}")
    print(f"               Num epochs: {args.num_epochs}\n")
    
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
    datasets = ['train', 'test', 'val']
    # different list for subset so that loader keys are the same name as input
    if args.protein_overlap:
        subsets = [d+'-overlap' for d in datasets]
    else:
        subsets = datasets
        
    for d, s in zip(datasets, subsets):
        dataset = Loader.load_dataset(DATA, FEATURE, EDGEW, subset=s)
        sampler = DistributedSampler(dataset, shuffle=True, 
                                    num_replicas=args.world_size,
                                    rank=args.rank, seed=args.rand_seed)
        bs = 1 if d == 'test' else BATCH_SIZE
        loader = DataLoader(dataset=dataset, 
                                sampler=sampler,
                                batch_size=bs, # batch size per gpu (https://stackoverflow.com/questions/73899097/distributed-data-parallel-ddp-batch-size)
                                num_workers=args.slurm_cpus_per_task, # number of subproc used for data loading
                                pin_memory=True,
                                drop_last=True
                                )
        loaders[d] = loader
    print(f"Data loaded")
    
    
    # ==== Load model ====
    # args.gpu is the local rank for this process
    model = Loader.load_model(MODEL,FEATURE, EDGEW, DROPOUT).cuda(args.gpu)
    cp_saver = CheckpointSaver(model=model, save_path=f'{MODEL_SAVE_DIR}/{MODEL_KEY}.model',
                            train_all=False,
                            patience=50, min_delta=0.2,
                            dist_rank=args.rank)
    # load ckpnt
    if os.path.exists(cp_saver.save_path + '_tmp') and args.rank == 0:
        print('# Model already trained, loading checkpoint')
        model.safe_load_state_dict(torch.load(cp_saver.save_path + '_tmp', 
                                map_location=torch.device(f'cuda:{args.gpu}')))
        
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    torch.distributed.barrier() # Sync params across GPUs before training
    
    
    # ==== train ====
    print("starting training:")
    logs = train(model=model, train_loader=loaders['train'], val_loader=loaders['val'], 
          device=args.gpu, saver=cp_saver, epochs=EPOCHS, lr_0=LEARNING_RATE)
    torch.distributed.barrier() # Sync params across GPUs
    
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
                    logs=logs
                    )