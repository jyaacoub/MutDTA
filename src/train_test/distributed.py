import time, os

import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from src.utils.loader import Loader
from src.data_analysis.metrics import get_metrics

from src.train_test.training import train, test
from src.train_test.utils import CheckpointSaver, init_node, init_dist_gpu, print_device_info

from src.utils import config as cfg

# distributed training fn
def dtrain(args):
    # ==== initialize the node ====
    init_node(args)
    
    
    # ==== Set up distributed training environment ====
    init_dist_gpu(args)
    
    # TODO: update this to loop through all options.
    # only support for a single option for now:
    MODEL = args.model_opt[0]
    DATA = args.data_opt[0]
    FEATURE = args.feature_opt[0]
    EDGEW = args.edge_opt[0]
    ligand_feature = args.ligand_feature_opt[0]
    ligand_edge = args.ligand_edge_opt[0]
    
    media_save_p = f'{cfg.MEDIA_SAVE_DIR}/{DATA}/'
    MODEL_KEY = Loader.get_model_key(model=MODEL,data=DATA,pro_feature=FEATURE,edge=EDGEW,
                                     ligand_feature=ligand_feature, ligand_edge=ligand_edge,
                                     batch_size=args.batch_size*args.world_size,
                                     lr=args.learning_rate,dropout=args.dropout,
                                     n_epochs=args.num_epochs,
                                     pro_overlap=args.protein_overlap)
    
    print(os.getcwd())
    print(MODEL_KEY)
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
    print(f"         Local Batch size: {args.batch_size}")
    print(f"        Global Batch size: {args.batch_size*args.world_size}")
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
        bs = 1 if d == 'test' else args.batch_size
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
    model = Loader.init_model(MODEL, FEATURE, EDGEW, args.dropout).cuda(args.gpu)
    cp_saver = CheckpointSaver(model=model, save_path=f'{cfg.MODEL_SAVE_DIR}/{MODEL_KEY}.model',
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
          device=args.gpu, saver=cp_saver, epochs=args.num_epochs, lr_0=args.learning_rate)
    torch.distributed.barrier() # Sync params across GPUs
    
    cp_saver.save()
    
    
    # ==== Evaluate ====
    loss, pred, actual = test(model, loaders['test'], args.gpu)
    torch.distributed.barrier() # Sync params across GPUs
    if args.rank == 0:
        print("Test loss:", loss)
        get_metrics(actual, pred,
                    save_figs=False,
                    save_path=media_save_p,
                    model_key=MODEL_KEY,
                    csv_file=cfg.MODEL_STATS_CSV,
                    show=False,
                    logs=logs
                    )
        
    # validation
    loss, pred, actual = test(model, loaders['val'], args.gpu)
    torch.distributed.barrier() # Sync params across GPUs
    if args.rank == 0:
        print(f'# Val loss: {loss}')
        get_metrics(actual, pred,
                    save_figs=False,
                    save_path=media_save_p,
                    model_key=MODEL_KEY,
                    csv_file=cfg.MODEL_STATS_CSV_VAL,
                    show=False,
                    )