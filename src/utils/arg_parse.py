import argparse, random
from src.utils import config as cfg

def add_model_args(parser: argparse.ArgumentParser):
    """
    Adds the following arguments to the parser:
        - model_opt
        - data_opt
        - feature_opt
        - edge_opt
        - train
        - debug
    """
    # Define the options for data_opt and FEATURE_opt
        
    # Add the argument for model_opt
    parser.add_argument('-m',
        '--model_opt',
        choices=cfg.MODEL_OPT, nargs='+', required=True,
        help=f'Select one or more from {cfg.MODEL_OPT} where I = "improved" and '+ \
            'A = "all features". For example: DG is DGraphDTA ' + \
            'DGI is DGraphDTAImproved, ED is EsmDTA with esm_only set to true, '+ \
            'and EDA is the same but with esm_only set to False. Additional options:' + \
            '\n\t- EAT: EsmAttentionDTA (no graph for protein rep)' 
    )
    # Add the argument for EDGE_opt
    parser.add_argument('-e',
        '--edge_opt',
        choices=cfg.EDGE_OPT,
        nargs='+', default=cfg.EDGE_OPT[0:1], required=False,
        help=f'Select one or more from {cfg.EDGE_OPT}. "simple" is just taking ' + \
            'the normalized values from the protein cmap, "binary" means no edge weights'
    )
    parser.add_argument('-t',
        '--train',
        action='store_true',
        help='Forces training, if already trained it will load up the state dict'
    )
    parser.add_argument('-D',
        '--debug',
        action='store_true',
        help='Enters debug mode, no training is done, just model initialization.'
    )
    return parser

def add_hyperparam_args(parser: argparse.ArgumentParser):
    """
    Adds the following hyperparameter arguments to the parser:
        - batch_size
        - learning_rate
        - dropout
        - num_epochs
        
    """
    # hyperparameter options
    parser.add_argument('-bs',
        '--batch_size',
        action='store', type=int, default=64,
        help='Batch size for training (default: 64)'
    )
    parser.add_argument('-lr',
        '--learning_rate',
        action='store', type=float, default=1e-4,
        help='Learning rate for training (default: 0.0001)'
    )
    parser.add_argument('-do',
        '--dropout',
        action='store', type=float, default=0.4,
        help='Dropout rate for training (default: 0.4)'
    )
    parser.add_argument('-ne',
        '--num_epochs',
        action='store', type=int, default=2000,
        help='Number of epochs for training (default: 2000)'
    )
    return parser

def add_dataset_args(parser: argparse.ArgumentParser):
    """
    Adds the following dataset arguments to the parser:
        - data_opt
        - feature_opt
        - train_split
        - val_split
        - shuffle_data
        - rand_seed
    """

    # Add the argument for data_opt
    parser.add_argument('-d',
        '--data_opt',
        choices=cfg.DATA_OPT, nargs='+',   required=True,
        help=f'Select one of {cfg.DATA_OPT} (default: {cfg.DATA_OPT[0]}).'
    )
    # Add the argument for FEATURE_opt
    parser.add_argument('-f',
        '--feature_opt',
        choices=cfg.PRO_FEAT_OPT, nargs='+', required=True,
        help=f'Select one or more from {cfg.PRO_FEAT_OPT}.'
    )
    parser.add_argument('-ts',
        '--train_split',
        action='store', type=float, default=0.8,
        help='Percentage of data for training (default: 0.8)'
    )
    parser.add_argument('-vs',
        '--val_split',
        action='store', type=float, default=0.1,
        help='Percentage of data for validation (default: 0.1)'
    )
    parser.add_argument('-pro_o',
        '--protein_overlap', action='store_true',
        help='Disable split by protein, allowing protein overlap in train and test.')
    parser.add_argument('-nos',
        '--no_shuffle', action='store_true',
        help='Dont shuffle the data before splitting (default: True)'
    )
    parser.add_argument('-rs',
        '--rand_seed',
        action='store', type=int, default=0,
        help='Random seed for shuffling (default: 0)'
    )
    return parser

def add_slurm_dist_args(parser: argparse.ArgumentParser):
    parser.add_argument('-p',
        '--port',
        action='store', type=int, default=random.randint(49152,65535),
        help='Port for DDP (default: random int)'
    )
    parser.add_argument('-odir',
        '--output_dir',
        action='store', type=str, default="./slurm_tests/DDP/%j",
        help='Output dir for DDP (default: ./slurm_tests/DDP/%j)'
    )
    parser.add_argument('-s_ng',
        '--slurm_ngpus',
        action='store', type=int, default=1,
        help='Number of GPUs for DDP (default: 1)'
    )
    parser.add_argument('-s_nn',
        '--slurm_nnodes',
        action='store', type=int, default=1,
        help='Number of nodes for DDP (default: 1)'
    )
    parser.add_argument('-s_nl',
        '--slurm_nodelist',
        action='store', type=str, default=None,
        help='Node list for DDP (default: None)'
    )
    parser.add_argument('-s_t',
        '--slurm_time',
        action='store', type=int, default=60,
        help='Time for DDP (default: 60)'
    )
    parser.add_argument('-s_m',
        '--slurm_mem',
        action='store', type=str, default='15GB',
        help='Memory required for DDP (default: 15GB)'
    )
    parser.add_argument('-s_cp',
        '--slurm_cpus_per_task',
        action='store', type=int, default=2,
        help='CPUs per task for DDP (default: 2)'
    )
    return parser

def safe_parse(parser: argparse.ArgumentParser, 
               jyp_args:str='-m EAT -d davis -f nomsa -e simple -D'):
    """Safe argument parsing for jupyter notebooks"""
    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            # Jupyter notebook
            in_args = jyp_args.split()
            args = parser.parse_args(args=in_args)
        else:  
            args = parser.parse_args()
    except NameError:
        # Python script
        args = parser.parse_args()
    return args

def parse_train_test_args(verbose=True, distributed=False, 
                          jyp_args='-m EAT -d davis -f nomsa -e simple -D'):
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Arguments for train test.")
    add_model_args(parser)
    add_dataset_args(parser)
    add_hyperparam_args(parser)
    if distributed: 
        add_slurm_dist_args(parser)
    args = safe_parse(parser, jyp_args=jyp_args)

    # Model training args
    
    if verbose:
        # Now you can use the selected options in your code as needed
        if args.debug: print(f'|============|! DEBUG MODE !|============|\n')
        global_bs = args.batch_size
        if distributed:
            global_bs *= args.slurm_nnodes * args.slurm_ngpus
        print(f"---------------- DATA OPT ----------------")
        print(f"             data_opt: {args.data_opt}")
        print(f"      protein_overlap: {args.protein_overlap}\n")
        print(f"---------------- MODEL OPT ---------------")
        print(f"   Selected model_opt: {args.model_opt}")
        print(f"    Selected data_opt: {args.data_opt}")
        print(f" Selected feature_opt: {args.feature_opt}")
        print(f"    Selected edge_opt: {args.edge_opt}")
        print(f"      forced training: {args.train}\n")

        print(f"-------------- HYPERPARAMETERS -----------")
        print(f"   Global Batch size: {global_bs}")
        print(f"       Learning rate: {args.learning_rate}")
        print(f"             Dropout: {args.dropout}")
        print(f"          Num epochs: {args.num_epochs}\n")
    return args