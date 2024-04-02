import argparse, random
from src.utils import config as cfg

def add_model_args(parser: argparse.ArgumentParser):
    """
    Adds the following arguments to the parser:
        - model_opt
        - feature_opt
        - edge_opt
        - ligand_feature_opt
        - ligand_edge_opt
        - train
        - debug
    """
    # Define the options for data_opt and FEATURE_opt
        
    # Add the argument for model_opt
    parser.add_argument('-m',
        '--model_opt',
        choices=cfg.MODEL_OPT.list(), nargs='+', required=True,
        help=f'Model option where I = "improved" and '+ \
            'A = "all features". For example: DG is DGraphDTA ' + \
            'DGI is DGraphDTAImproved, ED is EsmDTA with esm_only set to true, '+ \
            'and EDA is the same but with esm_only set to False. Additional options:' + \
            '\n\t- EAT: EsmAttentionDTA (no graph for protein rep)' 
    )
    # Add the argument for FEATURE_opt
    parser.add_argument('-f',
        '--feature_opt',
        choices=cfg.PRO_FEAT_OPT.list(), nargs='+', required=True,
        help=f'Protein feture option'
    )
    # Add the argument for EDGE_opt
    parser.add_argument('-e',
        '--edge_opt',
        choices=cfg.PRO_EDGE_OPT.list(),
        nargs='+', default=['binary'], required=False,
        help=f'Protein edge option. "simple" is just taking ' + \
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
        help='Enters debug mode, no training is done, just model initialization and batch input.'
    )
    parser.add_argument('-nr',
        '--no_rename',
        action='store_true',
        help='Dont rename the model to remove "_tmp" postfix, used for test.py'
    )
    
    # Arguments for ligand options
    parser.add_argument('-lf',
        '--ligand_feature_opt',
        choices=cfg.LIG_FEAT_OPT.list(), 
        nargs='+', default=[cfg.LIG_FEAT_OPT.original], required=False,
        help=f'Ligand features option'
    )
    parser.add_argument('-le',
        '--ligand_edge_opt',
        choices=cfg.LIG_EDGE_OPT.list(),
        nargs='+', default=[cfg.LIG_EDGE_OPT.binary], required=False,
        help=f'Ligand edge option'
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
    parser.add_argument('-dop',
        '--dropout_prot',
        action='store', type=float, default=0.4,
        help='Dropout rate for protein GCN branch for training (default: 0.4)'
    )
    parser.add_argument('-embP',
        '--pro_emb_dim',
        action='store', type=int, default=128,
        help='Embedding dimension for protein GCN branch for training (default: 128)'
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
        - train_split
        - val_split
        - shuffle_data
        - rand_seed
        - fold_selection
    """

    # Add the argument for data_opt
    parser.add_argument('-d',
        '--data_opt',
        choices=cfg.DATA_OPT.list(), nargs='+',   required=True,
        help=f'Dataset option (default: {cfg.DATA_OPT[0]}).'
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
    parser.add_argument('-folds',
        '--fold_selection',
        action='store', type=int, default=0,
        help='Fold selection (default: 0 - first fold)'
    )
    
    # for test.py:
    parser.add_argument('-spte', # default is not to save predictions
        '--save_pred_test',
        action='store_true',
        help='Save predictions of model on test set to csv file.'
    )
    parser.add_argument('-sptr', 
        '--save_pred_train',
        action='store_true', # default is not to save predictions
        help='Save predictions of model on training set to csv file.'
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
        action='store', type=int, default=4,
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
            args, unknown_args = parser.parse_known_args(args=in_args)
        else:  
            args, unknown_args = parser.parse_known_args()
    except NameError:
        # Python script
        args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def process_unknown_args(unknown_args):
    """Converts a list of unknown arguments into a dictionary, handling both '--key value' and '--key=value' formats."""
    kwargs = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg[2:]
            # Check if the next item is a value (not another key)
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                # The next item is the value
                value = unknown_args[i + 1]
                i += 2  # Move past the value for the next iteration
            else:
                # Assume a flag-like argument that implies a boolean value
                value = True
                i += 1
            kwargs[key] = value
        else:
            # Move to the next item if the current one doesn't start with '--'
            i += 1
    return kwargs

def parse_train_test_args(verbose=True, distributed=False, 
                          jyp_args='-m EAT -d davis -f nomsa -e simple -D'):
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Arguments for train test.")
    add_model_args(parser)
    add_dataset_args(parser)
    add_hyperparam_args(parser)
    if distributed: 
        add_slurm_dist_args(parser)
    args, unknown_args = safe_parse(parser, jyp_args=jyp_args)
    unknown_args = process_unknown_args(unknown_args)

    # Model training args
    
    if verbose:
        # Now you can use the selected options in your code as needed
        if args.debug: print(f'|============|! DEBUG MODE !|============|\n')
        global_bs = args.batch_size
        if distributed:
            global_bs *= args.slurm_nnodes * args.slurm_ngpus
        print(f"---------------- DATA OPT ----------------")
        print(f"             data_opt: {args.data_opt}")
        print(f"      protein_overlap: {args.protein_overlap}")
        print(f"       fold_selection: {args.fold_selection}\n")
        print(f"---------------- MODEL OPT ---------------")
        print(f"   Selected model_opt: {args.model_opt}")
        print(f"    Selected data_opt: {args.data_opt}")
        print(f" Selected feature_opt: {args.feature_opt}")
        print(f"    Selected edge_opt: {args.edge_opt}")
        print(f"      forced training: {args.train}")
        print(f"                   -----")
        print(f"   ligand_feature_opt: {args.ligand_feature_opt}")
        print(f"      ligand_edge_opt: {args.ligand_edge_opt}\n")

        print(f"-------------- HYPERPARAMETERS -----------")
        print(f"   Global Batch size: {global_bs}")
        print(f"       Learning rate: {args.learning_rate}")
        print(f"             Dropout: {args.dropout}")
        print(f"          Num epochs: {args.num_epochs}\n")
    return args, unknown_args