import argparse

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
    model_opt_choices = ['DG', 'DGI', 'ED', 'EDA', 'EDI', 'EDAI', 'EAT']
    data_opt_choices = ['davis', 'kiba', 'PDBbind']
    feature_opt_choices = ['nomsa', 'msa', 'shannon']
    edge_opt_choices = ['simple', 'binary']
        
    # Add the argument for model_opt
    parser.add_argument('-m',
        '--model_opt',
        choices=model_opt_choices,
        nargs='+',  # Allows accepting multiple arguments for FEATURE_opt
        required=True,
        help=f'Select one or more from {model_opt_choices} where I = "improved" and '+ \
            'A = "all features". For example: DG is DGraphDTA ' + \
            'DGI is DGraphDTAImproved, ED is EsmDTA with esm_only set to true, '+ \
            'and EDA is the same but with esm_only set to False. Additional options:' + \
            '\n\t- EAT: EsmAttentionDTA (no graph for protein rep)' 
    )

    # Add the argument for data_opt
    parser.add_argument('-d',
        '--data_opt',
        choices=data_opt_choices,
        nargs='+',  # Allows accepting multiple arguments
        # default=data_opt_choices[0],
        required=True,
        help=f'Select one of {data_opt_choices} (default: {data_opt_choices[0]}).'
    )

    # Add the argument for FEATURE_opt
    parser.add_argument('-f',
        '--feature_opt',
        choices=feature_opt_choices,
        nargs='+',  # Allows accepting multiple arguments for FEATURE_opt
        required=True,
        help=f'Select one or more from {feature_opt_choices}.'
    )

    # Add the argument for EDGE_opt
    parser.add_argument('-e',
        '--edge_opt',
        choices=edge_opt_choices,
        nargs='+',  # Allows accepting multiple arguments for EDGE_opt
        default=edge_opt_choices[0:1],
        required=False,
        help=f'Select one or more from {edge_opt_choices}. "simple" is just taking ' + \
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
        action='store',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )

    parser.add_argument('-lr',
        '--learning_rate',
        action='store',
        type=float,
        default=1e-4,
        help='Learning rate for training (default: 0.0001)'
    )

    parser.add_argument('-do',
        '--dropout',
        action='store',
        type=float,
        default=0.4,
        help='Dropout rate for training (default: 0.4)'
    )

    parser.add_argument('-ne',
        '--num_epochs',
        action='store',
        type=int,
        default=2000,
        help='Number of epochs for training (default: 2000)'
    )
    
    return parser

def add_dataset_args(parser: argparse.ArgumentParser):
    """
    Adds the following dataset arguments to the parser:
        - train_split
        - val_split
        - shuffle_data
        - rand_seed
    """
    
    parser.add_argument('-ts',
        '--train_split',
        action='store',
        type=float,
        default=0.8,
        help='Percentage of data for training (default: 0.8)'
    )
    
    parser.add_argument('-vs',
        '--val_split',
        action='store',
        type=float,
        default=0.1,
        help='Percentage of data for validation (default: 0.1)'
    )
    
    parser.add_argument('-nos',
        '--no_shuffle',
        action='store_true',
        help='Dont shuffle the data before splitting (default: True)'
    )
    
    parser.add_argument('-rs',
        '--rand_seed',
        action='store',
        type=int,
        default=0,
        help='Random seed for shuffling (default: 0)'
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


def parse_train_test_args(verbose=True):
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Arguments for train test.")
    add_model_args(parser)
    add_hyperparam_args(parser)
    add_dataset_args(parser)
    args = safe_parse(parser, jyp_args='-m EAT -d davis -f nomsa -e simple -D')

    # Model training args:
    model_opt = args.model_opt
    data_opt = args.data_opt
    feature_opt = args.feature_opt
    edge_opt = args.edge_opt
    
    if verbose:
        # Now you can use the selected options in your code as needed
        if args.debug: print(f'|==========|! DEBUG MODE !|==============|\n')

        print(f"---------------- MODEL OPT ---------------")
        print(f"     Selected og_model_opt: {model_opt}")
        print(f"         Selected data_opt: {data_opt}")
        print(f" Selected feature_opt list: {feature_opt}")
        print(f"    Selected edge_opt list: {edge_opt}")
        print(f"           forced training: {args.train}\n")

        print(f"-------------- HYPERPARAMETERS -------------")
        print(f"               Batch size: {args.bs}")
        print(f"            Learning rate: {args.lr}")
        print(f"                  Dropout: {args.do}")
        print(f"               Num epochs: {args.ne}\n")
    return args