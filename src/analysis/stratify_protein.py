# To resolve https://github.com/jyaacoub/MutDTA/issues/57
# list of protein names for davis is in ../data/davis/proteins.txt 
# or downloaded from https://staff.cs.utu.fi/~aatapa/data/DrugTarget/target_gene_names.txt

from typing import Iterable
from src.utils import config as cfg
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def kinbase_to_df(fasta_fp:str=f'{cfg.DATA_ROOT}/misc/Human_kinase_domain.fasta'):
    """
    Converts the KinBase fasta file containing all Human Kinase Domains into a 
    dataframe for easy parsing.
    
    The Human_kinase_domain can be retrieved from:
    https://web.archive.org/web/20230517032418/http://kinase.com/kinbase/FastaFiles/Human_kinase_domain.fasta
        Sample of the FASTA headers:
            >TTBK2_Hsap (CK1/TTBK) *
            >TTBK1_Hsap (CK1/TTBK)
            >TSSK4_Hsap (CAMK/TSSK)
            >TSSK3_Hsap (CAMK/TSSK)
        - The first couple characters before the underscore are the protein names (*e.g.: TTBK2).
        - The characters after the underscore are the species (*e.g.: Hsap == homo sapiens).
        - Most importantly, the characters between the parenthesis are the protein family and 
          subgroups in that order (*e.g.: CK1/TTBK).

    Parameters
    ----------
    `fasta_fp` : str, optional
        The path to the downloaded fasta file path, by default f'{cfg.DATA_ROOT}/misc/Human_kinase_domain.fasta'
    """
    prots = {}
    with open(fasta_fp, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line[0] == '>': # header
                seq = lines[i+1].strip()
                name = re.search(r'^>(.+?)_Hsap', line).group(1)
                # all in the fasta has a protein family discriptor with at least 2 elements
                protein_family = re.search(r'\((.*)\)', line).group(1)
                main_family, subgroup = protein_family.split('/')[:2]
                
                prots[name] = (main_family, subgroup, seq)
                
                i+=2
            else:
                i+=1
    # convert to dataframe
    df = pd.DataFrame.from_dict(prots, orient='index', columns=['main_family', 'subgroup', 'seq'])
    df.index.name = 'protein_name'
    return df

def map_davis_to_kinbase(davis_prots:Iterable, df:pd.DataFrame) -> list:
    """
    Maps davis protein names to kinbase fasta file containing human kinase domains. 
    Returns list of proteins in davis that were found in the fasta.
    None if not found.
    
    NOTE: that for some in davis they have mutation information in brackets after 
    the protein name, but we only need the protein name for this function.
    Example:
        ABL1(F317I)p -> ABL1
    
    There are also some with "-alpha?" (or "beta" etc..) where ? is a number, we 
    can't ignore these and must include them by appending to the protein name "a?".
        
    Parameters
    ----------
    `davis_prots` : Iterable
        List of davis protein names from the davis dataset.
    `df` : pd.DataFrame
        The dataframe containing the human kinase domains (see kinbase_to_df()).
    """
    
    df = kinbase_to_df() if df is None else df
    
    greek = {'alpha', 'beta', 'gamma', 'delta', 'epsilon'} # for checking if protein name has greek letter
    
    found_prots = {}
    for k in davis_prots:
        name = k.split('(')[0]
        alpha = ''
        if '-' in name:
            name, alpha = name.split('-')
        
        # removing any 'p' at the end of the name which indicates phosphorylation
        if name[-1] == 'p':
            name = name[:-1]
        
        # getting alpha info if it exists
        if len(alpha) >= 1:
            alpha_name, alpha_num = re.search(r'([a-z]*)(\d*)', alpha).groups()
            if alpha_name in greek:
                name += f'{alpha_name[0]}{alpha_num}'
            
        # checking if name is in the dataframe ignoring case
        matches = df.index[df.index.str.lower() == name.lower()]
        if len(matches) > 0:
            name = matches[0] # with proper case
            found_prots[k] = (name, df.loc[name, 'main_family'], df.loc[name, 'subgroup'])
        else:
            # check if name is from a subgroup
            matches = df.index[df.subgroup.str.lower() == name.lower()]
            if len(matches) > 0:
                name = matches[0]
                found_prots[k] = (name, df.loc[name, 'main_family'], df.loc[name, 'subgroup'])
            else:
                # check if name is from a main family
                matches = df.index[df.main_family.str.lower() == name.lower()]
                if len(matches) > 0:
                    name = matches[0]
                    found_prots[k] = (name, df.loc[name, 'main_family'], df.loc[name, 'subgroup'])
                else:
                    found_prots[k] = None
                    print(f'MISSING: {k}-{name}')
            
    return found_prots

def plot_breakdown(models_to_plot:list=['DG', 'EDI'], 
                   subgroups_to_plot:list=['TK', 'STE', 'Other', 'CAMK', 'AGC']):
    
    fig, axes = plt.subplots(len(subgroups_to_plot)+1, len(models_to_plot), 
                        figsize=(5*len(models_to_plot), 4*(len(subgroups_to_plot)+1)))
    for i, model_type in enumerate(models_to_plot):
        if model_type == 'EDI':
            model_path = lambda x: f'results/model_media/test_set_pred/EDIM_davis{x}D_nomsaF_binaryE_48B_0.0001LR_0.4D_2000E_testPred.csv'
        elif model_type == 'DG':
            model_path = lambda x: f'results/model_media/test_set_pred/DGM_davis{x}D_nomsaF_binaryE_64B_0.0001LR_0.4D_2000E_testPred.csv'


        # Do the same but this time with error bars by using cross validation
        # data will be a dict of {main_family: [mse1, mse2, ...], ...}
        data_main = {}
        data_subgroups = {} # {main_family: {subgroup: [mse1, mse2, ...], ...}, ...}

        for fold in range(5):
            pred = pd.read_csv(model_path(fold), index_col='name')
            
            # returns a dict of {davis_name: (kinbase_name, main_family, subgroup)}
            pred_kb = map_davis_to_kinbase(pred.index.unique(), df=kin_df) # should be the same for all folds (same test set)
            
            # update pred to have kinbase info
            pred['kinbase_name'] = pred.index.map(lambda x: pred_kb[x][0])
            pred['main_family'] = pred.index.map(lambda x: pred_kb[x][1])
            pred['subgroup'] = pred.index.map(lambda x: pred_kb[x][2])
            
            for f in pred.main_family.unique():
                matched = pred[pred.main_family == f]
                mse = ((matched.pred - matched.actual)**2).mean()
                
                # add main family mse to dict
                data_main[f] = data_main.get(f, []) + [mse]
                
                # add main_family subgroup mse to dict
                data_subgroups[f] = data_subgroups.get(f, {})
                
                for g in matched.subgroup.unique():
                    g_matched = matched[matched.subgroup == g]
                    mse = ((g_matched.pred - g_matched.actual)**2).mean()
                    data_subgroups[f][g] = data_subgroups[f].get(g, []) + [mse]
            

        # plot mse as bar chart
        plot_df = pd.DataFrame(data_main)
        curr_ax = axes[0, i]
        sns.barplot(data=plot_df, ax=curr_ax)
        curr_ax.set_ylabel(f'MSE')
        curr_ax.set_xlabel('Main Family')
        curr_ax.set_title(f'MSE loss for {model_type}M by Protein Family')
        curr_ax.set_ylim(0, 1.4)
        
        
        for j, f in enumerate(subgroups_to_plot):
            curr_ax = axes[j+1, i]
            
            sns.barplot(data=pd.DataFrame(data_subgroups[f]), ax=curr_ax)
            if i == 0:
                curr_ax.set_ylabel('MSE')
            curr_ax.set_xlabel(f'{f} Subgroups')
            curr_ax.set_ylim(0, 1.4)
        
    plt.tight_layout()

            
if __name__ == '__main__':
    import json
    from src.analysis.stratify_protein import map_davis_to_kinbase, kinbase_to_df

    prot_dict = json.load(open('/home/jyaacoub/projects/data/davis/proteins.txt', 'r'))
    
    df = kinbase_to_df()
    prots = map_davis_to_kinbase(prot_dict, df)
    
    # figure to show breakdown of proteins in davis dataset and their performance with EDI and DG
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from src.analysis.stratify_protein import map_davis_to_kinbase

    # Get kinbase data to map to family names
    kin_df = pd.read_csv('../data/misc/kinase_base_updated.csv', index_col='name')


    #%%
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    for i, model_type in enumerate(['EDI', 'DG']):
        if model_type == 'EDI':
            model_path = lambda x: f'results/model_media/test_set_pred/EDIM_davis{x}D_nomsaF_binaryE_48B_0.0001LR_0.4D_2000E_testPred.csv'
        elif model_type == 'DG':
            model_path = lambda x: f'results/model_media/test_set_pred/DGM_davis{x}D_nomsaF_binaryE_64B_0.0001LR_0.4D_2000E_testPred.csv'


        # Do the same but this time with error bars by using cross validation
        # data will be a dict of {main_family: [mse1, mse2, ...], ...}
        data = {}

        for fold in [0,1,2,3,4]:
            pred = pd.read_csv(model_path(fold), index_col='name')
            
            # returns a dict of {davis_name: (kinbase_name, main_family, subgroup)}
            pred_kb = map_davis_to_kinbase(pred.index.unique(), df=kin_df) # should be the same for all folds (same test set)
            
            # update pred to have kinbase info
            pred['kinbase_name'] = pred.index.map(lambda x: pred_kb[x][0])
            pred['main_family'] = pred.index.map(lambda x: pred_kb[x][1])
            pred['subgroup'] = pred.index.map(lambda x: pred_kb[x][2])
            
            for f in pred.main_family.unique():
                matched = pred[pred.main_family == f]
                mse = ((matched.pred - matched.actual)**2).mean()
                
                # get subfamily
                data[f] = data.get(f, []) + [mse]
            

        # plot mse as bar chart
        plot_df = pd.DataFrame(data)
        curr_ax = ax[i, 0]
        sns.barplot(data=plot_df, ax=curr_ax)
        curr_ax.set_ylabel(f'MSE ({model_type})')
        
        if i == 0:
            curr_ax.set_title('MSE by Main Family')
        else:
            curr_ax.set_xlabel('Main Family')
        


        # plot breakdown for specific main family and its subgroups
        f = 'TK'
        data = {} # {subgroup: [mse1, mse2, ...]}

        for fold in [0,1,2,3,4]:
            pred = pd.read_csv(model_path(fold), index_col='name')
            
            # returns a dict of {davis_name: (kinbase_name, main_family, subgroup)}
            pred_kb = map_davis_to_kinbase(pred.index.unique(), df=kin_df) # should be the same for all folds (same test set)

            # update pred to have kinbase info
            pred['kinbase_name'] = pred.index.map(lambda x: pred_kb[x][0])
            pred['main_family'] = pred.index.map(lambda x: pred_kb[x][1])
            pred['subgroup'] = pred.index.map(lambda x: pred_kb[x][2])
            
            matched = pred[pred.main_family == f]
            for g in matched.subgroup.unique():
                g_matched = matched[matched.subgroup == g]
                mse = ((g_matched.pred - g_matched.actual)**2).mean()
                data[g] = data.get(g, []) + [mse]

        curr_ax = ax[i, 1]
        sns.barplot(data=pd.DataFrame(data), ax=curr_ax)
        # curr_ax.set_ylabel('MSE')
        if i == 0:
            curr_ax.set_title(f'MSE by Subgroup in {f} Main Family')
        else:
            curr_ax.set_xlabel('Subgroup')
