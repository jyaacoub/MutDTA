"""
A collection of functions that generate a graph
mask from a binding pocket sequence.
"""

import json
import os
import shutil

from Bio import Align
from Bio.Align import substitution_matrices
import numpy as np
import pandas as pd
import torch

from src.data_prep.downloaders import Downloader


def create_pocket_mask(target_seq: str, pocket_seq: str) -> list[bool]:
    """
    Return an index mask of a pocket on a protein sequence.
    
    Parameters
    ----------
    target_seq : str
        The protein sequence you want to query in
    pocket_seq : str
        The binding pocket sequence for the protein

    Returns
    -------
    index_mask : list[bool]
        A boolean list of indices that are True if the residue at that
        position is part of the binding pocket and false otherwise
    """
    # Ensure that no '-' characters are present in the query sequence
    query_seq = pocket_seq.replace('-', 'X')
    # Taken from tutorial https://biopython.org/docs/dev/Tutorial/chapter_pairwise.html
    aligner = Align.PairwiseAligner()
    # Pairwise alignment parameters as specified in paragraph 2
    # of Methods - Structure and sequence data in "Calibrated 
    # geometric deep learning improves kinase-drug binding predictions"
    # by Luo et al. (https://www.nature.com/articles/s42256-023-00751-0)
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(target_seq, query_seq)
    alignment = alignments[0]

    index_mask = [False] * len(target_seq)
    for index_range in alignment.aligned[0]:
        start, end = index_range[0], index_range[1]
        for i in range(start, end):
            index_mask[i] = True
    return index_mask


def mask_graph(data, mask: list[bool]):
    """
    Apply a binding pocket mask to a torch_geometric graph.
    Remove nodes that aren't in the binding pocket and remove
    edges corresponding to these removed nodes.
    
    Parameters
    ----------
    data : torch_geometric.data.Data
        -x: node feature matrix with shape [num_residues, num_features]
        -edge_index: pairs of indices that share an edge with shape [2, num_total_edges]
        -pro_seq: full target protein sequence
        -prot_id: protein ID with mutations if applicable
    mask: list[bool]
        A boolean list of indices that are True if the residue at that
        position is part of the binding pocket and false otherwise

    Return
    ------
    data : torch_geometric.data.Data
        The same data object that is in the parameters, with the following
        additional attributes:
        -pocket_mask : list[bool]
            The mask specified by the mask parameter of dimension [full_seuqence_length]
        -x : torch.Tensor
            The nodes of only the pocket of the protein sequence of dimension
            [pocket_sequence_length, num_features]
        -edge_index : torch.Tensor
            The edge connections in COO format only relating to 
            the pocket nodes of the protein sequence of dimension [2, num_pocket_edges]
    """
    # node map for updating edge indicies after mask
    node_map = np.cumsum(mask) - 1
    
    nodes = data.x[mask]
    edges = []
    edge_mask = []
    for i in range(data.edge_index.shape[1]):
        # Throw out edges that are not part of connecting two nodes in the pocket...
        node_1, node_2 = data.edge_index[:,i][0], data.edge_index[:,i][1]
        if mask[node_1] and mask[node_2]:
            # append mapped index:
            edges.append([node_map[node_1], node_map[node_2]])
            edge_mask.append(True)  
        else:
            edge_mask.append(False)
    
    data.x = nodes
    data.pocket_mask = mask
    data.edge_index = torch.tensor(edges).T # reshape to (2, E)
    if 'edge_weight' in  data:
        data.edge_weight = data.edge_weight[edge_mask]
    return data


def _parse_json(json_path: str) -> str:
    """
    Parse a JSON file that holds binding pocket data downloaded from KLIFS.

    Parameters
    ----------
    json_path : str 
        The path to the JSON file

    Returns
    -------
    str
        The binding pocket sequence
    """
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        return data[0]['pocket']


def get_dataset_binding_pockets(
        dataset_path: str = 'data/DavisKibaDataset/kiba/nomsa_binary_original_binary/full',
        pockets_path: str = 'data/DavisKibaDataset/kiba_pocket',
        skip_download: bool = False,
    ) -> tuple[dict[str, str], set[str]]:
    """
    Get all binding pocket sequences for a dataset

    Parameters
    ----------
    dataset_path : str
        The path to the directory containing the dataset (as of July 24, 2024,
        only expecting Kiba dataset). Specify only the path to one of 'davis', 'kiba',
        or 'PDBbind' (e.g., 'data/DavisKibaDataset/kiba')
    pockets_path: str
        The path to the new dataset directory after all the binding pockets have been found
    
    Returns
    -------
    tuple[dict[str, str], set[str]]
        A tuple consisting of:
            -A map of protein ID, binding pocket sequence pairs
            -A set of protein IDs with no KLIFS binding pockets
    """
    csv_path = os.path.join(dataset_path, 'cleaned_XY.csv')
    df = pd.read_csv(csv_path, usecols=['prot_id'])
    prot_ids = list(set(df['prot_id']))
    # Strip out mutations and '-(alpha, beta, gamma)' tags if they are present,
    # the binding pocket sequence will be the same for mutated and non-mutated genes
    prot_ids = [id.split('(')[0].split('-')[0] for id in prot_ids]
    seq_save_dir = os.path.join(pockets_path, 'pockets')
    
    if not skip_download: # to use cached downloads only! (useful when on compute node)
        dl = Downloader()
        os.makedirs(seq_save_dir, exist_ok=True)
        dl.download_pocket_seq(prot_ids, seq_save_dir)
            
    download_errors = set()
    sequences = {}
    for file in os.listdir(seq_save_dir):
        pocket_seq = _parse_json(os.path.join(seq_save_dir, file))
        if pocket_seq == 0 or len(pocket_seq) == 0:
            download_errors.add(file.split('.')[0])
        else:
            sequences[file.split('.')[0]] = pocket_seq
    
    # adding any remainder prots not downloaded.
    for p in prot_ids:
        if p not in sequences:
            download_errors.add(p)
    
    return (sequences, download_errors)


def create_binding_pocket_dataset(
    dataset_path: str,
    pocket_sequences: dict[str, str],
    download_errors: set[str],
    new_dataset_path: str
) -> None:
    """
    Apply the graph mask based on binding pocket sequence for each
    Data object in a PyTorch dataset.

    dataset_path : str
        The path to the PyTorch dataset object to be transformed
    pocket_sequences : dict[str, str]
        A map of protein ID, binding pocket sequence pairs
    download_errors : set[str]
        A set of protein IDs that have no binding pocket sequence
        to be downloaded from KLIFS
    new_dataset_path : str
        A path to where the new dataset should be saved
    """
    dataset = torch.load(dataset_path)
    new_dataset = {}
    for id, data in dataset.items():
        # If there are any mutations or (-alpha,beta,gamma) tags, strip them
        stripped_id = id.split('(')[0].split('-')[0]
        if stripped_id not in download_errors:
            mask = create_pocket_mask(data.pro_seq, pocket_sequences[stripped_id])
            new_data = mask_graph(data, mask)
            new_dataset[id] = new_data
    os.makedirs(os.path.dirname(new_dataset_path), exist_ok=True)
    torch.save(new_dataset, new_dataset_path)


def binding_pocket_filter(dataset_csv_path: str, download_errors: set[str], csv_save_path: str):
    """
    Filter out protein IDs that do not have a corresponding KLIFS
    binding pocket sequence from the dataset.

    Parameters
    ----------
    dataset_csv_path : str
        The path to the original cleaned CSV. Will probably be a CSV named cleaned_XY.csv
        or something like that.
    download_errors : set[str]
        A set of protein IDs with no KLIFS binding pocket sequences.
    csv_save_path : str
        The path to save the new CSV file to.
    """
    df = pd.read_csv(dataset_csv_path, index_col=0)
    df = df[~df.prot_id.str.split('(').str[0].str.split('-').str[0].isin(download_errors)]
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    df.to_csv(csv_save_path)


def pocket_dataset_full(
    dataset_dir: str,
    pocket_dir: str,
    save_dir: str,
    skip_download: bool = False
) -> None:
    """
    Create all elements of a dataset that includes binding pockets. This
    function assumes the PyTorch object holding the dataset is named 'data_pro.pt'
    and the CSV holding the cleaned data is named 'cleaned_XY.csv'.

    Parameters
    ----------
    dataset_dir : str
        The path to the dataset to be transformed
    pocket_dir : str
        The path to where the dataset raw pocket sequences are to be saved
    save_dir : str
        The path to where the new dataset is to be saved
    """
    pocket_map, download_errors = get_dataset_binding_pockets(dataset_dir, pocket_dir, skip_download)
    print(f'Binding pocket sequences were not found for the following {len(download_errors)} protein IDs:')
    print(','.join(list(download_errors)))
    create_binding_pocket_dataset(
        os.path.join(dataset_dir, 'data_pro.pt'),
        pocket_map,
        download_errors,
        os.path.join(save_dir, 'data_pro.pt')
    )
    binding_pocket_filter(
        os.path.join(dataset_dir, 'cleaned_XY.csv'),
        download_errors,
        os.path.join(save_dir, 'cleaned_XY.csv')
    )
    if dataset_dir != save_dir:
        shutil.copy2(os.path.join(dataset_dir, 'data_mol.pt'), os.path.join(save_dir, 'data_mol.pt'))
        shutil.copy2(os.path.join(dataset_dir, 'XY.csv'), os.path.join(save_dir, 'XY.csv'))


if __name__ == '__main__':
    pocket_dataset_full(
        'data/DavisKibaDataset/kiba/nomsa_binary_original_binary/full/',
        'data/DavisKibaDataset/kiba_pocket',
        'data/DavisKibaDataset/kiba_pocket/nomsa_binary_original_binary/full/'
    )
