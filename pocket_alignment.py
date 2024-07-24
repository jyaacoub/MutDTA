"""
A collection of functions that generate a graph
mask from a binding pocket sequence.
"""

import json
import os

from Bio import Align
from Bio.Align import substitution_matrices
import pandas as pd
import torch

from src.data_prep.downloaders import Downloader


def create_pocket_mask(target_seq: str, query_seq: str) -> list[bool]:
    """
    Return an index mask of a pocket on a protein sequence.
    
    Parameters
    ----------
    target_seq : str
        The protein sequence you want to query in
    query_seq : str
        The binding pocket sequence for the protein

    Returns
    -------
    index_mask : list[bool]
        A boolean list of indices that are True if the residue at that
        position is part of the binding pocket and false otherwise
    """
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
        -pocket_mask_x : torch.Tensor
            The nodes of only the pocket of the protein sequence of dimension
            [pocket_sequence_length, num_features]
        -pocket_mask_edge_index : torch.Tensor
            The edge connections in COO format only relating to 
            the pocket nodes of the protein sequence of dimension [2, num_pocket_edges]
    """
    nodes = data.x[mask]
    edges = data.edge_index
    edge_mask = []
    for i in range(edges.shape[1]):
        # Throw out edges that are connected to at least one node not in the
        # binding pocket
        node_1, node_2 = edges[:,i][0], edges[:,i][1]
        edge_mask.append(True) if mask[node_1] and mask[node_2] else edge_mask.append(False)  
    edges = torch.transpose(torch.transpose(edges, 0, 1)[edge_mask], 0, 1)
    
    data.pocket_mask = mask
    data.pocket_mask_x = nodes
    data.pocket_mask_edge_index = edges
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
        dataset_path: str = 'data/DavisKibaDataset/kiba/',
        pockets_path: str = 'data/DavisKibaDataset/kiba_pocket'
    ) -> None:
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
    """
    csv_path = os.path.join(dataset_path, 'nomsa_binary_original_binary', 'full', 'cleaned_XY.csv')
    df = pd.read_csv(csv_path, usecols=['prot_id'])
    prot_ids = list(set(df['prot_id']))
    dl = Downloader()
    seq_save_dir = os.path.join(pockets_path, 'pockets')
    os.makedirs(seq_save_dir, exist_ok=True)
    download_check = dl.download_pocket_seq(prot_ids, seq_save_dir)
    print(download_check)


if __name__ == '__main__':
    graph_data = torch.load('sample_pro_data.torch')
    seq = graph_data.pro_seq
    seq = seq[:857] + 'R' + seq[858:]
    graph_data.pro_seq = seq
    torch.save(graph_data, 'sample_pro_data_unmutated.torch')
    binding_pocket_sequence = 'KVLGSGAFGTVYKVAIKELEILDEAYVMASVDPHVCRLLGIQLITQLMPFGCLLDYVREYLEDRRLVHRDLAARNVLVITDFGLA'
    mask = create_pocket_mask(
        graph_data.pro_seq,
        binding_pocket_sequence
    )
    masked_data = mask_graph(graph_data, mask)
    # print(masked_data)
    # dl = Downloader()
    # seqs = Downloader.download_pocket_seq(['O00141', 'O14920', 'O15111'])
    # print(seqs)
    # get_dataset_binding_pockets()
    _parse_json('data/DavisKibaDataset/kiba_pocket/pockets/O00141.json')
