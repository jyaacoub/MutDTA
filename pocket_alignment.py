"""
A collection of functions that generate a graph
mask from a binding pocket sequence.
"""

from Bio import Align
from Bio.Align import substitution_matrices
import torch
from torch_geometric.data import Data


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
    masked_data : torch_geometric.data.Data
        A data object with the same attributes as data, but representing only
        the binding pocket part of the sequence in the graph
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
    
    masked_data = Data(
        x=nodes,
        edge_index=edges,
        pro_seq=data.pro_seq,
        prot_id=data.prot_id
    )
    return masked_data


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
