"""
A collection of pocket sequence alignment functions.
"""

from Bio import Align
from Bio.Align import substitution_matrices
import torch


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
    index_mask : list[int]
        A mask of indices of the binding pocket on the protein sequence
    """
    aligner = Align.PairwiseAligner()
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


if __name__ == '__main__':
    mask1 = create_pocket_mask(
        'MTAPWVALALLWGSLCAGSGRGEAETRECIYYNANWELERTNQSGLERCEGEQDKRLHCYASWRNSSGTIELVKKGCWLDDFNCYDRQECVATEENPQVYFCCCEGNFCNERFTHLPEAGGPEVTYEPPPTAPTLLTVLAYSLLPIGGLSLIVLLAFWMYRHRKPPYGHVDIHEDPGPPPPSPLVGLKPLQLLEIKARGRFGCVWKAQLMNDFVAVKIFPLQDKQSWQSEREIFSTPGMKHENLLQFIAAEKRGSNLEVELWLITAFHDKGSLTDYLKGNIITWNELCHVAETMSRGLSYLHEDVPWCRGEGHKPSIAHRDFKSKNVLLKSDLTAVLADFGLAVRFEPGKPPGDTHGQVGTRRYMAPEVLEGAINFQRDAFLRIDMYAMGLVLWELVSRCKAADGPVDEYMLPFEEEIGQHPSLEELQEVVVHKKMRPTIKDHWLKHPGLAQLCVTIEECWDHDAEARLSAGCVEERVSLIRRSVNGTTSDCLVSLVTSVTNVDLPPKESSI',
        'EIKARGRFGCVWKVAVKIFSWQSEREIFSTPGENLLQFIAAWLITAFHDKGSLTDYLKGEGHKPSIAHRDFKSKNVLLLADFGLA'
    )
    mask2 = create_pocket_mask(
        'GAACT', 'GAT'
    )
    print(mask2)

    graph_data = torch.load('sample_pro_data.torch')
    # print(graph_data)
