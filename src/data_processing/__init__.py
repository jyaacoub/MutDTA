from src.data_processing.general import Processor
from src.data_processing.pdbbind import PDBbindProcessor
from src.data_processing.download import Downloader


RES_CODE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'ASX': 'B', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q',
    
    'GLX': 'Z', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}