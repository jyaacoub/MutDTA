import numpy as np

# one hot encoding
def one_hot(x, allowable_set, cap=False):
    """Return the one-hot encoding of x as a numpy array."""
    if x not in allowable_set:
        if not cap:
            raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
        else:
            x = allowable_set[-1]
    return np.eye(len(allowable_set))[allowable_set.index(x)]


class ResInfo():
    """
    A class for managing residue constants and properties.

    The ResidueInfo class provides a convenient way to access and organize various constants 
    and properties related to amino acid residues.
    """
    pep_to_code = { # peptide codes
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'ASX': 'B', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q',
        
        'GLX': 'Z', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
        
        'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
        'TYR': 'Y', 'VAL': 'V'
    }
    code_to_pep = {v: k for k, v in pep_to_code.items()}
    
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 
                   'R', 'S', 'T', 'V', 'W', 'Y',
                    'X']

    aliphatic = ['A', 'I', 'L', 'M', 'V']
    aromatic = ['F', 'W', 'Y']
    polar_neutral = ['C', 'N', 'Q', 'S', 'T']
    acidic_charged = ['D', 'E']
    basic_charged = ['H', 'K', 'R']

    weight = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

    pka = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
            'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
            'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

    pkb = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
            'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
            'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

    pkx = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
            'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
            'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

    pl = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
            'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
            'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

    hydrophobic_ph2 = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                        'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                        'T': 13, 'V': 79, 'W': 84, 'Y': 49}

    hydrophobic_ph7 = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                        'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                        'T': 13, 'V': 76, 'W': 97, 'Y': 63}
    
    @staticmethod
    def normalize_add_x(dic):
        max_value = dic[max(dic, key=dic.get)]
        min_value = dic[min(dic, key=dic.get)]
        interval = float(max_value) - float(min_value)
        
        for key in dic.keys():
            dic[key] = (dic[key] - min_value) / interval
        
        # X represents all other amino acids
        # we set it to be the mid point of the interval #TODO: why is this done, instead of just doing standard normalizing where all sum to 1?
        dic['X'] = (max_value + min_value) / 2.0
        return dic
    
    @staticmethod
    def normalize_dict(dictionary): # TODO: why not this instead?
        """
        Normalize the values of a dictionary to sum up to 1.
        """
        total = sum(dictionary.values())
        return {key: value / total for key, value in dictionary.items()}

    weight = normalize_add_x(weight)
    pka = normalize_add_x(pka)
    pkb = normalize_add_x(pkb)
    pkx = normalize_add_x(pkx)
    pl = normalize_add_x(pl)
    hydrophobic_ph2 = normalize_add_x(hydrophobic_ph2)
    hydrophobic_ph7 = normalize_add_x(hydrophobic_ph7)
        