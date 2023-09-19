from matplotlib import pyplot as plt
import numpy as np
import os

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
        'TYR': 'Y', 'VAL': 'V',
        
        # Non-standard amino acids:
    }
    code_to_pep = {v: k for k, v in pep_to_code.items()}
    
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 
                   'I', 'K', 'L', 'M', 'N', 'P', 'Q', 
                   'R', 'S', 'T', 'V', 'W', 'Y', 'X']
    
    res_to_i = {k: i for i, k in enumerate(amino_acids)}

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


from collections import OrderedDict

class Chain:
    def __init__(self, pdb_file:str, model:int=1, t_chain=None):
        """
        This class was created to mimic the AtomGroup class from ProDy but optimized for fast parsing 
        only parses what is required for ANM simulations.

        Args:
            pdb_file (str): file path to pdb file to parse
            model (int, optional): model to parse from file. Defaults to 1.
            t_chain (_type_, optional): target chain to focus on. Defaults to None.
        """
        # parse chain -> {<chain>: {<residue_key>: {<atom_type>: np.array([x,y,z], "name": <res_name>)}}}
        self._chains = self._pdb_get_chains(pdb_file, model)
        self.pdb_file = pdb_file
        
        # if t_chain is not specified then set it to be the largest chain
        self.t_chain = t_chain or max(self._chains, key=lambda x: len(self._chains[x]))
        
        self.reset_attributes()
        
    def __len__(self):
        return len(self.getSequence())
    
    def __repr__(self):
        return f'<Chain {os.path.basename(self.pdb_file).split(".")[0]}:{self.t_chain}>'
        
    def reset_attributes(self):
        self._seq = None        
        self._coords = None
        self._hessian = None        
    
    @property
    def hessian(self) -> np.array:
        if self._hessian is None:
            self._hessian = self.buildHessian()
        return self._hessian
    
    @property
    def chain(self) -> OrderedDict:
        # {<chain>: {<residue_key>: {<atom_type>: np.array([x,y,z], "name": <res_name>)}}}
        # -> {<residue_key>: {<atom_type>: np.array([x,y,z], "name": <res_name>)}}
        return self._chains[self.t_chain]
    
    @property
    def t_chain(self) -> str:
        return self._t_chain
    
    @t_chain.setter
    def t_chain(self, chain_ID:str):
        assert len(chain_ID) == 1, f"Invalid chain ID {chain_ID}"
        # reset so that they are updated on next getter calls
        self.reset_attributes()
        self._t_chain = chain_ID        
    
    def getSequence(self) -> str:
        """
        camelCase to mimic ProDy Chain class

        Returns:
            str: The sequence of the chain
        """
        if self._seq is None:
            # Get sequence from chain
            seq = ''
            for res_v in self.chain.values():
                seq += ResInfo.pep_to_code[res_v["name"]]
            self._seq = seq
        return self._seq
        
    def getCoords(self) -> np.array:
        """
        camelCase to mimic ProDy

        Returns:
            np.array: Lx3 shape array
        """
        if self._coords is None:
            coords = []
            # chain has format: {<residue_key>: {<atom_type>: np.array([x,y,z], "name": <res_name>)}}
            for res in self.chain.values():
                if "CA" in res:
                    coords.append(res["CA"])
                else:
                    coords.append(res["CB"])
            self._coords = np.array(coords)
        return self._coords
    
    @staticmethod
    def align_coords(c1:np.array,c2:np.array) -> tuple[np.array, np.array]:
        """Aligns the given two 3D coordinate sets"""
        # Calculate the centroid (center of mass) of each set of coordinates
        centroid1 = np.mean(c1, axis=0)
        centroid2 = np.mean(c2, axis=0)

        # Translate both sets of coordinates to their respective centroids
        c1_centered = c1 - centroid1
        c2_centered = c2 - centroid2

        # Calculate the covariance matrix
        covariance_matrix = np.dot(c2_centered.T, c1_centered)

        # Use singular value decomposition (SVD) to find the optimal rotation matrix
        u, _, vt = np.linalg.svd(covariance_matrix)
        rotation_matrix = np.dot(u, vt)

        # Apply the calculated rotation matrix to c2_centered
        c2_aligned = np.dot(c2_centered, rotation_matrix)
        return c1_centered, c2_aligned
    
    def TM_score(self, template:'Chain'):
        # getting and aligning coords
        c1, c2 = self.align_coords(self.getCoords(), template.getCoords())
        
        # Calculating score:
        L = len(c1)
        # d0 is less than 0.5 for L < 22 
        # and nan for L < 15 (root of a negative number)
        d0 = 1.24 * np.power(L - 15, 1/3) - 1.8
        d0 = max(0.5, d0) 

        # compute the distance for each pair of atoms
        di = np.sum((c1 - c2) ** 2, 1) # sum along first axis
        return np.sum(1 / (1 + (di / d0) ** 2)) / L
        

    def get_mutated_seq(self, muts:list[str], reversed:bool=False) -> tuple[str, str]:
        """
        Given the protein chain dict and a list of mutations, this returns the 
        mutated and reference sequences.
        
        IMPORTANT: Currently only works for substitution mutations.

        Parameters
        ----------
        `chain` : OrderedDict
            The chain dict of dicts (see `pdb_get_chains`) can be the reference or 
            mutated chain.
        `muts` : List[str]
            List of mutations in the form '{ref}{pos}{mut}' e.g.: ['A10G', 'T20C']
        `reversed` : bool, optional
            If True the input chain is assumed to be the mutated chain and thus the 
            mutations provided act in reverese to get the reference sequence, 
            by default False.

        Returns
        -------
        Tuple[str, str]
            The mutated and reference sequences, respectively.
        """
        # prepare the mutations:
        mut_dict = OrderedDict()
        for mut in muts:
            if reversed:
                mut, pos, ref = mut[0], mut[1:-1], mut[-1]
            else:
                ref, pos, mut = mut[0], mut[1:-1], mut[-1]
            mut_dict[pos] = (ref, mut)

        # apply mutations
        mut_seq = list(self.getSequence())
        mut_done = []
        for i, res in enumerate(self._chains):
            pos = str(res.getResnum())
            if pos in mut_dict: # should always be the case unless mutation passed in is incorrect (see check below)
                ref, mut = mut_dict[pos] 
                assert ref == mut_seq[i], f"source ref '{mut_seq[i]}' doesnt match with mutation ref '{ref}'"
                mut_seq[i] = mut
                mut_done.append(pos)
                
        # check    
        for m in mut_dict:
            if m not in mut_done:
                raise Exception('Mutation sequence translation failed (due to no res at target position).')
            
        return ''.join(mut_seq)
    
    @staticmethod
    def _pdb_get_chains(pdb_file: str, model:int=1) -> OrderedDict:
        """
        Reads a pdb file and returns a dict of dicts with the following structure:
            {<chain>: {<residue_key>: {<atom_type>: np.array([x,y,z], "name": <res_name>)}}}
            
        See [PDB file format](https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html).
            
        Parameters
        ----------
        `pdb_file` : str
            Path to pdb file
        `model`: int, optional
            Model number to read, by default 1.
            
        Returns
        -------
        OrderedDict
            Dict of dicts with the chain as the key and the value is a dict with the residue as the key
        """
        assert model == 1, 'Model selection not supported, only first model is read!'

        # read and filter
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
            chains = OrderedDict() # chain dict of dicts
            curr_res = None
            curr_chain = None
            for line in lines:
                if line[:6].strip() == 'ENDMDL': break # only get first model
                if (line[:6].strip() != 'ATOM'): continue # skip non-atom lines
                
                # only want CA and CB atoms
                atm_type = line[12:16].strip()
                if atm_type not in ['CA', 'CB']: continue
                icode = line[26].strip() # dumb icode because residues will sometimes share the same res num 
                                # (https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html)


                curr_res = int(line[22:26])
                curr_chain = line[21]

                # Glycine has no CB atom, so only CA is saved
                res_key = f"{curr_res}_{icode}"            
                chains.setdefault(curr_chain, OrderedDict())
                assert atm_type not in chains[curr_chain].get(res_key, {}), \
                        f"Duplicate {atm_type} for residue {res_key} in {pdb_file}"

                # adding atom to residue
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                chains[curr_chain].setdefault(res_key, OrderedDict())[atm_type] = np.array([x,y,z])

                # Saving residue name
                res_name = line[17:20].strip()
                assert ("name" not in chains[curr_chain].get(res_key, {})) or \
                    (chains[curr_chain][res_key]["name"] == res_name), \
                                            f"Inconsistent residue name for residue {res_key} in {pdb_file}"
                chains[curr_chain][res_key]["name"] = res_name
        return chains  
            
    def buildHessian(self, cutoff:int=15., g:float=1.0):
        """
        Build Hessian matrix for given coordinate set.
        - this is the most simplified version of the code from ProDy adapted for my purposes
        
        See http://prody.csb.pitt.edu/_modules/prody/dynamics/gnm.html#GNM for chekENMParameters 
        fn if more complex input parameters are needed.
        """
        coords = self.getCoords()
        n_atoms = coords.shape[0]
        dof = n_atoms * 3 # 3 dimensions

        kirchhoff = np.zeros((n_atoms, n_atoms), 'd')
        hessian = np.zeros((dof, dof), float)

        cutoff2 = cutoff * cutoff
        for i in range(n_atoms):
            res_i3 = i*3
            res_i33 = res_i3+3
            i_p1 = i+1
            i2j_all = coords[i_p1:, :] - coords[i]
            for j, dist2 in enumerate((i2j_all ** 2).sum(1)):
                if dist2 > cutoff2:
                    continue
                i2j = i2j_all[j]
                j += i_p1
                res_j3 = j*3
                res_j33 = res_j3+3
                super_element = np.outer(i2j, i2j) * (- g / dist2)
                hessian[res_i3:res_i33, res_j3:res_j33] = super_element
                hessian[res_j3:res_j33, res_i3:res_i33] = super_element
                hessian[res_i3:res_i33, res_i3:res_i33] = \
                    hessian[res_i3:res_i33, res_i3:res_i33] - super_element
                hessian[res_j3:res_j33, res_j3:res_j33] = \
                    hessian[res_j3:res_j33, res_j3:res_j33] - super_element
                kirchhoff[i, j] = -g
                kirchhoff[j, i] = -g
                kirchhoff[i, i] = kirchhoff[i, i] + g
                kirchhoff[j, j] = kirchhoff[j, j] + g
        return hessian
    
    def get_contact_map(self, display=False, title="Residue Contact Map") -> np.array:
        """
        Returns the residue contact map for that structure.
            See: `get_sequence` for details on getting the residue chain dict.

        Parameters
        ----------
        `display` : bool, optional
            If true will display the contact map, by default False
        `title` : str, optional
            Title for cmap plot, by default "Residue Contact Map"

        Returns
        -------
        Tuple[np.array]
            residue contact map as a matrix

        Raises
        ------
        KeyError
            KeyError if a non-glycine residue is missing CB atom.
        """
        
        # getting coords from residues
        coords = self.getCoords() # shape == (L,3) where L is the sequence length
    
        # Calculate the pairwise distance matrix
        pairwise_distances = np.sqrt(np.sum((coords[:, np.newaxis] - coords) ** 2, axis=-1))

        # Fill the upper triangle and the diagonal of the distance matrix
        np.fill_diagonal(pairwise_distances, 0)
        
        if display:
            plt.imshow(pairwise_distances)
            plt.title(title)
            plt.show()
            
        return pairwise_distances
    