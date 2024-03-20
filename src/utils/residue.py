from matplotlib import pyplot as plt
import numpy as np
import os

# one hot encoding
def one_hot(x, allowable_set, cap=False):
    """Return the one-hot encoding of x as a numpy array."""
    if x not in allowable_set:
        if cap: # last element is the catch all/unknown
            x = allowable_set[-1]
        else:
            raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
            
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
                   'R', 'S', 'T', 'V', 'W', 'Y', 'X'] # X is unknown
    
    foldseek_tokens = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 
                       'i', 'k', 'l', 'm', 'n', 'p', 'q', 
                       'r', 's', 't', 'v', 'w', 'y', '#'] # '#' is mask/unknown
    
    
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
    def __init__(self, pdb_file:str, model:int=1, t_chain:str=None):
        """
        This class was created to mimic the AtomGroup class from ProDy but optimized for fast parsing 
        only parses what is required for ANM simulations.

        Args:
            pdb_file (str): file path to pdb file to parse
            model (int, optional): model to parse from file. Defaults to 1.
            t_chain (str, optional): target chain to focus on. Defaults to None.
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
        if self.t_chain.isdigit():
            # if t_chain is a number then we assume they are trying to index the chain
            # index starts at 1 so we do -1 to ensure this
            idx = int(self.t_chain)-1
            if idx < 0: raise IndexError('Chain index must start at 1')
            k = list(self._chains.keys())[idx]
            return self._chains[k]
        else:
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
    
    @property
    def sequence(self) -> str:
        return self.getSequence()
      
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
        mut_dict = {}
        for mut in muts:
            if reversed:
                mut, pos, ref = mut[0], mut[1:-1], mut[-1]
            else:
                ref, pos, mut = mut[0], mut[1:-1], mut[-1]
            mut_dict[pos] = [ref, mut, False] # False is to indicate if done

        # apply mutations
        mut_seq = list(self.getSequence())
        for i, res in enumerate(self.chain):
            pos = ''.join(res.split('_')) # split out icode from resnum
            if pos in mut_dict: # should always be the case unless mutation passed in is incorrect (see check below)
                ref, mut, _ = mut_dict[pos] 
                assert ref == mut_seq[i], f"Source ref '{mut_seq[i]}' "+\
                    f"doesnt match with mutation {ref}->{mut} at position {pos}."
                mut_seq[i] = mut
                mut_dict[pos][2] = True
                
        # check    
        for k,v in mut_dict.items():
            if not v[2]:
                raise Exception(f'Mutation sequence translation failed (due to no res at target position {k}).')
            
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
                alt_loc = line[16] # some can have multiple locations for each protein confirmation.
                res_name = line[17:20].strip()
                if res_name == 'UNK': continue # WARNING: unkown residues are skipped
                if atm_type not in ['CA', 'CB']: continue
                icode = line[26].strip() # dumb icode because residues will sometimes share the same res num 
                                # (https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html)

                curr_res = int(line[22:26])
                curr_chain = line[21]

                # Glycine has no CB atom, so only CA is saved
                res_key = f"{curr_res}_{icode}"            
                chains.setdefault(curr_chain, OrderedDict())
                
                # Only keep first alt_loc
                if atm_type in chains[curr_chain].get(res_key, {}) and bool(alt_loc.strip()):
                    continue
                    
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
    

import pandas as pd
from src import config as cfg
import os, subprocess, shutil, multiprocessing, logging
from tqdm import tqdm

class Ring3Runner():
    """
    RING $ -- Residue Interaction Network Generator - Version 3.0.0

    Input options:
    -i  <filename>   Input PDB/mmCif file
    -I  <filename>   Input text file with a list of PDB/mmCif files, absolute path needs to be specified for each file in the list
    --all_models      Calculates Ring on all models (default false)
    -m <number>      Model number to use (default first model)
    -c  <id>         Chain identifier to read (default all chains)

    Output options:
    --out_dir <dirname> Output node and edges to the selected dir. (default input dir)
    --md                It generates the standard output files inside the results/ folder and create a md subdirectory with 
                        four different types of data:
                            <fileName>_cm_<type>, all the contact maps, one per model. The first column is the model number
                            <fileName>_gcm_<type>, the global contact map, number of contacts across models
                            <fileName>_gfreq_<type>, how many times each node is in contact over all models
                            <fileName>_tdcm_<type>, how many contacts for each node (rows) and each model (columns)
                        Where <type> is the type of interaction: HBOND, IAC, IONIC, PICATION, PIPISTACK, SSBOND, VDW
    --write_out_repr    Writes the mmCif/PDB representation used by Ring

    Network options:
    -n <string>       Network policy to define a contact between 2 groups/residues
                        Options: closest, lollipop, ca, cb (default closest)
    -t <number>       Network distance threshold for the specified network type (default 5.0)
    -g <number>       Sequence separation (gap) between residues (default 3)
    --get_iac          Include generic Inter-Atomic Contacts (IAC) (default false)
    --no_het           Skip calculation of hetero atoms connections (default false)
    --water            Include water molecules (default false)
    --no_specific      Skip calculation of bond specification (default false)
    --energy           Calculates TAP and FRST potentials for each residue, slower. (default false)
    --no_add_H         Skip the addition of H atoms to amino acids and nucleotides (default false)

    Edges options:
    default:           Return multiple edges for a pair of nodes, those that should be the most interesting ones.
    --all_edges        Return all edges found for a pair of nodes. (default false)
    --best_edge        Return only the most valuable connection between two nodes (default false)

    Thresholds options:
    --relaxed       Relax distance thresholds (default false).
    -o <number>    Distance threshold for Salt bridges (default 4.0).
    -s <number>    Distance threshold for Disulfide bonds (default 2.5).
    -k <number>    Distance threshold for Pi-Pi interactions (default 6.5).
    -a <number>    Distance threshold for Pi-Cation interactions (default 5.0).
    -b <number>    Distance threshold for Hydrogen bonds (default 3.5).
    -w <number>    Distance (between atom surfaces) threshold for 
                    Van Der Waals interactions (default 0.5).

    -v             Verbose output
    """
    RING3_BIN = cfg.RING3_BIN
    default_args = " --all_models --md --relaxed" # relaxed threshold
    RELEVANT_INTERACTIONS = ['HBOND', 'PIPISTACK', 'PICATION', 'IONIC', 'VDW']
    RELEVANT_MD = 'gfreq'
    
    @staticmethod
    def _get_out_dir(pdb_fp:str, out_dir:str=None) -> str:
        """
        Returns the output directory for the given pdb file.
        """
        return out_dir or os.path.splitext(pdb_fp)[0] + "-ring3/"
    
    @staticmethod
    def _prepare_input(af_confs:list[str], pdb_fp:str=None, overwrite=False) -> str:
        """
        Prepares input of multiple pdb files into a single pdb file.
        
        Assumes simple pdb files with no Headers or extra models inside the file 
        (i.e.: af2 outputs)
        """
        pdb_fp = pdb_fp or af_confs[0]
        combined_pdb_fp = f'{os.path.splitext(pdb_fp)[0]}.pdb_af_combined'
        
        if os.path.exists(combined_pdb_fp) and not overwrite:
            logging.debug(f'Combined pdb file already exists at {combined_pdb_fp}, skipping...')
            return combined_pdb_fp

        def safe_write(f, lines):
            for i, line in enumerate(lines):
                # Removing model tags since they are added in the outer loop
                if line.strip().split()[0] == 'MODEL' or line.strip() == 'ENDMDL':
                    # logging.debug(f'Removing {i}:{line}')
                    continue
                # 'END' should always be the last line or second to last
                if line.strip() == 'END':
                    extra_lines = len(lines)-i
                    if extra_lines > 1: 
                        logging.warning(f'{extra_lines} extra lines after END in {c}')
                    break
                f.write(line)
                
        with open(combined_pdb_fp, 'w') as f:
            for i, c in enumerate(af_confs):
                if 'af_combined' in c: 
                    logging.debug(f'Skipping {c}')
                    continue
                # add MODEL tag
                # logging.debug(f'Adding MODEL {os.path.basename(c).split("model_")[-1].split("_seed")[0]}')
                f.write(f'MODEL {i+1}\n')
                with open(c, 'r') as c_f:
                    lines = c_f.readlines()
                    safe_write(f, lines)
                # add ENDMDL tag
                f.write('ENDMDL\n')
            f.write('END\n')
        return combined_pdb_fp
        
    @staticmethod
    def check_outputs(pdb_fp:str, out_dir:str=None) -> dict:
        """
        Checks to see that output files exist. If out_dir is None it uses input dirname.
        
        Returns a dictionary of the output files if they exist with interaction type as key.
        """
        pdb_name = os.path.splitext(os.path.basename(pdb_fp))[0]
        out_dir = Ring3Runner._get_out_dir(pdb_fp, out_dir)
        
        # check only relevant files
        md_dir = f'{out_dir}/md/'
        files = {}
        for i in Ring3Runner.RELEVANT_INTERACTIONS:
            files[i] = f'{md_dir}{pdb_name}.{Ring3Runner.RELEVANT_MD}_{i}'
                    
        # all files must exist
        for f in files.values():
            if not os.path.exists(f):
                # logging.debug(f'Output file not found at {f}')
                return None
        return files
    
    @staticmethod
    def cleanup(pdb_fp:str, out_dir:str=None, all=False) -> None:
        """
        Removes the output files generated by RING3.
        """
        if all:
            out_dir = Ring3Runner._get_out_dir(pdb_fp, out_dir)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
                logging.debug(f'Removed directory {out_dir}')
            return
        
        outputs = Ring3Runner.check_outputs(pdb_fp, out_dir)
        if outputs:
            for fp in outputs.values():
                os.remove(fp)
                logging.debug(f'Removed {fp}')

    @staticmethod
    def run(pdb_fp:str|list[str], out_dir:str=None, chain_id:str=None,
            verbose:bool=False, overwrite:bool=False) -> dict:
        """
        Runs RING3 on a given pdb file that contains multiple models to get 
        the residue interaction network. Forces --all_models and --md options.  

        Args:
            pdb_fp (str or list[str]): Input pdb file path with multiple models/confirmations or list of pdb file paths.
            out_dir (str, optional): Output directory to save the results, defaults to the input directory.
            chain_id (str, optional): Chain ID if the pdb file contains multiple chains. Defaults to None.
            
        returns:
            dict[str] file paths for <fileName>_gfreq_<type> for relevant edge types (self.RELEVANT_INTERACTIONS)
            
        **Example usage of output files:**
        ```
            edge_fp = files['HBOND']
            # output csv looks like this:
            # A:7:_:ASP	HBOND	A:10:_:GLU	0.75
            # where the columns are:
                # 1. Residue 1
                # 2. Interaction type
                # 3. Residue 2
                # 4. Frequency of interaction (0-1) out of all the conformations
            import pandas as pd
            df = pd.read_csv(edge_fp, sep='\t', 
                            names=['res1', 'type', 'res2', 'freq'])
        ```
        """
        if isinstance(pdb_fp, list):
            combined_pdb_fp = Ring3Runner._prepare_input(pdb_fp, pdb_fp[0], overwrite=overwrite)
            pdb_fp = combined_pdb_fp
        
        # check if pdb file exists
        if not os.path.exists(pdb_fp):
            raise FileNotFoundError(f'PDB file not found at {pdb_fp}')
        
        out_dir = Ring3Runner._get_out_dir(pdb_fp, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        # check to see if output files already exist 
        outputs = Ring3Runner.check_outputs(pdb_fp, out_dir)       
        if outputs:
            if not overwrite:
                logging.debug(f'RING3 output files already exist at {out_dir}, skipping...')
                return pdb_fp, outputs
            logging.warning(f'Overwriting RING3 output files at {out_dir}...')
        
        # build cmd and run
        cmd = f"{Ring3Runner.RING3_BIN} -i {pdb_fp}"
        cmd += f" --out_dir {out_dir}" if out_dir else ""
        cmd += f" -c {chain_id}" if chain_id else ""
        cmd += Ring3Runner.default_args
        cmd += " -v" if verbose else ""
        
        logging.debug(f"Running RING3 with command: {cmd}")
        out = subprocess.run(cmd, shell=True, capture_output=True, check=True)
        
        logging.debug(out.stdout.decode('utf-8'))
        logging.debug(out.stderr.decode('utf-8'))
        
        # getting output file names:
        pdb_name = os.path.splitext(os.path.basename(pdb_fp))[0]
        
        # checking existence of output files
        outputs = Ring3Runner.check_outputs(pdb_fp, out_dir)
        if not outputs:
            logging.error(f'RING3 failed for {pdb_fp}, output file(s) not found at {out_dir}.')
            raise FileNotFoundError(f'RING3 failed to generate outputs to {out_dir} for {pdb_name}.')
        
        return pdb_fp, outputs
    
    @staticmethod
    def _run_proc(args):
        pdb_fp, out_dir, verbose, overwrite = args
        try:
            return Ring3Runner.run(pdb_fp, out_dir, verbose, overwrite)
        except Exception as e:
            logging.error(e)
            return (pdb_fp, str(e))

    @staticmethod
    def run_multiprocess(pdb_fps:list[str]|list[list[str]], out_dir:str=None, verbose:bool=False, overwrite:bool=False,
                         processes=None):
        """
        Runs RING3 on multiple PDB files using multiprocessing.
        
        Args:
            `pdb_fps` (list[str] | list[list[str]]): List of input pdb file paths as single file with muliple 
                                                  "models" or multiple pdb files with a single model each.
            `out_dir` (str, optional): Output directory to save the results, defaults to the input directory.
            `chain_id` (str, optional): Chain ID if the pdb file contains multiple chains. Defaults to None.
            `verbose` (bool, optional): Whether to display verbose output. Defaults to False.
            `overwrite` (bool, optional): Whether to overwrite existing output files. Defaults to False.
            `processes` (int, optional): # of cores to distribute across, default is to use all available.
            
        Returns:
            results list[tuple[str]]: list of (PDB file paths, output file paths)
        """
        args = [(pdb_fp, out_dir, verbose, overwrite) for pdb_fp in pdb_fps]
        
        with multiprocessing.Pool(processes=processes) as pool:
            results = list(tqdm(pool.imap(Ring3Runner._run_proc, args),
                                total=len(args),
                                desc="Running multiproc. RING3"))
        return results
    
    @staticmethod
    def build_cmap(output_gfreq_fp:str, res_len:int, self_loop=True) -> np.ndarray:
        """Converts the output gfreq file to a contact map"""
        cmap = np.eye(res_len) if self_loop else np.zeros((res_len, res_len))
        
        # build cmap with values from csv
        df = pd.read_csv(output_gfreq_fp, sep='\t', 
                         names=['res1', 'type', 'res2', 'freq'])
        
        # fill cmap
        for _, row in df.iterrows():
            i = int(row['res1'].split(':')[1]) - 1 # index starts at 1
            j = int(row['res2'].split(':')[1]) - 1 # index starts at 1
            cmap[i, j] = row['freq']
            cmap[j, i] = row['freq']
            
        return cmap
