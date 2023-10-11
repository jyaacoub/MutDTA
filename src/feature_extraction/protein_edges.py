from typing import Iterable, Tuple
import numpy as np
from src.utils.residue import Chain


def get_target_edge(target_sequence:str, contact_map:str or np.array,
                          threshold=10.5) -> Tuple[np.array]:
    """
    Returns edge index for target sequence given a contact map.

    Parameters
    ----------
    `target_sequence` : str
        Sequence of the target protein.
    `contact_map` : strornp.array
        File path for contact map or the actual map itself.
    `threshold` : float, optional
        Threshold for what defines an edge, by default 10.5
        
    Returns
    -------
    Tuple[np.array]
        edge index, edge weight for target sequence
    """
    # loading up contact map if it is a file path
    if type(contact_map) == str: contact_map = np.load(contact_map)
    
    target_size = len(target_sequence)
    assert contact_map.shape[0] == contact_map.shape[1], 'contact map is not square'
    # its ok if it is smaller, but not larger (due to missing residues in pdb)
    assert contact_map.shape[0] == target_size, \
            f'contact map size does not match target sequence size,'+\
            f'{contact_map.shape[0]} != {target_size}'
    
    
    # threshold
    # array of points for edge index (2,L) where L < seq_len**2
    if threshold >= 0.0:
        # NOTE: for real cmaps self loop is implied since the diagonal is 0
        edge_index = np.array(np.where(contact_map <= threshold))
        edge_weight = contact_map[edge_index[0], edge_index[1]]
        # normalize to be between 6A and 14A
        #  - "in contact" is anywhere between 8 and 12A: https://en.wikipedia.org/wiki/Protein_contact_map
        a_min, a_max = 6.0, 14.0
        edge_weight = np.clip(edge_weight, a_min, a_max)
        edge_weight = (edge_weight - a_min) / (a_max - a_min)
        
    else: # negative threshold flips the sign
        contact_map += np.matrix(np.eye(contact_map.shape[0])*threshold) # Self loop
        edge_index = np.array(np.where(contact_map >= abs(threshold)))
        # no norm needed for probabilistic cmaps
        edge_weight = contact_map[edge_index[0], edge_index[1]]  
        
    assert edge_index.max() < target_size, 'contact map size does not match target sequence size'
    
    return edge_index, edge_weight


################## Temporary fix until issue is resolved (https://github.com/prody/ProDy/issues/1749) ##################
from prody import Mode, NMA, ModeSet, calcCovariance
from prody.utilities import div0
import time
def calcCrossCorr(modes, n_cpu=1, norm=True):
    """Returns cross-correlations matrix.  For a 3-d model, cross-correlations
    matrix is an NxN matrix, where N is the number of atoms.  Each element of
    this matrix is the trace of the submatrix corresponding to a pair of atoms.
    Covariance matrix may be calculated using all modes or a subset of modes
    of an NMA instance.  For large systems, calculation of cross-correlations
    matrix may be time consuming.  Optionally, multiple processors may be
    employed to perform calculations by passing ``n_cpu=2`` or more."""

    if not isinstance(n_cpu, int):
        raise TypeError('n_cpu must be an integer')
    elif n_cpu < 1:
        raise ValueError('n_cpu must be equal to or greater than 1')

    if not isinstance(modes, (Mode, NMA, ModeSet)):
        if isinstance(modes, list):
            try:
                is3d = modes[0].is3d()
            except:
                raise TypeError('modes must be a list of Mode or Vector instances, '
                            'not {0}'.format(type(modes)))
        else:
            raise TypeError('modes must be a Mode, NMA, or ModeSet instance, '
                            'not {0}'.format(type(modes)))
    else:
        is3d = modes.is3d()
    if is3d:
        model = modes
        if isinstance(modes, (Mode, ModeSet)):
            model = modes._model
            if isinstance(modes, (Mode)):
                indices = [modes.getIndex()]
                n_modes = 1
            else:
                indices = modes.getIndices()
                n_modes = len(modes)
        else:
            n_modes = len(modes)
            indices = np.arange(n_modes)
        array = model._getArray()
        n_atoms = model._n_atoms
        variances = model._vars
        if n_cpu == 1:
            s = (n_modes, n_atoms, 3)
            arvar = (array[:, indices]*variances[indices]).T.reshape(s)
            array = array[:, indices].T.reshape(s)
            covariance = np.tensordot(array.transpose(2, 0, 1),
                                      arvar.transpose(0, 2, 1),
                                      axes=([0, 1], [1, 0]))
        else:
            import multiprocessing
            n_cpu = min(multiprocessing.cpu_count(), n_cpu)
            queue = multiprocessing.Queue()
            size = n_modes // n_cpu
            for i in range(n_cpu):
                if n_cpu - i == 1:
                    indices = modes.getIndices()[i*size:]
                else:
                    indices = modes.getIndices()[i*size:(i+1)*size]
                process = multiprocessing.Process(
                    target=_crossCorrelations,
                    args=(queue, n_atoms, array, variances, indices))
                process.start()
            while queue.qsize() < n_cpu:
                time.sleep(0.05)
            covariance = queue.get()
            while queue.qsize() > 0:
                covariance += queue.get()
    else:
        covariance = calcCovariance(modes)
    if norm:
        diag = np.power(covariance.diagonal(), 0.5)
        D = np.outer(diag, diag)
        covariance = div0(covariance, D)
    return covariance

def _crossCorrelations(queue, n_atoms, array, variances, indices):
    """Calculate covariance-matrix for a subset of modes."""

    n_modes = len(indices)
    arvar = (array[:, indices] * variances[indices]).T.reshape((n_modes,
                                                                n_atoms, 3))
    array = array[:, indices].T.reshape((n_modes, n_atoms, 3))
    covariance = np.tensordot(array.transpose(2, 0, 1),
                              arvar.transpose(0, 2, 1),
                              axes=([0, 1], [1, 0]))
    queue.put(covariance)

####################################

def get_cross_correlation(pdb_fp:str, target_seq:str, n_modes=10, n_cpu=1):
    """Gets the cross correlation matrix after running ANM simulation w/ProDy"""
    from prody import calcANM

    chain = Chain(pdb_fp)
    #WARNING: assuming the target_chain is always the max len chain!
    assert chain.getSequence() == target_seq, f'Target seq is not chain seq {pdb_fp}'
    anm = calcANM(chain.hessian, selstr='calpha', n_modes=n_modes)

    
    cc = calcCrossCorr(anm[:n_modes], n_cpu=n_cpu)
    cc_min, cc_max = cc.min(), cc.max()
    # min-max normalization into [0,1] range
    return (cc-cc_min)/(cc_max-cc_min)

def get_af_edge_weights(chains:Iterable[Chain]) -> np.array:
    M = np.array([c.get_contact_map() for c in chains])
    return np.sum(M < 8.0, axis=0)/len(M)

def get_target_edge_weights(pdb_fp:str, target_seq:str, edge_opt:str,
                            n_modes:int=5, n_cpu=4,
                            cmap:str or np.array=None,
                            af_confs:Iterable[str]=None,
                            filter=False) -> np.array:
    """ Returns an LxL matrix of the edge weights"""
    # edge weights should be returned as a list of Z weights
    # where Z is the number of edges (|E|)
    if edge_opt is None or edge_opt == 'binary':
        return None
    elif edge_opt == 'anm':
        # shape of |V|x|V| (V=vertices |V|=len(target_seq))
        cc = get_cross_correlation(pdb_fp, target_seq, n_modes, n_cpu=n_cpu)
        return cc
    elif edge_opt == 'simple':
        assert cmap is not None, "Simple edge selected, but no contact map passed in."
        if type(cmap) == str: cmap = np.load(cmap)
        # normalize cmap from 0.0 to 1.0 range using min-max normalization
        cmap_min, cmap_max = cmap.min(), cmap.max()
        return (cmap-cmap_min)/(cmap_max-cmap_min)
    elif edge_opt == 'af2':
        chains = [Chain(p) for p in af_confs]
        # filter chains by template modeling score:
        if filter:
            template = Chain(pdb_fp)
            chains = [c for c in chains if 0.85 < c.TM_score(template) < 0.98]
        
        #TODO: if chains is empty then just return np.zeros matrix
        if len(chains) == 0:
            print(f'WARNING: no af2 pdbs for {pdb_fp}')
            return np.ones(shape=(len(target_seq), len(target_seq)))
        else:
            ew = get_af_edge_weights(chains=chains)
            assert len(ew) == len(target_seq), f'Mismatch sequence length for {pdb_fp}'
            return ew
    else:
        raise ValueError(f'Invalid edge_opt {edge_opt}')
