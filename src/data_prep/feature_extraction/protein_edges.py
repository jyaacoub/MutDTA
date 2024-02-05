import logging
from typing import Iterable, Tuple
import numpy as np
from prody import calcANM

from src.utils.residue import Chain, Ring3Runner


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

def get_cross_correlation(pdb:str or Chain, target_seq:str=None, n_modes=10, n_cpu=1):
    """Gets the cross correlation matrix after running ANM simulation w/ProDy"""
    if isinstance(pdb, str):
        chain = Chain(pdb)
    elif isinstance(pdb, Chain):
        chain = pdb
    else:
        raise TypeError(f'pdb arg is not str or Chain {pdb}')
        
    assert target_seq is None or chain.getSequence() == target_seq, f'Target seq is not chain seq {pdb}'
    anm = calcANM(chain.hessian, selstr='calpha', n_modes=n_modes)

    
    cc = calcCrossCorr(anm[:n_modes], n_cpu=n_cpu)
    cc_min, cc_max = cc.min(), cc.max()
    # min-max normalization into [0,1] range
    return (cc-cc_min)/(cc_max-cc_min)

def get_af_edge_weights(chains:Iterable[Chain], anm_cc=False, n_modes=5, n_cpu=4) -> np.array:
    if anm_cc:
        # run anm on each chain then average the cmaps
        M = np.array([get_cross_correlation(c, n_modes=n_modes, 
                                            n_cpu=n_cpu) for c in chains])
    else:
        M = np.array([c.get_contact_map() for c in chains]) < 8.0
    
    # simple averaging of cmaps/crosscorr.
    return np.sum(M, axis=0) / len(M)

def get_target_edge_weights(pdb_fp:str, target_seq:str, edge_opt:str,
                            n_modes:int=5, n_cpu=4,
                            cmap:str or np.array=None,
                            af_confs:Iterable[str]=None,
                            filter=False) -> np.array:
    """
    Returns an LxL matrix representing the edge weights of the protein

    Args:
        pdb_fp (str): The pdb file to extract edge weights from, or in the case of 
        'af2' this is the template path for filtering out misfolds.
        target_seq (str): Target sequence for sanity checking that the pdbs match.
        edge_opt (str): See src.utils.config.EDGE_OPT
        
        n_modes (int, optional): ANM arg. Defaults to 5.
        n_cpu (int, optional): ANM arg. Defaults to 4.
        
        cmap (str or np.array, optional): contact map for 'simple'. Defaults to None.
        
        af_confs (Iterable[str], optional): configurations for af2 structs. Defaults to None.
        filter (bool, optional): Whether or not to filter misfolds in 'af2'. Defaults to False.

    Raises:
        ValueError: invalid edge option (See `src.utils.config.EDGE_OPT`)

    Returns:
        np.array: The LxL edge weight matrix or None if binary, or LxLxZ if edge attributes are selected like ring3.
    """
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
    elif 'af2' in edge_opt:
        chains = [Chain(p) for p in af_confs]
        # filter chains by template modeling score:
        if filter:
            template = Chain(pdb_fp)
            chains = [c for c in chains if 0.85 < c.TM_score(template) < 0.98]
        
        # NOTE: if chains (no pdbs found) is empty then we treat all edges as the same
        if len(chains) == 0:
            print(f'WARNING: no af2 pdbs for {pdb_fp}')
            # treat all edges as the same if no confirmations are found
            return np.ones(shape=(len(target_seq), len(target_seq))) 
        else:
            # NOTE: af2-anm gets run here:
            ew = get_af_edge_weights(chains=chains, anm_cc=('anm' in edge_opt))
            assert len(ew) == len(target_seq), f'Mismatch sequence length for {pdb_fp}'
            return ew
    elif edge_opt == 'ring3':
        chains = [Chain(p) for p in af_confs]
        M = np.array([c.get_contact_map() for c in chains]) < 8.0

        dist_cmap = np.sum(M, axis=0) / len(M)

        # ring3 edge attribute extraction
        # Note: this will create a "combined" pdb file in the same directory as the confirmaions
        input_pdb, files = Ring3Runner.run(af_confs, overwrite=False)
        seq_len = len(Chain(input_pdb))
        logging.info(f'Ring3Runner seq_len: {seq_len}')

        # Converts output files into LxLx6 matrix for the 6 ring3 edge attributes
        r3_cmaps = []
        for k, fp in files.items():
            cmap = Ring3Runner.build_cmap(fp, seq_len)
            r3_cmaps.append(cmap)
        
        # Convert to numpy array of shape (L, L, 6)
        all_cmaps = np.array(r3_cmaps + [dist_cmap], dtype=np.float32) # [6, L, L]
        all_cmaps = all_cmaps.transpose(1,2,0) # [L, L, 6]
        
        logging.info(f'Ring3Runner all_cmaps: {all_cmaps.shape}')

        # deletes all intermediate output files, since the main LxLx6 matrix should be saved at the end
        Ring3Runner.cleanup(input_pdb, all=True)
        return all_cmaps
    else:
        raise ValueError(f'Invalid edge_opt {edge_opt}')
    
if __name__ == '__main__':
    # for multiprocessing across multiple job arrays 
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import os, re

    from glob import glob
    from src.data_prep.datasets import BaseDataset
    from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights

    data = 'davis'
    data_dir = '/cluster/home/t122995uhn/projects/data/'
    csv = f'{data_dir}/DavisKibaDataset/davis/nomsa_af2-anm/full/XY.csv'
    raw_dir = f'{data_dir}/{data}/'
    af_conf_dir = f'../colabfold/{data}_af2_out/'

    def af_conf_files(code) -> list[str]:
        code = re.sub(r'[()]', '_', code)
        return glob(f'{af_conf_dir}/out?/{code}_unrelaxed_rank_*.pdb')

    def pdb_p(code, safe=True):
        code = re.sub(r'[()]', '_', code)
        # davis and kiba dont have their own structures so this must be made using 
        # af or some other method beforehand.
        file = glob(os.path.join(af_conf_dir, f'highQ/{code}_unrelaxed_rank_001*.pdb'))
        # should only be one file
        assert not safe or len(file) == 1, f'Incorrect pdb pathing, {len(file)}# of structures for {code}.'
        return file[0] if len(file) >= 1 else None


    df = pd.read_csv(csv, index_col=0)
    #################### Get unique proteins:
    unique_df = BaseDataset.get_unique_prots(df)

    #%%######################### Get job partition
    num_arrays = 140 # NOTE: number of ARRAYS HERE
    array_idx = 0#${SLURM_ARRAY_TASK_ID}
    partition_size = len(unique_df) / num_arrays
    start, end = int(array_idx*partition_size), int((array_idx+1)*partition_size)

    unique_df_part = unique_df[start:end]

    print(unique_df_part.index)
    ##################################### Run hhblits
    np_dir = os.path.join(raw_dir, 'edge_weights', 'af2-anm')
    os.makedirs(np_dir, exist_ok=True)

    # running
    for code, (prot_id, pro_seq) in tqdm(
                    unique_df_part[['prot_id', 'prot_seq']].iterrows(), 
                    desc='Running edgw',
                    total=len(unique_df_part)):
        out_fp = os.path.join(np_dir, f"{code}.npy")
        af_confs = af_conf_files(code)
        
        if not os.path.isfile(out_fp):
            pro_edge_weight = get_target_edge_weights(pdb_p(code), pro_seq, 
                                                edge_opt='af2-anm',
                                                n_modes=5, n_cpu=4,
                                                af_confs=af_confs)
            np.save(out_fp, pro_edge_weight)
            
    ################################ ANM VERSION IS PRETTY MUCH THE SAME:
    # for multiprocessing across multiple job arrays 
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import os, re

    from glob import glob
    from src.data_prep.datasets import BaseDataset
    from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights

    data = 'davis'
    edgew = 'anm'
    data_dir = '/cluster/home/t122995uhn/projects/data/'
    csv = f'{data_dir}/DavisKibaDataset/davis/nomsa_anm_original_binary/full/XY.csv'
    raw_dir = f'{data_dir}/{data}/'
    af_conf_dir = f'../colabfold/{data}_af2_out/'

    def pdb_p(code, safe=True):
        code = re.sub(r'[()]', '_', code)
        # davis and kiba dont have their own structures so this must be made using 
        # af or some other method beforehand.
        file = glob(os.path.join(af_conf_dir, f'highQ/{code}_unrelaxed_rank_001*.pdb'))
        # should only be one file
        assert not safe or len(file) == 1, f'Incorrect pdb pathing, {len(file)}# of structures for {code}.'
        return file[0] if len(file) >= 1 else None


    # Get protein names:
    df = pd.read_csv(csv, index_col=0)
    unique_df = BaseDataset.get_unique_prots(df)

    #%%## Get job partition ###
    num_arrays = 140 # NOTE: number of ARRAYS HERE
    array_idx = 46#${SLURM_ARRAY_TASK_ID}
    partition_size = len(unique_df) / num_arrays
    start, end = int(array_idx*partition_size), int((array_idx+1)*partition_size)

    unique_df_part = unique_df[start:end]

    print(unique_df_part.index)

    #%%### Run anm simulation ###
    np_dir = os.path.join(raw_dir, 'edge_weights', 'anm')
    os.makedirs(np_dir, exist_ok=True)

    # running
    for code, (prot_id, pro_seq) in tqdm(
                    unique_df_part[['prot_id', 'prot_seq']].iterrows(), 
                    desc='Running edgw',
                    total=len(unique_df_part)):
        out_fp = os.path.join(np_dir, f"{code}.npy")

        if not os.path.isfile(out_fp):
            pro_edge_weight = get_target_edge_weights(pdb_p(code), pro_seq, 
                                                edge_opt='anm',
                                                n_modes=5, n_cpu=4)
            np.save(out_fp, pro_edge_weight)

