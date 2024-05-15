# %%
import pandas as pd
from src.utils.residue import Chain


df = pd.read_csv('/cluster/home/t122995uhn/projects/MutDTA/PDBbind_all_geneNames.csv')

#%% identify pocket location:
def get_residue_range(pdb_filename):
    with open(pdb_filename, 'r') as pdb_file:
        chain_residues = {}
        for line in pdb_file:
            if line.startswith('ATOM'):
                chain_id = line[21].strip()
                residue_number = int(line[22:26].strip())
                if chain_id not in chain_residues:
                    chain_residues[chain_id] = set()
                chain_residues[chain_id].add(residue_number)
        
        chain_ranges = {}
        for chain_id, residues in chain_residues.items():
            min_residue = min(residues)
            max_residue = max(residues)
            chain_ranges[chain_id] = (min_residue, max_residue)
        
        return chain_ranges

# %%
from tqdm import tqdm
dir_p = '/cluster/home/t122995uhn/projects/data/pdbbind/v2020-other-PL/'
pocket_fp = lambda x: f'{dir_p}/{x}/{x}_pocket.pdb'
prot_fp = lambda x: f'{dir_p}/{x}/{x}_protein.pdb'

dfu = df.drop_duplicates(subset='code')

mapped_prange = {}
for idx, (code, seq) in tqdm(dfu[['code', 'prot_seq']].iterrows(), total=len(dfu)):
    code = code.lower()
    # find the seq range for all chains in pocket and in protein
    prot = get_residue_range(prot_fp(code))
    pocket = get_residue_range(pocket_fp(code))
    
    # find the right change that matches seq_len
    chainID = None
    for cid, s in Chain(prot_fp(code)):
        if s == seq: # +1 bc PDB starts at index 1
            chainID = cid
            break
    
    # Find the protein residue range for the same chain
    prot_ch = prot.get(chainID)
    if not prot_ch or chainID not in pocket:
        # print(f'No matching chain in protein file for {prot_fp(code)}')
        continue
    
    # Reset pocket range so that it matches exactly with proteins sequence that is 0-indexed:
    prange = (pocket[chainID][0] - prot_ch[0], pocket[chainID][1] - prot_ch[0])  # Convert to 0-indexed
    mapped_prange[code.upper()] = f"{prange[0]}-{prange[1]}"


# %%
# Add the pocket range to the dataframe
df['pocket_range'] = df['code'].map(mapped_prange)
df.to_csv("pdbbind_all_prange.csv")

# %%
