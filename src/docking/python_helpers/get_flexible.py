# %%
import os, sys, argparse

def get_flexres_argstring(pocket_file:str) -> str:
    """
    Creates an argstring to pass to ADT script `prepare_flexreceptor4.py` 
    given an input pdb file containing the residues to be made flexible.
    
    e.g.: the input file can be a pocket file containing residues in 
    the binding pocket of the protein.

    Parameters
    ----------
    `pocket_file` : str
        The path to the pdb file containing the residues to be made flexible.

    Returns
    -------
    str
        The argstring to pass to ADT script `prepare_flexreceptor4.py`
    """

    with open(pocket_file, 'r') as f:
        arg_string = ''
        mol_name = ''#os.path.basename(pocket_file).split('.')[0] # no model name since pdbqt files remove them
        prev_chain = None
        residues_added = {} # key: chain:residue, value: True
        for l in f.readlines():
            if l[:6].strip() == 'TER':
                print('new chain!')
            
            if l[:6].strip() != 'ATOM': continue
            curr_res_name = l[17:20].strip()
            curr_chain = l[21:22]
            curr_res_n = l[22:26].strip()
            curr_res = f'{curr_res_name}{curr_res_n}'
            
            # To avoid duplicates we make sure we haven't added this residue already
            res_id = f'{curr_chain}:{curr_res}'
            if res_id not in residues_added:
                residues_added[res_id] = True
            else:
                continue
            
            if prev_chain is None:
                prev_chain = curr_chain # first res in the argstring
                arg_string += f'{mol_name}:{curr_chain}:{curr_res}'
            else:
                if prev_chain != curr_chain: #adding ,
                    arg_string += f',{mol_name}:{curr_chain}:{curr_res}'
                else: # append to res list
                    arg_string += f'_{curr_res}'
    return arg_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an argstring to pass to ADT script `prepare_flexreceptor4.py` given an input pdb file containing the residues to be made flexible.')
    parser.add_argument('-pf','--pocket_file', type=str, 
                        help='The path to the pdb file containing the residues to be made flexible.',
                        default='/home/jyaacoub/projects/data/refined-set/1a1e/1a1e_pocket.pdb')
    args = parser.parse_args()
    pocket_file = args.pocket_file
    
    arg_string = get_flexres_argstring(pocket_file)
    print(arg_string) # output to stdout to be used in bash scripts
    # with open(pocket_file+'.argstring', 'w') as f:
    #     f.write(arg_string)
            