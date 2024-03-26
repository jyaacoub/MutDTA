import os
import math
from typing import Iterable
import logging

def merge_pdb(files:Iterable[str], pdb_fp:str=None,overwrite=False):
    """
    Merges multiple individual model files (from af2) into one pdb file.
    """
    pdb_fp = pdb_fp or files[0]
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
        for i, c in enumerate(files):
            if 'af_combined' in c: 
                logging.debug(f'Skipping {c}')
                continue
            # add MODEL tag
            f.write(f'MODEL {i+1}\n')
            with open(c, 'r') as c_f:
                lines = c_f.readlines()
                safe_write(f, lines)
            # add ENDMDL tag
            f.write('ENDMDL\n')
        f.write('END\n')
    return combined_pdb_fp

def split_fused_pdb(dir, res_start):
    for f in os.listdir(dir):
        fp = os.path.join(dir, f)
        print(fp)
        with open(fp, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line[:6].strip() == 'ATOM' and int(line[23:26]) >= res_start:
                    break
            
            i = i if i < len(lines)-1 else 0
            out_lines = ''.join(lines[i:]) 
            
        with open(fp, 'w') as f:
            f.write(out_lines)

def split_models(fpi,fpo):
    os.makedirs(os.path.dirname(fpo(0)), exist_ok=True)

    with open(fpi, 'r') as f:
        mdl_n = 0
        buffer = ''
        for line in f.readlines():
            if line[:6].strip() == "MODEL":
                # print(line)
                new_mdl_n = int(line[10:].strip())
                if new_mdl_n != mdl_n:
                    # print(f'{mdl_n} -> {new_mdl_n}')
                    with open(fpo(mdl_n), 'w') as fo:
                        fo.write(buffer)
                        
                    mdl_n = new_mdl_n
                    buffer = ''
            buffer += line

    if len(buffer) > 0:
        with open(fpo(mdl_n), 'w') as fo:
            fo.write(buffer)
        buffer = ''
              
def remove_res_tails(fpi:str, fpo:str, max_res:int=math.inf, inverse=False):
    """
    Removes target residues, useful for removing tails from a file containing multiple models.
    """
    os.makedirs(os.path.dirname(fpo), exist_ok=True)
    
    def isnt_valid(num):
        return (inverse and num < max_res) or \
                (not inverse and num > max_res)
    
    with open(fpi, 'r') as f:
        mdl_n = 0
        lines = ''
        for line in f.readlines():
            if line[:6].strip() == "MODEL":
                new_mdl_n = int(line[10:].strip())
                mdl_n = new_mdl_n
            elif line[:6].strip() == "ATOM" and \
                isnt_valid(int(line[22:26])): # not within valid range
                continue
            lines += line

    print("num models parsed:", mdl_n)
    with open(fpo, 'w') as fo:
        fo.write(lines)
        
def remove_linker(fpi, fpo, start_res=149, end_res=156, rename_chain=True):
    os.makedirs(os.path.dirname(fpo), exist_ok=True)
    
    def isnt_valid(num): # returns true if part of linker
        return start_res <= num <= end_res
    
    with open(fpi, 'r') as f:
        mdl_n = 0
        lines = ''
        found_linker = False
        for line in f.readlines():
            tag = line[:6].strip()
            if tag == "MODEL":
                found_linker = False
                new_mdl_n = int(line[10:].strip())
                mdl_n = new_mdl_n
            elif tag == "ATOM" and \
                isnt_valid(int(line[22:26])): # not within valid range
                found_linker = True
                continue
            
            if found_linker and rename_chain and (tag == "ATOM" or 
                                                  tag == "HETATM"):
                # residues after linker will have new chainID
                cID = line[21]
                arr =  list(line)
                arr[21] = chr(ord(cID) + 1)
                line = ''.join(arr)
            
            lines += line

    print("num models parsed:", mdl_n)
    with open(fpo, 'w') as fo:
        fo.write(lines)
        
def reset_numbering(fpi, fpo, rename_chain=None, model_n_max=20):
    """Resets residue and serial numbering, also limits number of models to model_n_max"""
    os.makedirs(os.path.dirname(fpo), exist_ok=True)
    
    # tags with serial, chainID, and resSeq info
    # located at 7-11,       22, and 23-26 (for all tags)
    #
    # https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
    target_tags = {'ATOM', 'HETATM', 'ANISOU', 'TER'}
    with open(fpi, 'r') as f:
        mdl_n = 0
        lines = ''
        serial_n = 1
        res_n = 1
        prev_res_n = None
        for line in f.readlines():
            tag = line[:6].strip()
            if tag == "MODEL":
                # update model number
                curr_mdl_n = int(line[10:].strip())
                if curr_mdl_n > model_n_max: break
                mdl_n = curr_mdl_n
                # reset numbering back to zero
                serial_n = 1
                res_n = 1
                prev_res_n = None
                
            if tag in target_tags:
                curr_res_n = int(line[22:26])
                if prev_res_n is None: # initalize prev_res_n
                    prev_res_n = curr_res_n
                
                arr = list(line)
                # update serial number
                arr[7:11] = list(f'{serial_n:4d}')
                serial_n += 1
                                
                # checking to see if we can increment the res number
                if prev_res_n != curr_res_n:
                    res_n += 1
                    prev_res_n = curr_res_n
                    
                # update res number
                arr[22:26] = list(f'{res_n:4d}')
            
                # updating chainID to new ID
                if rename_chain is not None:
                    arr[21] = rename_chain
                
                line = ''.join(arr)
            
            lines += line

    print("num models parsed:", mdl_n)
    with open(fpo, 'w') as fo:
        fo.write(lines)
                

if __name__ == '__main__':
    # MB0_all = '/cluster/home/t122995uhn/projects/MYC-PNUTS/MB0/MB0_all.pdb'
    # MB0_reset = '/cluster/home/t122995uhn/projects/MYC-PNUTS/MB0/MB0_reset.pdb'
    # reset_numbering(MB0_all, MB0_reset, rename_chain='B')
    
    from glob import glob
    import os
    
    target='ELOA'
    root_dir = '/cluster/home/t122995uhn/projects/MYC-PNUTS/haddock_runs/local/'
    
    for target in ['IWS1', 'MED26', 'PSIP1', 'TFIIS']:
        print(target)
        pdb =              glob(f'{root_dir}/{target}/data/????.pdb')[0]
        pdb_nolink =       os.path.splitext(pdb)[0] + '_nolink.pdb'
        pdb_nolink_reset = os.path.splitext(pdb_nolink)[0] + '_reset.pdb'
        
        if target == 'ELOA':
            remove_linker(pdb, pdb_nolink, start_res=109, end_res=100000)
        elif target == 'IWS1':
            remove_linker(pdb, pdb_nolink, start_res=693, end_res=100000)
        elif target == 'MED26':
            remove_linker(pdb, pdb_nolink, start_res=89, end_res=100000)
        elif target == 'PSIP1': # NOTE: there is no linker residue for this protein 
            pdb_nolink = pdb
            pass
        elif target == 'TFIIS':
            remove_linker(pdb, pdb_nolink, start_res=80, end_res=100000)
        else:
            raise Exception('Invalid target choice.')
        
        reset_numbering(pdb_nolink, pdb_nolink_reset, rename_chain='A', model_n_max=20)
        

# np.array([648,649,651,652,654,656,658,659,662,665,667,677,684]) - 544