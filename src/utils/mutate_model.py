import sys
import os
import logging
import numpy as np
from tqdm import tqdm
from src.utils.residue import ResInfo, Chain
from src.data_prep.quick_prep import get_protein_features

# try catch around modeller since it is only really needed for run_mutagenesis.py
try:
    from modeller import *
    from modeller.optimizers import MolecularDynamics, ConjugateGradients
    from modeller.automodel import autosched
except ImportError:
    logging.warning("Modeller failed to import - will not able to run mutagenesis scripts.")


def optimize(atmsel, sched):
    #conjugate gradient
    for step in sched:
        step.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
    #md
    refine(atmsel)
    cg = ConjugateGradients()
    cg.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)


#molecular dynamics
def refine(atmsel):
    # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
    md = MolecularDynamics(cap_atom_shift=0.39, md_time_step=4.0,
                           md_return='FINAL')
    init_vel = True
    for (its, equil, temps) in ((200, 20, (150.0, 250.0, 400.0, 700.0, 1000.0)),
                                (200, 600,
                                 (1000.0, 800.0, 600.0, 500.0, 400.0, 300.0))):
        for temp in temps:
            md.optimize(atmsel, init_velocities=init_vel, temperature=temp,
                         max_iterations=its, equilibrate=equil)
            init_vel = False


#use homologs and dihedral library for dihedral angle restraints
def make_restraints(mdl1, aln):
   rsr = mdl1.restraints
   rsr.clear()
   s = Selection(mdl1)
   for typ in ('stereo', 'phi-psi_binormal'):
       rsr.make(s, restraint_type=typ, aln=aln, spline_on_site=True)
   for typ in ('omega', 'chi1', 'chi2', 'chi3', 'chi4'):
       rsr.make(s, restraint_type=typ+'_dihedral', spline_range=4.0,
                spline_dx=0.3, spline_min_points = 5, aln=aln,
                spline_on_site=True)


def run_modeller(modelname:str, respos:int|str, restyp:str, chain:str, out_fp:str=None, overwrite=False, 
                 n_attempts=5):
    """
    Takes in the model path (excluding .pdb extension) and the residue index to 
    change and the new residue. Outputs to same dir as "{modelname}-{respos}_restype.pdb"

    Args:
        modelname (str): Model file path.
        respos (int): Index position for target residue (1-indexed)
        restyp (str): 3 letter residue name for what to change into
        chain (str): Single character chain identifier.
        out_path (str): output path for pdb file to override default of "{modelname}-{respos}_restype.pdb".
        n_attempts (int): number of attempts to try for aleviating steric clashes
    """
    modelname = modelname.split('.pdb')[0]
    if out_fp and modelname == out_fp.split('.pdb')[0]:
        if overwrite:
            logging.warning(f"overwritting pdb file at {out_fp}")
        else:
            raise FileExistsError(f'would overwrite existing file at "{modelname}.pdb"')
    
    respos = str(respos)
    log.none()

    TMP_FILE_PATH = f"{modelname}-{restyp}_{respos}.tmp"
    OUT_FILE_PATH = out_fp or f"{modelname}-{restyp}_{respos}.pdb"
    
    # Set a different value for rand_seed to get a different final model
    while n_attempts > 0:
        env = Environ(rand_seed=-49837+n_attempts)

        env.io.hetatm = True
        #soft sphere potential
        env.edat.dynamic_sphere=False
        #lennard-jones potential (more accurate)
        env.edat.dynamic_lennard=True
        env.edat.contact_shell = 4.0
        env.edat.update_dynamic = 0.39

        # Read customized topology file with phosphoserines (or standard one)
        env.libs.topology.read(file='$(LIB)/top_heav.lib')

        # Read customized CHARMM parameter library with phosphoserines (or standard one)
        env.libs.parameters.read(file='$(LIB)/par.lib')


        # Read the original PDB file and copy its sequence to the alignment array:
        mdl1 = Model(env, file=modelname)
        ali = Alignment(env)
        ali.append_model(mdl1, atom_files=modelname, align_codes=modelname)

        #set up the mutate residue selection segment
        s = Selection(mdl1.chains[chain].residues[respos])

        #perform the mutate residue operation
        s.mutate(residue_type=restyp)
        #get two copies of the sequence.  A modeller trick to get things set up
        ali.append_model(mdl1, align_codes=modelname)

        # Generate molecular topology for mutant
        mdl1.clear_topology()
        mdl1.generate_topology(ali[-1])


        # Transfer all the coordinates you can from the template native structure
        # to the mutant (this works even if the order of atoms in the native PDB
        # file is not standard):
        #here we are generating the model by reading the template coordinates
        mdl1.transfer_xyz(ali)

        # Build the remaining unknown coordinates
        mdl1.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')

        #yes model2 is the same file as model1.  It's a modeller trick.
        mdl2 = Model(env, file=modelname)

        #required to do a transfer_res_numb
        #ali.append_model(mdl2, atom_files=modelname, align_codes=modelname)
        #transfers from "model 2" to "model 1"
        mdl1.res_num_from(mdl2,ali)

        #It is usually necessary to write the mutated sequence out and read it in
        #before proceeding, because not all sequence related information about MODEL
        #is changed by this command (e.g., internal coordinates, charges, and atom
        #types and radii are not updated).
        mdl1.write(file=TMP_FILE_PATH)
        mdl1.read(file=TMP_FILE_PATH)

        #set up restraints before computing energy
        #we do this a second time because the model has been written out and read in,
        #clearing the previously set restraints
        make_restraints(mdl1, ali)

        #a non-bonded pair has to have at least as many selected atoms
        mdl1.env.edat.nonbonded_sel_atoms=1

        sched = autosched.loop.make_for_model(mdl1)

        #only optimize the selected residue (in first pass, just atoms in selected
        #residue, in second pass, include nonbonded neighboring atoms)
        #set up the mutate residue selection segment
        s = Selection(mdl1.chains[chain].residues[respos])

        mdl1.restraints.unpick_all()
        mdl1.restraints.pick(s)

        s.energy()

        s.randomize_xyz(deviation=4.0)

        mdl1.env.edat.nonbonded_sel_atoms=2
        try:
            optimize(s, sched)
        except OverflowError as e: # failed once
            if n_attempts == 0:
                raise e
            print("Overflow error - trying again")
            n_attempts -= 1
            continue

        #feels environment (energy computed on pairs that have at least one member
        #in the selected)
        mdl1.env.edat.nonbonded_sel_atoms=1
        optimize(s, sched)

        s.energy()

        #give a proper name
        mdl1.write(file=OUT_FILE_PATH)

        #delete the temporary file
        os.remove(TMP_FILE_PATH)
        return OUT_FILE_PATH

def run_modeller_multiple(modelname, mutations, chain="A", out_fp=None, check_refmatch=True, **kwargs):
    """
    Runs Modeller on a PDB file multiple times to apply all specified mutations and output a final PDB file.

    Args:
        modelname (str): Path to the initial PDB model file.
        mutations (list of str): List of mutations to apply, formatted as strings (e.g., "A123V"), where
            - The first character is the reference amino acid.
            - The middle characters are the residue position (1-indexed).
            - The last character is the mutation amino acid.
        chain (str, optional): Chain identifier in the PDB file where mutations should be applied. Default is "A".
        out_fp (str, optional): File path for the output mutated PDB file. If None, a path is automatically generated 
            using the modelname and mutations list.
        check_refmatch (bool, optional): If True, verifies that the reference amino acid in the mutation string matches 
            the residue in the original PDB sequence at the specified position. Raises KeyError if there is a mismatch.
            Default is True.

    Returns:
        str: The file path of the final mutated PDB file.

    Raises:
        KeyError: If `check_refmatch` is True and the reference amino acid does not match the PDB sequence.

    Example:
        >>> run_modeller_multiple("input_model.pdb", ["A123V", "D145G"], chain="B")
        
        This example applies mutations A123V and D145G to chain B of "input_model.pdb" and outputs the final 
        mutated PDB file.
    """
    mutations.sort()
    out_fp = out_fp or f"{modelname.split('.pdb')[0]}_{'-'.join(mutations)}.pdb"
    native_seq = Chain(modelname).getSequence()
    
    with tqdm(mutations, ncols=100, total=len(mutations), desc="Applying mutations") as muts:
        for rpm in muts:
            muts.set_postfix(mut=rpm)
            
            ref, pos, mut = rpm[0], int(rpm[1:-1]), rpm[-1]
            
            if native_seq[pos-1] != ref and check_refmatch:
                raise KeyError(f"Reference AA {ref} at position {pos} doesnt "+ \
                                "match with what is defined in mutation {rpm}")
            
            run_modeller(modelname, respos=pos, 
                        restyp=ResInfo.code_to_pep[mut], 
                        chain=chain, 
                        out_fp=out_fp,
                        overwrite=True,
                        **kwargs)
            modelname = out_fp
    return out_fp