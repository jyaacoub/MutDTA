#!/bin/bash

# the following causes an error ("AttributeError: member babel_type not found")
# full stack trace:
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7//MGLToolsPckgs/AutoDockTools/Utilities24//prepare_receptor4.py", line 204, in <module>
#     dict=dictionary)    
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/MoleculePreparation.py", line 558, in __init__
#     version=version, delete_single_nonstd_residues=delete_single_nonstd_residues)
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/MoleculePreparation.py", line 141, in __init__
#     self.addCharges(mol, charges_to_add)
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/MoleculePreparation.py", line 227, in addCharges
#     chargeCalculator.addCharges(mol.allAtoms)
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/MolKit/chargeCalculator.py", line 80, in addCharges
#     babel.assignHybridization(atoms)
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/PyBabel/atomTypes.py", line 136, in assignHybridization
#     self.valence_three()
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/PyBabel/atomTypes.py", line 236, in valence_three
#     elif self.count_free_ox(a) >= 2: a.babel_type="Cac"
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/PyBabel/atomTypes.py", line 167, in count_free_ox
#     self.count_heavy_atoms(bonded_atom) == 1:
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/PyBabel/atomTypes.py", line 157, in count_heavy_atoms
#     if bonded_atom.babel_type[0] == 'H': count = count + 1
#   File "/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/MolKit/molecule.py", line 409, in __getattr__
#     raise AttributeError('member %s not found'%member)
# AttributeError: member babel_type not found

ADT_path=/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/
t=/home/jyaacoub/mgltools_x86_64Linux2_1.5.7/
protein=/home/jyaacoub/projects/MutDTA/data/PDBbind/raw/refined-set/3ao2/3ao2_protein.pdb

"${t}/bin/pythonsh" "${ADT_path}/prepare_receptor4.py" -r $protein -o "./3ao2.pdbqt"  -A checkhydrogens -U nphs_lps_waters_nonstdres