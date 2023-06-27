> These are general notes I have taken on how to dock a single protein-ligand pair using AutoDock Vina. This is not a complete guide and is not meant to be used as such. It is meant to be used as a reference for myself and others who are familiar with docking.
See [Vina Docs](https://autodock-vina.readthedocs.io/en/latest/docking_basic.html#preparing-the-receptor) if you want more comprehensive information.


## 3. Preparing receptor and ligand PDBQT files
The receptor must be cleaned of water molecules and other non-residue molecules. The receptor must also be converted to a pdbqt file that includes charge (Q) and atom type (T) information. This can be done using the `prepare_receptor4.py` script from ADT.

### **3.2**
The following are general instructions for preparing the receptor and ligand PDBQT files. This is useful if you have a PDB file that is not from PDBbind.
>#TODO: THIS IS NOT COMPLETE YET

### *3.2.A Receptor + ligand as PDB complex*
> For PDBbind we can download individual structures using wget as with `wget http://www.pdbbind.org.cn/v2007/10gs/10gs_complex.pdb` where `10gs` is the PDB ID

If the receptor and ligand already exist as a complex in a single PDB file we can automatically get the binding site information. For this, run `prep_pdbs.sh` with the following arguments. It will clean the files and split it into ligand (not usable) and receptor (usable) pqbqt files.

```bash
prep_pdbs.sh <path> <ADT_path>
```
>Make sure the PDB file has the following path format: `<path>/<pdbcode>/<pdbcode>.pdb`
>To get help message run `prep_pdb.sh` with no arguments.

The `<path>/<pdbcode>` directory should now contain the following files:
```bash
<pdbcode>.pdb
prep/<pdbcode>-split-<num_atoms>_ligand.pdbqt
prep/<pdbcode>-split-<num_atoms>_receptor.pdbqt
```

### *3.2.B Receptor on its own*
If the receptor is on its own in a PDB file we need to manually get the binding site information. We can still run `prep_pdbs.sh` but with `l` argument so that it only extracts the receptor.
```bash
prep_pdbs.sh <path> l <ADT_path>
```

### **3.3 Preparing Ligand PDBQT file**
For the ligand you need to download its SDF file and prepare it using OpenBabel or similar tools...
If you have the ligand name you can download it from PDB using the following address: `https://files.rcsb.org/ligands/download/{x}_ideal.sdf`

We can start from the SDF file and convert it to a PDBQT file using OpenBabel. To do so run the following:
```bash
obabel -isdf <path>/<ligand_name>_ideal.sdf -opdbqt -O <path>/<ligand_name>.pdbqt
```
or we can use the SMILES string to convert it to a PDBQT file using OpenBabel. To do so run the following:
```bash
obabel -:"<SMILES>" --gen3d -opdbqt -O <path>/<ligand_name>.pdbqt
```

### **3.4 Preparing Grid files**
For AutoDock Vina grid files and AutoGrid are not needed (see "AutoDock Tools Compatibility": https://vina.scripps.edu/manual/).

From Vina help message we can see how to input the search space:
```bash
Input:
  --receptor arg        rigid part of the receptor (PDBQT)
  --flex arg            flexible side chains, if any (PDBQT)
  --ligand arg          ligand (PDBQT)

Search space (required):
  --center_x arg        X coordinate of the center
  --center_y arg        Y coordinate of the center
  --center_z arg        Z coordinate of the center
  --size_x arg          size in the X dimension (Angstroms)
  --size_y arg          size in the Y dimension (Angstroms)
  --size_z arg          size in the Z dimension (Angstroms)

Output (optional):
  --out arg             output models (PDBQT), the default is chosen based on 
                        the ligand file name
  --log arg             optionally, write log file

Misc (optional):
  --cpu arg                 the number of CPUs to use (the default is to try to
                            detect the number of CPUs or, failing that, use 1)
  --seed arg                explicit random seed
  --exhaustiveness arg (=8) exhaustiveness of the global search (roughly 
                            proportional to time): 1+
  --num_modes arg (=9)      maximum number of binding modes to generate
  --energy_range arg (=3)   maximum energy difference between the best binding 
                            mode and the worst one displayed (kcal/mol)

Configuration file (optional):
  --config arg          the above options can be put here

Information (optional):
  --help                display usage summary
  --help_advanced       display usage summary with advanced options
  --version             display program version
```

This search space should be centered at the binding pocket and can be retrieved from the PDB file if provided as a complex. The size of the search space should be large enough to cover the entire binding pocket.

To create a config file with the search space provided, run the following:
```bash
python prep_conf.py -r <prep_path>
```
***
## 4. Running AutoDock Vina

Now that we have the prepared receptor, ligand, and `conf.txt` file set up, we can run AutoDock Vina.

To do so run the following:
```bash
vina --config <path>conf.txt
```

### **4.1 Running on PDBbind dataset**
To run on the PDBbind dataset we can use the `run_vina.sh` script with the following arguments:
```bash
run_vina.sh <path> [<shortlist>]
```
Where `<path>` is the path to the *PDBbind* dataset. Optionally we can also pass in `<shortlist>` the path to a csv file containing the PDB codes we want to run. This is useful if we want to only dock a subset of the PDBbind dataset.
