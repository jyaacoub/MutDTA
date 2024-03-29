# Vina Run Configurations
This is to keep track of the different runs of vina I have tried. See [vina_out](/results/PDBbind/vina_out/) directory for output files.

# 1 - Basic docking
## 1.1 prelim runs
### run1
first run of vina, due to errors in prep only 3015 PDBs were run out of 4650. No random seed defined.
```conf
energy_range = 3
exhaustiveness = 8
num_modes = 9
```
### run2 
Same as run1 but with random seed set to `904455071`.


## 1.2 mini runs
these are test runs for tuning params
### run3
```conf
energy_range = 5
exhaustiveness = 8
num_modes = 9
```
### run4
```conf
energy_range = 3 
exhaustiveness = 12
num_modes = 9
```
### run5
```conf
energy_range = 5
exhaustiveness = 12
num_modes = 9
```
### run6
```conf
energy_range = 5
exhaustiveness = 20
num_modes = 9
out = /cluster/projects/kumargroup/jean/data/vina_out/run6
log = /cluster/projects/kumargroup/jean/data/vina_out/run6
```

## 1.3 full runs
Back to running on entire dataset to get results
### run7
Same as 6 but for the entire kd_ki set + increase in energy range
```conf
energy_range = 10
exhaustiveness = 20
num_modes = 9
out = /cluster/projects/kumargroup/jean/data/vina_out/run7
log = /cluster/projects/kumargroup/jean/data/vina_out/run7
```

### run8
Same configuration as https://pubs.acs.org/doi/full/10.1021/acs.jcim.6b00740:
```conf
energy_range = 10
exhaustiveness = 50
num_modes = 20
out = /cluster/projects/kumargroup/jean/data/vina_out/run8
log = /cluster/projects/kumargroup/jean/data/vina_out/run8
```
run8 FAILED - vina_conf were not set up properly and default values were used instead of above

### run9
Increased search space by +10 for binding pocket
```conf
energy_range = 10
exhaustiveness = 50
num_modes = 20
out = /cluster/projects/kumargroup/jean/data/vina_out/run9
log = /cluster/projects/kumargroup/jean/data/vina_out/run9
```
retrying run8 conf with ex=50

### run10
Increased search space by +30 for binding pocket
```conf
energy_range = 20
exhaustiveness = 50
num_modes = 10
out = /cluster/projects/kumargroup/jean/data/vina_out/run9
log = /cluster/projects/kumargroup/jean/data/vina_out/run9
```
increased energy range and search space.


# 2 - Flexible docking
Following https://autodock-vina.readthedocs.io/en/latest/docking_flexible.html, the next runs will use flexible docking.

New parameters for flexible docking:
### run11
```conf
energy_range = 10
exhaustiveness = 20
num_modes = 9
out = /cluster/projects/kumargroup/jean/data/vina_out/run11
log = /cluster/projects/kumargroup/jean/data/vina_out/run11
```