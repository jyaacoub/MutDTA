## run1
first run of vina, due to errors in prep only 3015 PDBs were run out of 4650. No random seed defined.
```conf
energy_range = 3
exhaustiveness = 8
num_modes = 9
```
## run2 
Same as run1 but with random seed set to `904455071`.
***
# mini runs
these are test runs for tuning params
## run3
```conf
energy_range = 5
exhaustiveness = 8
num_modes = 9
```
## run4
```conf
energy_range = 3 
exhaustiveness = 12
num_modes = 9
```
## run5
```conf
energy_range = 5
exhaustiveness = 12
num_modes = 9
```
## run6
```conf
energy_range = 5
exhaustiveness = 20
num_modes = 9
out = /cluster/projects/kumargroup/jean/data/vina_out/run6
log = /cluster/projects/kumargroup/jean/data/vina_out/run6
```
## run7
Same as 6 but for the entire kd_ki set + increase in energy range
```conf
energy_range = 10
exhaustiveness = 20
num_modes = 9
out = /cluster/projects/kumargroup/jean/data/vina_out/run7
log = /cluster/projects/kumargroup/jean/data/vina_out/run7
```

## run8
Same configuration as https://pubs.acs.org/doi/full/10.1021/acs.jcim.6b00740:
```conf
energy_range = 10
exhaustiveness = 50
num_modes = 20
out = /cluster/projects/kumargroup/jean/data/vina_out/run8
log = /cluster/projects/kumargroup/jean/data/vina_out/run8
```
