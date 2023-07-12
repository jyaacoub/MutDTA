## from compute node to PC when working on seperate pc
1. copy files to login node (compute node doesn't have internet)
2. ssh -A h4h_login
3. scp -r in_file user@<ipaddr>:out_file
4. mv to wsl folder (if using wsl)

## when working at PC - to scp from cluster to pc:
scp -r -oProxyJump=<cluster> <cluster_compute>:<from_remote_path> <to_local_path>