# steps to transfer files from compute node to PC
1. copy files to login node
2. ssh -A h4h_login
3. scp -r in_file user@<ipaddr>:out_file
4. mv to wsl folder (if using wsl)