import numpy as np
# from random import seed, randint

NUM_OF_SERVERS = 4
NUM_OF_PoA = 2
NUM_OF_USERS = 2
NUM_OF_CHAINS = NUM_OF_USERS
servers_path_delay = np.random.rand (NUM_OF_SERVERS, NUM_OF_SERVERS)
for s in range (NUM_OF_SERVERS):
    servers_path_delay [s][s] = 0 # Delay from server to itself is 0

# The Points of Accs will make the first rows in the path_costs matrix
PoA_of_user     = np.random.randint(NUM_OF_PoA, size = NUM_OF_USERS)
server_to_user_delay = np.zeros ((NUM_OF_SERVERS, NUM_OF_USERS))
for s in range (NUM_OF_SERVERS):
    for u in range (NUM_OF_USERS):
            server_to_user_delay[s][u] = servers_path_delay [s][PoA_of_user[u]]

# print (servers_path_delay)
# print (PoA_of_user)
# print (server_to_user_delay)

capacity_of_server = 2 * np.ones (NUM_OF_SERVERS, dtype='uint8')

num_of_vnfs_in_chain = 2 * np.ones (NUM_OF_USERS, dtype='uint8')
NUM_OF_VNFs          = sum (num_of_vnfs_in_chain).astype ('uint')
theta = np.ones (NUM_OF_VNFs)
Lambda = np.ones (NUM_OF_VNFs)
cur_cpu_alloc_per_vnf = [2, 1, 1, 2]
cur_loc_of_vnf = [1, 2, 2, 1]

# Calculate v^+ of each VNF v.
# v_plus_of_vnf(v) will hold the idx of the next VNF in the same chain.
# if v is the last VNF in its chain, then v_plus_of_vnf(v) will hold the PoA of this chain's user  
v_plus_of_vnf = np.zeros (NUM_OF_VNFs, dtype = 'uint')
v = 0
for chain in range (NUM_OF_CHAINS):
    for idx_in_chain in range (num_of_vnfs_in_chain[chain]):
        v_plus_of_vnf [v] = v+1 if (idx_in_chain < num_of_vnfs_in_chain[chain]-1) else PoA_of_user[chain]
        v += 1

print ('PoA of user = ', PoA_of_user)
print (v_plus_of_vnf)
# mig_cost = np.ones (NUM_OF_VNFs)
# mig_data = 2 * np.ones (NUM_OF_VNFs)
# 
# print (capacity_of_server)

