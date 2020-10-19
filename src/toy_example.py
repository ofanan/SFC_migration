import numpy as np
# from random import seed, randint

NUM_OF_SERVERS = 4
NUM_OF_PoA = 2
NUM_OF_USERS = 2
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

capacity_of_server = 2 * np.ones (NUM_OF_SERVERS)

NUM_OF_VNFs_OF_CHAIN = 2 * np.ones (NUM_OF_USERS)
NUM_OF_VNFs          = sum (NUM_OF_VNFs_OF_CHAIN)
theta = np.ones (NUM_OF_VNFs)
Lambda = np.ones (NUM_OF_VNFs)
cur_cpu_alloc_per_vnf = [2, 1, 1, 2]
cur_loc_of_vnf = [1, 2, 2, 1]
v_plus_of_vnf = 1

mig_cost = np.ones (NUM_OF_VNFs)
mig_data = 2 * np.ones (NUM_OF_VNFs)

print (capacity_of_server)