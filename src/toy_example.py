import numpy as np
# from random import seed, randint

class Toy_example (object):
    
    def __init__ (self):
        self.NUM_OF_SERVERS = 4
        self.NUM_OF_PoA = 2
        self.NUM_OF_USERS = 2
        self.NUM_OF_CHAINS = self.NUM_OF_USERS
        self.servers_path_delay = np.random.rand (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS) #servers_path_delay[i][j] holds the netw' delay of the path from server i to server j
        self.servers_path_cost  = np.random.rand (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS)
        
        for s in range (self.NUM_OF_SERVERS):
            self.servers_path_delay [s][s] = 0 # Delay from server to itself is 0
        
        # The Points of Accs will make the first rows in the path_costs matrix
        self.PoA_of_user     = np.random.randint(self.NUM_OF_PoA, size = self.NUM_OF_USERS)
        self.server_to_user_delay = np.zeros ((self.NUM_OF_SERVERS, self.NUM_OF_USERS))
        for s in range (self.NUM_OF_SERVERS):
            for u in range (self.NUM_OF_USERS):
                    self.server_to_user_delay[s][u] = self.servers_path_delay [s][self.PoA_of_user[u]]
        
        # print (servers_path_delay)
        # print (PoA_of_user)
        # print (server_to_user_delay)
        
        self.capacity_of_server = 2 * np.ones (self.NUM_OF_SERVERS, dtype='uint8')
        
        self.num_of_vnfs_in_chain = 2 * np.ones (self.NUM_OF_USERS, dtype='uint8')
        self.NUM_OF_VNFs          = sum (self.num_of_vnfs_in_chain).astype ('uint')
        self.theta = np.ones (self.NUM_OF_VNFs) #cpu units to process one unit of data
        self.Lambda = np.ones (self.NUM_OF_VNFs)
        self.cur_cpu_alloc_per_vnf = [2, 1, 1, 2]
        self.cur_loc_of_vnf = [1, 2, 2, 1]
        self.target_delay # the desired (max) delay (aka Delta)

        
        # Calculate v^+ of each VNF v.
        # v_plus_of_vnf(v) will hold the idx of the next VNF in the same chain.
        # if v is the last VNF in its chain, then v_plus_of_vnf(v) will hold the PoA of this chain's user  
        self.v_plus_of_vnf = np.zeros (self.NUM_OF_VNFs, dtype = 'uint')
        v = 0
        for chain in range (self.NUM_OF_CHAINS):
            for idx_in_chain in range (self.num_of_vnfs_in_chain[chain]):
                self.v_plus_of_vnf [v] = v+1 if (idx_in_chain < self.num_of_vnfs_in_chain[chain]-1) else self.PoA_of_user[chain]
                v += 1
        
        print ('PoA of user = ', self.PoA_of_user)
        print (self.v_plus_of_vnf)
        self.mig_comp_cost = np.ones (self.NUM_OF_VNFs)
        self.mig_data = 2 * np.ones (self.NUM_OF_VNFs)
        self.mig_bw   = 1 * np.ones (self.NUM_OF_VNFs)
        self.Lambda   = 1 * np.ones (self.NUM_OF_VNFs) # self.Lambda[v] will hold the input BW of v.
        self.mig_comp_cost # self.mig_comp_cost[v] hold the migration's computational cost of VM v
        self.mig_data # self.mig_data[v] amount of data units to transfer during the migration of VM v
        self.servers_path_cost # self.servers_path_cost [i][j] is the cost of transmitting 1 unit of data from i to j
    
    def perf_degradation (self, loc_v, loc_vpp, cpu_alloc):
        """
        Calculate the performance degradation of a VM. 
        Inputs
        - loc_v - location of v, 
        - loc_vpp - location of v^+ (the next VM after v in the service chain)
        - cpu_alloc - (suggested) cpu allocation of v
        
        """
        return ( 1 /(cpu_alloc - self.theta_v * self.lambda_v) + self.servers_path_delay[loc_v][loc_vpp] ) / self.target_delay
            
    
    def mig_cost (self, src, dst):
        """
        Calculate the cost of migration a VM from src to dst
        """
        return 42
    
    
        
    
