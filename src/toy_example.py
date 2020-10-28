import numpy as np
# from random import seed, randint

class toy_example (object):
    
    def __init__ (self):
        self.NUM_OF_SERVERS = 2
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
        
        self.servers_path_cost # self.servers_path_cost [i][j] is the cost of transmitting 1 unit of data from i to j

        self.uniform_cpu_capacity   = 2
        self.cpu_capacity_of_server = self.uniform_cpu_capacity * np.ones (self.NUM_OF_SERVERS, dtype='uint8')
        
        self.num_of_vnfs_in_chain   = 2 * np.ones (self.NUM_OF_USERS, dtype='uint8')
        self.NUM_OF_VNFs            = sum (self.num_of_vnfs_in_chain).astype ('uint')
        self.theta                  = np.ones (self.NUM_OF_VNFs) #cpu units to process one unit of data
        self.Lambda                 = np.ones (self.NUM_OF_VNFs) #Lambda[v] is the input bw of VM v
        self.cur_cpu_alloc_of_vnf   = np.array([2, 1, 1, 2])
        self.nxt_cpu_alloc_of_vnf   = np.array([2, 1, 1, 2])
        self.cur_loc_of_vnf         = [1, 2, 2, 1] #cur_loc_of_vnf[v] will hold the id of the server currently hosting VNF v
        self.nxt_loc_of_vnf         = [1, 2, 2, 1] #cur_loc_of_vnf[v] will hold the id of the server planned to host VNF v
        self.target_delay           = np.ones (self.NUM_OF_VNFs) # the desired (max) delay (aka Delta)
        self.perf_deg_of_VNF        = np.zeros (self.NUM_OF_VNFs)
        
        # Calculate v^+ of each VNF v.
        # v_plus_of_vnf(v) will hold the idx of the next VNF in the same chain.
        # if v is the last VNF in its chain, then v_plus_of_vnf(v) will hold the PoA of this chain's user  
        self.v_plus_of_vnf = np.zeros (self.NUM_OF_VNFs, dtype = 'uint')
        v = 0
        for chain in range (self.NUM_OF_CHAINS):
            for idx_in_chain in range (self.num_of_vnfs_in_chain[chain]):
                self.v_plus_of_vnf [v] = v+1 if (idx_in_chain < self.num_of_vnfs_in_chain[chain]-1) else self.PoA_of_user[chain]
                v += 1
        
        self.mig_comp_cost  = np.ones (self.NUM_OF_VNFs)     # self.mig_comp_cost[v] hold the migration's computational cost of VM v
        self.mig_data       = 2 * np.ones (self.NUM_OF_VNFs) # self.mig_data[v] amount of data units to transfer during the migration of VM v
        self.mig_bw         = 1 * np.ones (self.NUM_OF_VNFs)
        self.Lambda         = 1 * np.ones (self.NUM_OF_VNFs) # self.Lambda[v] will hold the input BW of v.
           
        self.paths_of_link = [ 
                                [
                                    [],               # list of paths in which l(0,0) appears 
                                    [ [0,1], [0,2] ], # list of paths in which l(0,1) appears
                                    []                # list of paths in which l(0,2) appears
                                ],
                                [
                                    [ [1,0], [2,0] ], # list of paths in which l(1,0) appears 
                                    [],               # list of paths in which l(1,1) appears
                                    [ [0,2], [1,2]]                # list of paths in which l(1,2) appears
                                ],
                                [
                                    [],               # list of paths in which l(2,0) appears 
                                    [ [2,1], [2,0] ], # list of paths in which l(2,1) appears
                                    []                # list of paths in which l(2,2) appears
                                ]
                            ]
        self.NUM_OF_LINKS = 4
        self.capacity_of_link = np.zeros ( (self.NUM_OF_SERVERS+1, self.NUM_OF_SERVERS+1), dtype='uint8')
        self.uniform_link_capacity   = 3
        self.capacity_of_link[0][1] = self.uniform_link_capacity
        self.capacity_of_link[1][0] = self.uniform_link_capacity
        self.capacity_of_link[1][2] = self.uniform_link_capacity
        self.capacity_of_link[2][1] = self.uniform_link_capacity
        
        
    def perf_degradation (self, loc_v, loc_vpp, denominator):
        """
        Calculate the performance degradation of a VM. 
        Inputs
        - loc_v - location of v, 
        - loc_vpp - location of v^+ (the next VM after v in the service chain)
        - cpu_alloc - (suggested) cpu allocation of v
        
        """
        return ( 1 / denominator + self.servers_path_delay[loc_v][loc_vpp] ) / self.target_delay
            
    
    def mig_cost (self, src, dst):
        """
        Calculate the cost of migration a VM from src to dst
        """
        return 42
    
    def is_feasible(self):
        
        # Max CPU capacity of server constraint
        available_cpu_at_server = self.cpu_capacity_of_server
        for v in range (self.NUM_OF_VNFs):
            available_cpu_at_server[self.nxt_loc_of_vnf[v]] -= self.nxt_cpu_alloc_of_vnf[v]
            if (available_cpu_at_server[self.nxt_loc_of_vnf[v]] < 0):
                return False # allocated more cpu than s's capacity

        # Finite computation delay constraint and max perf' degradation constraint 
        for v in range (self.NUM_OF_VNFs):    
            denominator = self.nxt_cpu_alloc_of_vnf[v] - self.theta[v] * self.Lambda[v] 
            if (denominator <= 0):
                return False
            self.perf_degradation_of_VNF[v] = self.perf_degradation (
                self.nxt_loc_of_vnf[v], self.nxt_loc_of_vnf[self.v_plus_of_vnf[v]], denominator)
            if (self.perf_degradation_of_VNF[v] > 1):
                return False
            
        # Total link capacity constraint
        for s in range (self.NUM_OF_LINKS):
            for d in range (self.NUM_OF_LINKS):
                available_bw = self.capacity_of_link[s][d]
                if (available_bw == 0): 
                    continue # There's no physical link (s,d)
                for pair in self.paths_of_link [s][d]:
                    print (pair)
         
        
        
        return True
    

    def inc_array (self, ar, base):
        for idx in range (ar.size-1, -1, -1):
            if (ar[idx] < base - 1):
                ar[idx] += 1
                return ar
            ar[idx] = 0
    
    
    def inc_nxt_loc_of_vnf (self):
        for idx in range (self.NUM_OF_VNFs-1, -1, -1):
            if (self.nxt_loc_of_vnf[idx] < self.NUM_OF_SERVERS-1):
                self.nxt_loc_of_vnf[idx] += 1
                return
            self.nxt_loc_of_vnf[idx] = 0
    
    def brute_force (self):    
        self.nxt_loc_of_vnf = np.zeros(self.NUM_OF_VNFs, dtype = 'uint8')
        for __ in range (pow (self.NUM_OF_SERVERS, self.NUM_OF_VNFs)):
#             print (self.nxt_loc_of_vnf)
#             print ('*************************************')
            self.nxt_loc_of_vnf = self.inc_array(self.nxt_loc_of_vnf, self.NUM_OF_SERVERS)
            self.nxt_cpu_alloc_of_vnf = np.zeros(self.NUM_OF_VNFs, dtype = 'uint8')
            for __ in range (pow (self.NUM_OF_SERVERS, self.NUM_OF_VNFs)):
#                 print (self.nxt_cpu_alloc_of_vnf)
                self.nxt_cpu_alloc_of_vnf = self.inc_array(self.nxt_cpu_alloc_of_vnf, self.uniform_cpu_capacity)
                
                self.is_feasible ()

#         for __ in range (pow (self.NUM_OF_SERVERS, self.NUM_OF_VNFs)):
#             print (self.nxt_loc_of_vnf)
#             self.nxt_cpu_alloc_of_vnf = np.zeros(self.NUM_OF_VNFs, dtype = 'uint8')
#             self.inc_nxt_loc_of_vnf()
                        
#             if (idx > 0):
#                 self.nxt_loc_of_vnf[idx-1] += 1
#             for idx in range (self.NUM_OF_VNFs, -1, 1):
#                 self.nxt_loc_of_vnf[idx] = (self.nxt_loc_of_vnf[idx] + 1) if (self.nxt_loc_of_vnf[idx] < self.NUM_OF_SERVERS-1) else 0  
#         for all permutations 
#             for all possible cpu allocations
#                 check feasibility
    
