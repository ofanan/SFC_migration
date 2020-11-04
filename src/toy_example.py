import networkx as nx
import numpy as np

from printf import printf
# from random import seed, randint

class toy_example (object):
    
    def gen_tree (self):
        """
        Create a new random edge and delete one of its current edge if del_orig is True.
        param graph: networkx graph
        param del_orig: bool
        return: networkx graph
        """
        self.G = nx.generators.classic.balanced_tree (r=2, h=2) # Generate a tree of height h where each node has r children.
#         print (self.G.nodes())
#         print (self.G.edges)
        sp = nx.shortest_path(self.G)
#         print (sp[0][0])
          

    def custom_three_nodes_tree (self):
        self.NUM_OF_SERVERS = 3
        self.NUM_OF_USERS = 2
        self.NUM_OF_PoA = 2
        self.NUM_OF_CHAINS = self.NUM_OF_USERS
    
        #servers_path_delay[i][j] holds the netw' delay of the path from server i to server j
        #         self.servers_path_delay        = np.array ((self.NUM_OF_SERVERS, self.NUM_OF_SERVERS))
        #         self.servers_path_delay =  np.random.rand (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS) 
        self.servers_path_delay  = [
                                   [0, 1, 2],
                                   [1, 0, 1],
                                   [2, 1, 0]
                                  ]
    
        # for s in range (self.NUM_OF_SERVERS):
        # self.servers_path_delay [s][s] = 0 # Delay from server to itself is 0
        
        # self.servers_path_cost = self.servers_path_delay # # self.servers_path_cost [i][j] is the cost of transmitting 1 unit of data from i to j#np.random.rand (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS) #servers_path_delay[i][j] holds the netw' delay of the path from server i to server j. Currently unused
        self.links_of_path = [ 
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
        self.capacity_of_link = np.zeros ( (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS))
        self.uniform_link_capacity   = 3
        self.capacity_of_link[0][1] = self.uniform_link_capacity
        self.capacity_of_link[1][0] = self.uniform_link_capacity
        self.capacity_of_link[1][2] = self.uniform_link_capacity
        self.capacity_of_link[2][1] = self.uniform_link_capacity
        self.cur_loc_of_vnf         = [2, 2, 0, 0]                  # cur_loc_of_vnf[v] will hold the id of the server currently hosting VNF v
        self.cur_cpu_alloc_of_vnf   = np.array([2, 2, 2, 2])

    def __init__ (self):
        
        use_custom_netw = True
        if (use_custom_netw == True):
            self.custom_three_nodes_tree()
        
        # The Points of Accs will make the first rows in the path_costs matrix
        self.PoA_of_user     = np.random.randint(self.NUM_OF_PoA, size = self.NUM_OF_USERS) #[0,0]
#         self.server_to_user_delay = np.zeros ((self.NUM_OF_SERVERS, self.NUM_OF_USERS))
#         for s in range (self.NUM_OF_SERVERS):
#             for u in range (self.NUM_OF_USERS):
#                     self.server_to_user_delay[s][u] = self.servers_path_delay [s][self.PoA_of_user[u]]
        
        self.uniform_cpu_capacity   = 4
        self.cpu_capacity_of_server = self.uniform_cpu_capacity * np.ones (self.NUM_OF_SERVERS, dtype='int8')
        
        self.num_of_vnfs_in_chain   = 2 * np.ones (self.NUM_OF_USERS, dtype='uint8')
        self.NUM_OF_VNFs            = sum (self.num_of_vnfs_in_chain).astype ('uint')
        self.theta                  = 0.7 * np.ones (self.NUM_OF_VNFs) #cpu units to process one unit of data
        self.traffic_in             = np.ones (self.NUM_OF_VNFs) #traffic_in[v] is the bw of v's input traffic 
        #self.traffic_out            = np.ones (self.NUM_OF_VNFs) #traffic_in[v] is the bw of v's input traffic 
        self.nxt_loc_of_vnf         = np.array (self.NUM_OF_VNFs)   # nxt_loc_of_vnf[v] will hold the id of the server planned to host VNF v
        self.nxt_cpu_alloc_of_vnf   = np.array (self.NUM_OF_VNFs)
        self.target_delay           = 20 * np.ones (self.NUM_OF_VNFs)    # the desired (max) delay (aka Delta)
        self.perf_deg_of_VNF        = np.zeros (self.NUM_OF_VNFs)
        
        # Calculate v^+ of each VNF v.
        # vpp(v) will hold the idx of the next VNF in the same chain.
        # if v is the last VNF in its chain, then vpp(v) will hold the PoA of this chain's user  
        self.vpp = np.zeros (self.NUM_OF_VNFs, dtype = 'uint')
        v = 0
        for chain in range (self.NUM_OF_CHAINS):
            for idx_in_chain in range (self.num_of_vnfs_in_chain[chain]):
                self.vpp [v] = v+1 if (idx_in_chain < self.num_of_vnfs_in_chain[chain]-1) else self.PoA_of_user[chain]
                v += 1
       
#         self.mig_comp_cost  = np.ones (self.NUM_OF_VNFs)     # self.mig_comp_cost[v] hold the migration's computational cost of VM v. Currently unused.
#         self.mig_data       = 0 * np.ones (self.NUM_OF_VNFs) # self.mig_data[v] amount of data units to transfer during the migration of VM v. Currently unused.
        self.mig_bw         = 1.1 * np.ones (self.NUM_OF_VNFs)
           
        

        self.min_cost = float ('inf')
        self.best_nxt_cpu_alloc_of_vnf = np.array (self.NUM_OF_VNFs)
        self.best_nxt_loc_of_vnf       = np.array (self.NUM_OF_VNFs)   # nxt_loc_of_vnf[v] will hold the id of the server planned to host VNF v
        self.output_file = open ("../res.txt", "a")
        

    def perf_degradation (self, v, loc_v, loc_vpp, denominator):
        """
        Calculate the performance degradation of a VM. 
        Inputs
        - loc_v - location of v, 
        - loc_vpp - location of v^+ (the next VM after v in the service chain)
        - cpu_alloc - (suggested) cpu allocation of v
        
        """
        return ( 1 / denominator + self.servers_path_delay[loc_v][loc_vpp] ) / self.target_delay[v]
            
    
    def mig_cost (self, src = 0, dst = 0):
        """
        Calculate the cost of migration a VM from src to dst
        """
        return 0.35
    
    def is_feasible(self):
        
        # Max CPU capacity of server constraint
        available_cpu_at_server = self.cpu_capacity_of_server.copy () 
        for v in range (self.NUM_OF_VNFs):
            available_cpu_at_server[self.nxt_loc_of_vnf[v]] -= self.nxt_cpu_alloc_of_vnf[v]
            if (available_cpu_at_server[self.nxt_loc_of_vnf[v]] < 0):
                return False # allocated more cpu than s's capacity

        # Finite computation delay constraint and max perf' degradation constraint 
        for v in range (self.NUM_OF_VNFs):               
            denominator = self.nxt_cpu_alloc_of_vnf[v] - self.theta[v] * self.traffic_in[v] 
            if (denominator <= 0):
                return False
            self.perf_deg_of_VNF[v] = self.perf_degradation (
                v, self.nxt_loc_of_vnf[v], self.nxt_loc_of_vnf[self.vpp[v]], denominator)
            if (self.perf_deg_of_VNF[v] > 1):
                return False
            
        # Total link capacity constraint
        available_bw = self.capacity_of_link.copy()
        for v in range (self.NUM_OF_VNFs): # for each VNF v
            vpp = self.vpp[v] # vpp is the next VM in the chain
            for link in self.links_of_path [self.nxt_loc_of_vnf[v]] [self.nxt_loc_of_vnf[vpp]]: #for each link on the path from v's scheduled location to vpp's scheduled location
                available_bw[link[0]][link[1]] -= self.traffic_in[vpp]  
                if (available_bw[link[0]][link[1]] < 0):
                    return False
            for link in self.links_of_path [self.cur_loc_of_vnf[v]] [self.nxt_loc_of_vnf[v]]: # for each link on the path from v's cur location to v's new location (this path exists in practice only if v is scheduled to migrate) 
                available_bw[link[0]][link[1]] -= self.mig_bw[v]
                if (available_bw[link[0]][link[1]] < 0):
                    return False
                
        
        return True
    

    def inc_array (self, ar, min_val, max_val):
        for idx in range (ar.size-1, -1, -1):
            if (ar[idx] < max_val):
                ar[idx] += 1
                return ar
            ar[idx] = min_val
        return ar
    
    def cost (self):
        cost = 0.0
        if (not(self.cur_loc_of_vnf == self.nxt_loc_of_vnf).all()):
            cost += self.mig_cost ()
        for v in range (self.NUM_OF_VNFs):
            denominator = self.nxt_cpu_alloc_of_vnf[v] - self.theta[v] * self.traffic_in[v] 
            cost += self.perf_degradation (
                v, self.nxt_loc_of_vnf[v], self.nxt_loc_of_vnf[self.vpp[v]], denominator)
        if (cost < self.min_cost):
            self.min_cost = cost
            self.best_nxt_loc_of_vnf        = self.nxt_loc_of_vnf.copy () 
            self.best_nxt_cpu_alloc_of_vnf  = self.nxt_cpu_alloc_of_vnf.copy ()
        return cost

    def calc_beta (self, nxt_loc_of_vnf, nxt_cpu_alloc_of_vnf):
        """
        Convert the decision variable vectors used in the code to those in the paper (Y, \beta)
        """
        self.beta = np.zeros ((self.NUM_OF_VNFs, self.NUM_OF_SERVERS))
        for v in range (self.NUM_OF_VNFs):
            self.beta[v][nxt_loc_of_vnf[v]] = int (nxt_cpu_alloc_of_vnf[v])
        
    
    def brute_force (self):    
        self.nxt_loc_of_vnf = np.zeros(self.NUM_OF_VNFs, dtype = 'uint8')

        for i in range (pow (self.NUM_OF_SERVERS, self.NUM_OF_VNFs)): #$$$$ TBD: change to while, stopping at last relevant val
            
            self.nxt_cpu_alloc_of_vnf = 2 * np.ones(self.NUM_OF_VNFs, dtype = 'uint8')

            for j in range (pow (self.uniform_cpu_capacity - 1, self.NUM_OF_VNFs)): #$$$$ TBD: change to while, stopping at last relevant val
                
                if (self.is_feasible()):
                    self.calc_beta (self.nxt_loc_of_vnf, self.nxt_cpu_alloc_of_vnf)
                    printf (self.output_file, 'beta = \n{}\nCost = {:.4}\n\n' .format (self.beta, self.cost()))
                
                self.nxt_cpu_alloc_of_vnf = self.inc_array (self.nxt_cpu_alloc_of_vnf, 2, self.uniform_cpu_capacity)
            self.nxt_loc_of_vnf = self.inc_array(self.nxt_loc_of_vnf, 0, self.NUM_OF_SERVERS-1)
        self.calc_beta (self.best_nxt_loc_of_vnf, self.best_nxt_cpu_alloc_of_vnf)
        printf (self.output_file, '\nBest solution is:\nbeta = \n{}\nCost = {:.4}\n\n' .format (self.beta, self.min_cost))

