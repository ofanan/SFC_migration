import numpy as np
import networkx as nx
import math

from printf import printf

class toy_example (object):
    
    def custom_three_nodes_tree (self):
        self.NUM_OF_SERVERS = 3
        self.NUM_OF_USERS = 2
        self.NUM_OF_PoA = 2
    
        #servers_path_delay[i][j] holds the netw' delay of the path from server i to server j
        self.servers_path_delay  = [
                                   [0, 1, 2],
                                   [1, 0, 1],
                                   [2, 1, 0]
                                  ]
    
        
        # self.links_of_path[s][d] will contain the list of links found in the path from s to d
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
        self.NUM_OF_CHAINS = self.NUM_OF_USERS
       
        # The Points of Accs will make the first rows in the path_costs matrix
        self.PoA_of_user     = np.random.randint(self.NUM_OF_PoA, size = self.NUM_OF_USERS) # PoA_of_user[u] will hold the PoA of the user using chain u       
        self.output_file            = open ("../res/custom_tree.res", "a")
        self.LP_output_file         = open ("../res/custom_tree.LP", "a")

    def gen_tree (self):
        """
        Create a new random edge and delete one of its current edge if del_orig is True.
        param graph: networkx graph
        param del_orig: bool
        return: networkx graph
        """
        self.G = nx.generators.classic.balanced_tree (r=3, h=2) # Generate a tree of height h where each node has r children.
        
        self.NUM_OF_SERVERS = self.G.number_of_nodes()
        self.NUM_OF_USERS   = 2
        self.NUM_OF_PoA     = 2

        self.servers_path_delay = np.array ((self.NUM_OF_SERVERS, self.NUM_OF_SERVERS)) # $$$ TBD: fix links' delay, and calc servers_path_delay accordingly
        self.servers_path_delay = np.random.rand (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS) 

        shortest_path = nx.shortest_path(self.G)

        # self.links_of_path[s][d] will contain the list of links found in the path from s to d
        self.links_of_path = []
        for s in range (self.NUM_OF_SERVERS):
            row_in_links_of_path = []
            for d in range (self.NUM_OF_SERVERS):
                if (s == d):
                    row_in_links_of_path.append([]) # no links in the path 
                    continue
                path_from_s_to_d = []
                for link_src_node in range (len(shortest_path[s][d])-1):
                    path_from_s_to_d.append ([shortest_path[s][d][link_src_node], shortest_path[s][d][link_src_node+1]])
                row_in_links_of_path.append (path_from_s_to_d)

            self.links_of_path.append (row_in_links_of_path)
        
        self.NUM_OF_LINKS = self.G.number_of_edges()
        self.capacity_of_link = np.zeros ( (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS))
        
        # Generate from each undiretional edge in the tree 2 directional edges. Assign all links capacity = self.uniform_link_capacity
        self.uniform_link_capacity   = 3
        for edge in self.G.edges:
            self.capacity_of_link[edge[0]][edge[1]] = self.uniform_link_capacity
            self.capacity_of_link[edge[1]][edge[0]] = self.uniform_link_capacity

        self.NUM_OF_CHAINS = self.NUM_OF_USERS
       
        # The Points of Accs will make the first rows in the path_costs matrix
        self.PoA_of_user     = np.random.randint(self.NUM_OF_PoA, size = self.NUM_OF_USERS) #[0,0]
        
        self.cur_loc_of_vnf         = np.random.randint(self.NUM_OF_SERVERS, size = self.NUM_OF_VNFs) # Initially, allocate VMs on random VMs
        self.cur_cpu_alloc_of_vnf   = 2 * np.ones (self.NUM_OF_VNFs)                                  # Initially, allocate each VNs uniform amount CPU units
        self.output_file            = open ("../larger_tree.res.txt", "a")
                
    def print_cpu_cap_const (self):
        """
        Print the constraint of maximum server's CPU capacity in a LP format
        """
        
        for s in range (self.NUM_OF_SERVERS):
            is_first_in_list = True
            for item in list (filter (lambda item: item['s'] == s, self.n )): # for each decision var' related to server s
                if (is_first_in_list):
                    printf (self.LP_output_file, 'subject to max_cpu_C{}: {}*X{} ' .format (self.const_num, item['a'], item['id']))
                    self.const_num += 1
                    is_first_in_list = False
                else: 
                    printf (self.LP_output_file, '+ {}*X{} ' .format (self.const_num, item['a'], item['id']))
            printf (self.LP_output_file, ' <= <= {}\n' .format (self.cpu_capacity_of_server[s]))

    def print_single_alloc_const (self):
        """
        Print the constraint of a single allocation for each VM in a LP format
        """
        v = -1 
        for item in self.n:
            if (item['v'] == v): #Already seen decision var' related to this VM
                printf (self.LP_output_file, '+ X{}' .format (item['id']))
            else: # First time observing decision var' related to this VM
                if (v > -1):
                    printf (self.LP_output_file, ' = 1;\n' )
                printf (self.LP_output_file, 'subject to single_alloc_C{}:   X{} ' .format (self.const_num, item['id'])) 
                v = item['v']
            self.const_num += 1
        printf (self.LP_output_file, ' = 1;\n\n' )
               
    def print_vars (self):
        """
        Print the var' ranges constraints  >=0  
        """
        for __ in self.n:
            printf (self.LP_output_file, 'var X{} >= 0;\n' .format (__['id'], __['id']) )
        printf (self.LP_output_file, '\n')

    def print_vars_leq1_const (self):
        """
        Print the var' ranges constraints  >=0  
        """
        printf (self.LP_output_file, '\n') 
        for __ in self.n:
            printf (self.LP_output_file, 'subject to X_leq1_C{}: X{} <= 1;\n' .format (self.const_num, __['id'], __['id']) )
            self.const_num += 1
        printf (self.LP_output_file, '\n')

    
    def print_obj_function (self):
        """
        Print the objective function in a standard LP form (linear combination of the decision variables)
        """
        printf (self.LP_output_file, 'minimize z:   ')
        is_first_item = True
        for item in self.n:
            if (not (is_first_item)):
                printf (self.LP_output_file, ' + ')
            printf (self.LP_output_file, '{:.4f}*X{}' .format (item['cost'], item ['id']) ) 
            is_first_item = False
        printf (self.LP_output_file, ';\n')
    
    def gen_c (self):
        self.c = self.n.copy ()
        for item in self.c:
            v = item['v']
            if (self.verbose == 1):
                print ('v = {}, s = {}, a = {}' .format (v, item['s'], item['a']))
            item['value'] = 1  if (item['s'] == self.cur_loc_of_vnf[v] and item['a'] == self.cur_cpu_alloc_of_vnf[v]) else 0
            
    def calc_cost_by_n (self):
        """
        Translate the parameters "self.nxt_loc_of_vnf" and "self.nxt_alloc_of_vnf" to a list of set entries in the "n" decision variable
        The list of set items is written to self.n_on
        Returns the cost of this assignment, calculated by n_on
        """ 
        self.n_on = []
        cost = 0
        for v in range (self.NUM_OF_VNFs):
            s = self.nxt_loc_of_vnf[v]
            a = self.nxt_cpu_alloc_of_vnf[v]
            self.n_on.append ({'v' : v, 's' : s, 'a' : a})
            cost += sum (item['cost'] for item in self.n if (item['v'] == v and item['s'] == s and item['a'] == a))
        return cost
            
              
    def gen_n (self):
        """"
        Calculate the cost of a feasible solution, given by its nxt_loc_of_vnf (y) and nxt_cpu_alloc (\beta_{vs}), and assuming that the perf' deg' was already calculated.
        Note that it's only for a feasible sol', as the calculation uses the array self.perf_deg_of_vnf, that should have been calculated by is_feasible() 
        """

        self.n = [] # "n" decision variable, defining the scheduled server and CPU alloc' of each VM.
        id = int (0)
        
        # Loop over all combinations of v, s, and a
        for v in range (self.NUM_OF_VNFs):
            for s in range (self.NUM_OF_SERVERS):
                mig = 1 if (s != self.cur_loc_of_vnf[v]) else 0 # if s is s different than v's current location, this implies a mig'
                for a in range (math.ceil (self.theta_times_traffic_in[v]), self.cpu_capacity_of_server[s]+1): # skip too small values of a (capacity allocations), which cause infinite comp' delay
                    denominator = a - self.theta_times_traffic_in[v] 
                    if (denominator <= 0): # skip too small values of a (capacity allocations), which cause infinite comp' delay
                        continue
 
                    comp_delay = 1/denominator
                    if (comp_delay > self.target_delay[v]): # Too high perf' degradation
                        continue 
                     
                    total_delay = comp_delay + self.servers_path_delay [s] [self.PoA_of_vnf[v]]
                    if (total_delay > self.target_delay[v]): # Too high perf' degradation
                        continue
                    self.n.append (
                    {'v'        : v, 
                    's'         : s,
                    'a'         : a,
                    'id'        : id,
                    'comp delay' : comp_delay,
                    'mig'       : mig})
                    
                    id +=1 

        for item in self.n:
            s = item['s']
            item['cost'] = total_delay / self.target_delay[v] + item['a'] + item['mig'] * self.mig_cost[v]
            if (self.verbose == 1):
                print (item)
    
    def gen_LP (self):
        # Calculate the computation cost
        return 1
        
    def __init__ (self, verbose = 0):
        
        self.verbose = verbose
        use_custom_netw = True
        if (use_custom_netw == True):
            self.custom_three_nodes_tree()
        else:
            self.gen_tree()
                 
        self.num_of_vnfs_in_chain   = np.ones (self.NUM_OF_USERS, dtype='uint8')
        self.NUM_OF_VNFs            = sum (self.num_of_vnfs_in_chain).astype ('uint')

        self.mig_cost = np.ones (self.NUM_OF_SERVERS) # np.random.rand (self.NUM_OF_SERVERS)         
        self.uniform_cpu_capacity   = 3
        self.cpu_capacity_of_server = self.uniform_cpu_capacity * np.ones (self.NUM_OF_SERVERS, dtype='int8')     
        self.max_cpu_capacity_of_server = max (self.cpu_capacity_of_server)  
        self.theta                  = 0.5 * np.ones (self.NUM_OF_VNFs) #cpu units to process one unit of data
        self.traffic_in             = np.ones (self.NUM_OF_VNFs) #traffic_in[v] is the bw of v's input traffic ("\lambda_v")
        self.theta_times_traffic_in = self.theta * self.traffic_in 

        #self.traffic_out            = np.ones (self.NUM_OF_VNFs) #traffic_in[v] is the bw of v's input traffic
        self.nxt_loc_of_vnf         = np.array (self.NUM_OF_VNFs)   # nxt_loc_of_vnf[v] will hold the id of the server planned to host VNF v
        self.nxt_cpu_alloc_of_vnf   = np.array (self.NUM_OF_VNFs)
        self.target_delay           = 15 * np.ones (self.NUM_OF_VNFs)    # the desired (max) delay (aka Delta)
        self.perf_deg_of_vnf        = np.zeros (self.NUM_OF_VNFs)
        
        # Calculate v^+ of each VNF v.
        # vpp(v) will hold the idx of the next VNF in the same chain.
        # if v is the last VNF in its chain, then vpp(v) will hold the PoA of this chain's user  
        self.vpp = np.zeros (self.NUM_OF_VNFs, dtype = 'uint')
        v = 0
        for chain in range (self.NUM_OF_CHAINS):
            for idx_in_chain in range (self.num_of_vnfs_in_chain[chain]):
                self.vpp [v] = v+1 if (idx_in_chain < self.num_of_vnfs_in_chain[chain]-1) else self.PoA_of_user[chain]
                v += 1
       
        self.PoA_of_vnf = np.zeros (self.NUM_OF_VNFs, dtype = 'uint') # self.PoA_of_vnf[v] will hold the PoA of the user using VNF v
        v = 0
        for chain in range (self.NUM_OF_CHAINS):
            for __ in range (self.num_of_vnfs_in_chain[chain]):
                self.PoA_of_vnf [v] = self.PoA_of_user[chain]
                v += 1

        # self.mig_comp_delay  = np.ones (self.NUM_OF_VNFs)     # self.mig_comp_delay[v] hold the migration's computational cost of VM v. Currently unused.
        # self.mig_data       = 0 * np.ones (self.NUM_OF_VNFs) # self.mig_data[v] amount of data units to transfer during the migration of VM v. Currently unused.
        self.mig_bw         = 1.1 * np.ones (self.NUM_OF_VNFs)

        self.min_cost = float ('inf')
        self.best_nxt_cpu_alloc_of_vnf = np.array (self.NUM_OF_VNFs)
        self.best_nxt_loc_of_vnf       = np.array (self.NUM_OF_VNFs)   # nxt_loc_of_vnf[v] will hold the id of the server planned to host VNF v

        if (self.verbose == 1):
            print (self.PoA_of_vnf)
        self.gen_n()
        self.const_num = int(0)
#         self.print_vars ()
#         self.print_obj_function ()
#         self.print_vars_leq1_const ()
#         self.print_single_alloc_const ()
#         printf (self.LP_output_file, '\nend;\n')
#         print (next (item for item in self.n if item['v'] == 1))
        self.print_cpu_cap_const()
        exit ()
        
        self.gen_c()
        self.brute_force()
        
        
    def perf_degradation (self, v, loc_v, loc_vpp, denominator):
        """
        Calculate the performance degradation of a VM. 
        Inputs
        - loc_v - location of v, 
        - loc_vpp - location of v^+ (the next VM after v in the service chain)
        - cpu_alloc - (suggested) cpu allocation of v
        
        """
        return ( 1 / denominator + self.servers_path_delay[loc_v][loc_vpp] ) / self.target_delay[v]
            
    
    def is_feasible(self):
        
        # Max CPU capacity of server constraint
        available_cpu_at_server = self.cpu_capacity_of_server.copy () 
        for v in range (self.NUM_OF_VNFs):
            available_cpu_at_server[self.nxt_loc_of_vnf[v]] -= self.nxt_cpu_alloc_of_vnf[v]
            if (available_cpu_at_server[self.nxt_loc_of_vnf[v]] < 0):
                return False # allocated more cpu than s's capacity

        # Finite computation delay constraint and max perf' degradation constraint 
        for v in range (self.NUM_OF_VNFs):               
            denominator = self.nxt_cpu_alloc_of_vnf[v] - self.theta_times_traffic_in[v] 
            if (denominator <= 0):
                return False
            self.perf_deg_of_vnf[v] = self.perf_degradation (
                v, self.nxt_loc_of_vnf[v], self.nxt_loc_of_vnf[self.vpp[v]], denominator)
            if (self.perf_deg_of_vnf[v] > 1):
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
        """"
        Calculate the cost of a feasible solution, given by its nxt_loc_of_vnf (y) and nxt_cpu_alloc (\beta_{vs}), and assuming that the perf' deg' was already calculated.
        Note that it's only for a feasible sol', as the calculation uses the array self.perf_deg_of_vnf, that should have been calculated by is_feasible() 
        """
        cost = sum (self.perf_deg_of_vnf) + \
               sum (self.mig_cost[v] for v in range (self.NUM_OF_VNFs) if self.cur_loc_of_vnf[v] != self.nxt_loc_of_vnf[v]) + \
               sum (self.nxt_cpu_alloc_of_vnf)
#         for v in range (self.NUM_OF_VNFs):
#             cost = cost if (self.cur_loc_of_vnf[v] == self.nxt_loc_of_vnf[v]) else cost + self.mig_cost[v] 
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
                    printf (self.output_file, 'beta = \n{}\nCost = {:.4}\n\n' .format (self.beta, self.cost ()))
                    # printf (self.output_file, 'cost by n = {:.4}\n\n' .format (self.calc_cost_by_n()))
                
                self.nxt_cpu_alloc_of_vnf = self.inc_array (self.nxt_cpu_alloc_of_vnf, 2, self.uniform_cpu_capacity)
            self.nxt_loc_of_vnf = self.inc_array(self.nxt_loc_of_vnf, 0, self.NUM_OF_SERVERS-1)
        self.calc_beta (self.best_nxt_loc_of_vnf, self.best_nxt_cpu_alloc_of_vnf)
        printf (self.output_file, '\nBest solution is:\nbeta = \n{}\nCost = {:.4}\n\n' .format (self.beta, self.min_cost))

