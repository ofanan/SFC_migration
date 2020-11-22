import networkx as nx
import numpy as np
import math
import itertools 

from printf import printf
from LP_file_parser import LP_file_parser
import Check_sol
import obj_func
from _overlapped import NULL

class toy_example (object):
    
    def gen_custom_three_nodes_tree (self):
        """
        generate a custom three-nodes tree (root and 2 leaves). 
        """
  
        self.list_of_links = [ [0,1], [1,0], [1,2], [2,1] ]
        #servers_path_delay[i][j] holds the netw' delay of the path from server i to server j
        self.servers_path_delay  = self.uniform_link_delay * np.array ([
                                   [0, 1, 2],
                                   [1, 0, 1],
                                   [2, 1, 0]
                                  ]) 
    
        
        # self.links_of_path[s][d] will contain the list of links found in the path from s to d
        self.links_of_path = [
            [
                [],
                [[0,1]],
                [[0,1], [1,2]]
            ],
            [
                [[1,0]],
                [],
                [[1,2]]  
            ],
            [
                [[2,1],[1,0]],
                [[2,1]],
                []
            ]
        ]
        
#         self.paths_of_link = [ # self.paths_of_link[i][j] will contain a list of the paths [s,d] which use link (i,j)
#             [
#                 [],               # list of paths in which l(0,0) appears 
#                 [ [0,1], [0,2] ], # list of paths in which l(0,1) appears
#                 []                # list of paths in which l(0,2) appears
#             ],
#             [
#                 [ [1,0], [2,0] ], # list of paths in which l(1,0) appears 
#                 [],               # list of paths in which l(1,1) appears
#                 [ [0,2], [1,2]]                # list of paths in which l(1,2) appears
#             ],
#             [
#                 [],               # list of paths in which l(2,0) appears 
#                 [ [2,1], [2,0] ], # list of paths in which l(2,1) appears
#                 []                # list of paths in which l(2,2) appears
#             ]
#         ]

        self.NUM_OF_SERVERS             = 3
        self.NUM_OF_USERS               = 1
        self.NUM_OF_PoA                 = 1
        self.NUM_OF_LINKS               = 4
        self.capacity_of_link           = np.zeros ( (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS))
        self.capacity_of_link[0][1]     = self.uniform_link_capacity
        self.capacity_of_link[1][0]     = self.uniform_link_capacity
        self.capacity_of_link[1][2]     = self.uniform_link_capacity
        self.capacity_of_link[2][1]     = self.uniform_link_capacity
        self.NUM_OF_CHAINS              = self.NUM_OF_USERS
        self.res_output_file            = open ("../res/custom_tree.res", "a")

    def gen_parameterized_tree (self):
        """
        Generate a parameterized regular three-nodes tree (root and 2 leaves). 
        """
        self.G                  = nx.generators.classic.balanced_tree (r=3, h=2) # Generate a tree of height h where each node has r children.
        self.NUM_OF_SERVERS     = self.G.number_of_nodes()
        self.NUM_OF_USERS       = 2
        self.NUM_OF_PoA         = 2

        self.servers_path_delay = np.array ((self.NUM_OF_SERVERS, self.NUM_OF_SERVERS)) # $$$ TBD: fix links' delay, and calc servers_path_delay accordingly
        self.servers_path_delay = np.random.rand (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS) 

        shortest_path = nx.shortest_path(self.G)

        # self.links_of_path[s][d] will contain the list of links found in the path from s to d
        self.links_of_path      = []
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
        self.list_of_links = []
        for edge in self.G.edges:
            self.capacity_of_link[edge[0]][edge[1]] = self.uniform_link_capacity
            self.capacity_of_link[edge[1]][edge[0]] = self.uniform_link_capacity
            self.list_of_links.append ([edge[0], edge[1]])

       
        
    def __init__ (self, verbose = -1):
        
        self.verbose                = verbose
        self.uniform_link_capacity  = 20
        self.uniform_cpu_capacity   = 5
        self.uniform_link_delay     = 1

        use_custom_netw = True
        if (use_custom_netw == True):
            self.gen_custom_three_nodes_tree()
        else:
            self.gen_parameterized_tree()
            
        self.PoA_of_user            = 2 * np.ones (self.NUM_OF_USERS) # np.random.randint(self.NUM_OF_PoA, size = self.NUM_OF_USERS) # PoA_of_user[u] will hold the PoA of the user using chain u       
        self.num_of_vnfs_in_chain   = 2 * np.ones (self.NUM_OF_USERS, dtype='uint8')
        self.NUM_OF_CHAINS          = self.NUM_OF_USERS
        self.NUM_OF_VNFs            = sum (self.num_of_vnfs_in_chain).astype ('uint')

        self.cur_loc_of_vnf         = [0, 0] # np.random.randint(self.NUM_OF_SERVERS, size = self.NUM_OF_VNFs) # Initially, allocate VMs on random VMs
        self.cur_cpu_alloc_of_vnf   = [2, 1] #2 * np.ones (self.NUM_OF_VNFs)                                  # Initially, allocate each VNs uniform amount CPU units

        self.mig_bw                 = 5 * np.ones (self.NUM_OF_VNFs)
        self.mig_cost               = [3, 4] #5 * np.ones (self.NUM_OF_VNFs) # np.random.rand (self.NUM_OF_VNFs)         
        self.cpu_capacity_of_server = self.uniform_cpu_capacity * np.ones (self.NUM_OF_SERVERS, dtype='uint8')     
        self.theta                  = np.ones (self.NUM_OF_VNFs) #cpu units to process one unit of data
        self.traffic_in             = [0.1, 0.5] #traffic_in[v] is the bw of v's input traffic ("\lambda_v").
        self.theta_times_traffic_in = self.theta * self.traffic_in [0:self.NUM_OF_VNFs]
        self.traffic_out_of_chain   = 1 * np.ones (self.NUM_OF_USERS) #traffic_out_of_chain[c] will hold the output traffic (amount of traffic back to the user) of chain c 

        # self.VM_target_delay           = 10 * np.ones (self.NUM_OF_VNFs)    # the desired (max) delay (aka Delta). Currently unused
        # self.perf_deg_of_vnf        = np.zeros (self.NUM_OF_VNFs). Currently unused
        
        # Calculate v^+ of each VNF v.
        # vpp(v) will hold the idx of the next VNF in the same chain.
        # if v is the last VNF in its chain, then vpp(v) will hold the PoA of this chain's user  
        self.vpp                    = np.zeros (self.NUM_OF_VNFs, dtype = 'uint')
        self.v0                     = np.zeros (self.NUM_OF_CHAINS, dtype = 'uint') # self.v0 will hold a list of all the VNFs which are first in their chain
        self.v_inf                  = np.zeros (self.NUM_OF_CHAINS, dtype = 'uint') # self.v_inf will hold a list of all the VNFs which are last in their chain
        self.v_not_inf              = [] # list of vnf's that are NOT last in the chain
        self.PoA_of_vnf = np.zeros (self.NUM_OF_VNFs, dtype = 'uint') # self.PoA_of_vnf[v] will hold the PoA of the user using VNF v

        self.vnf_in_chain = np.empty (shape = self.NUM_OF_CHAINS, dtype = object) # self.vnf_in_chain[c] will hold a list of the VNFs in chain c  
        v = 0
        for chain_num in range (self.NUM_OF_CHAINS):
            self.vnf_in_chain[chain_num] = []
            for idx_in_chain in range (self.num_of_vnfs_in_chain[chain_num]):
                self.vnf_in_chain[chain_num].append (v)
                if (idx_in_chain == 0):
                    self.v0[chain_num] = v 
                if (idx_in_chain == self.num_of_vnfs_in_chain[chain_num]-1): # Not "elif", because in the case of a single-VM chain, the first is also the last
                    self.v_inf[chain_num] = v
                    self.vpp [v] = self.PoA_of_user[chain_num]
                else: # Not the last VM in the chain
                    self.v_not_inf.append(v)
                    self.vpp [v] = v+1 
                self.PoA_of_vnf [v] = self.PoA_of_user[chain_num]    
                v += 1
       
        # self.mig_comp_delay  = np.ones (self.NUM_OF_VNFs)     # self.mig_comp_delay[v] hold the migration's computational cost of VM v. Currently unused.
        # self.mig_data       = 0 * np.ones (self.NUM_OF_VNFs) # self.mig_data[v] amount of data units to transfer during the migration of VM v. Currently unused.

        self.cfg_output_file           = open ("../res/custom_tree.cfg", "w")
                   
    def run (self, uniform_mig_cost, chain_target_delay = 9, gen_LP = True, run_brute_force = True):
                
        self.chain_target_delay             = chain_target_delay * np.ones (self.NUM_OF_CHAINS)
        self.mig_cost                       = uniform_mig_cost * np.ones (self.NUM_OF_VNFs)
        if (self.verbose == 1):
            printf (self.cfg_output_file, 'PoA = {}\n' .format (self.PoA_of_user))
            printf (self.cfg_output_file, 'cur VM loc = {}\n' .format (self.cur_loc_of_vnf))
            printf (self.cfg_output_file, 'cur CPU alloc = {}\n' .format (self.cur_cpu_alloc_of_vnf))
            printf (self.cfg_output_file, 'mig bw = {}\n' .format (self.mig_bw))
            printf (self.cfg_output_file, 'mig cost = {}\n' .format (self.mig_cost))
            printf (self.cfg_output_file, 'lambda_v = {}\n' .format (self.traffic_in))
            printf (self.cfg_output_file, 'uniform cpu capacities = {}\n' .format (self.uniform_cpu_capacity))
            printf (self.cfg_output_file, 'uniform link capacities = {}\n' .format (self.uniform_link_capacity))
            printf (self.cfg_output_file, 'theta_times_traffic_in = {}\n' .format (self.theta_times_traffic_in))
            printf (self.cfg_output_file, 'traffic back to user = {}\n' .format (self.traffic_out_of_chain))
            printf (self.cfg_output_file, 'path delay = \n{}\n' .format (self.servers_path_delay))
            printf (self.cfg_output_file, 'chain_target_delay = {}\n\n' .format (self.chain_target_delay))

        self.gen_n()
        if (gen_LP):
            self.LP_output_file             = open ("../res/custom_tree.LP", "w")
            self.constraint_check_script    = open ("Check_sol.py", "w")
            self.obj_func_calc_script       = open ("obj_func.py", "w")
            self.constraint_check_script    = open ("Check_sol.py", "w")    
            self.constraint_num             = int(0)
            self.print_vars ()
            self.gen_p()
            self.print_obj_function ()
            self.gen_all_constraints ()
            self.LP_output_file.close ()
            self.constraint_check_script.close ()
        if (run_brute_force):
            self.min_cost                   = float ('inf')
            self.best_nxt_cpu_alloc_of_vnf  = np.array (self.NUM_OF_VNFs)
            self.best_nxt_loc_of_vnf        = np.array (self.NUM_OF_VNFs)   # nxt_loc_of_vnf[v] will hold the id of the server planned to host VNF v
            self.brute_force_by_n ()
            if (self.min_cost == float ('inf')):
                print ('Did not find a feasible sol')
                printf (self.res_output_file, '\t{:.0f} & N/A & No feasible solution \\tabularnewline \hline \n' .format(self.chain_target_delay[0]))
                return
            self.sol_to_loc_alloc (self.best_n)
            self.print_sol()
        
    def calc_paths_of_links (self):
        """
        Given self.links_of_path (list of links which appear in each path), 
        calculate self.paths_of_link (links of path in which each  link appears).
        """
        self.paths_of_link = [] #np.empty(shape = (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS), dtype = object)
        for link in self.list_of_links:
            paths = []
            for src in range (self.NUM_OF_SERVERS):
                for dst in range (self.NUM_OF_SERVERS): 
                    if (link in self.links_of_path[src][dst]): # Does think link appear in the path from src to dst
                        paths.append ([src, dst]) # Yep --> append it to the list of paths in which this link appears
            self.paths_of_link.append ({'link' : link, 'paths' : paths}) 
                        
                     
    def gen_all_constraints (self):
        """
        Generate all the constraints. 
        """ 
        printf (self.constraint_check_script, 'def Check_sol (X):\n')
        printf (self.constraint_check_script, '\t"""\n\tCheck whether a solution for the LP problem satisfies all the constraints\n\t"""\n')
        self.gen_leq1_constraints ()
        self.gen_single_alloc_constraints ()
        self.gen_cpu_cap_constraints ()
        self.calc_paths_of_links ()
        self.gen_link_cap_constraints ()
        self.gen_chain_delay_constraints ()
        printf (self.LP_output_file, '\nend;\n')
        printf (self.constraint_check_script, '\n\n\treturn True\n')


    def gen_leq1_constraints (self):
        """
        Print the var' ranges constraints: each var should be <=1  
        """
        for __ in self.n:
            printf (self.LP_output_file, 'subject to X_leq1_C{}: 1*X{} <= 1;\n\n' .format (self.constraint_num, __['id'], __['id']) )
            
            # The Python script doesn't really need these constraints, as anyway it checks only binary values for the decision variables. Hence, it's commented out.
            # printf (self.constraint_check_script, '\tif (X[{}] > 1):\n\t\treturn False\n\n' .format (__['id']))
            self.constraint_num += 1
        printf (self.LP_output_file, '\n')

    
    def gen_chain_delay_constraints (self):
        """
        Print the constraints of maximum delay of each chain in a LP format
        """
        for chain_num in range (self.NUM_OF_CHAINS):
            list_of_decision_vars_in_lin_eq  = [] # The decision vars that will appear in the relevant lin' constraint 
            list_of_coefs_in_lin_eq          = [] # coefficients of the decision vars that will appear in the relevant lin' constraint           
            first_decision_vars_in_mult_eq   = [] # The first decision vars that will appear in the relevant multiplicative constraint 
            scnd_decision_vars_in_mult_eq    = [] # The scnd decision vars that will appear in the relevant multiplicative constraint
            list_of_coefs_in_mult_eq         = [] # coefficients of the decision vars that will appear in the relevant lin' constraint           
 
            for v in self.vnf_in_chain[chain_num]:
                
                # Consider the computation delay
                for s in range (self.NUM_OF_SERVERS):
                    for y_vs in list (filter (lambda item : item['v'] == v and item['s'] == s, self.ids_of_y_vs) ):
                        for id in y_vs['ids']: # for each relevant decision var'                             
                            list_of_decision_vars_in_lin_eq.append (id) # Add this n_{vsa} decision var to the list of decision vars used in this constraint
                            
                            # Extract the comp' delay implied by this decision var
                            for n_vsa in list (filter (lambda item : item['id'] == id, self.n)):
                                list_of_coefs_in_lin_eq.append (n_vsa['comp delay'])
                                
                idx_in_chain = self.vnf_in_chain[chain_num].index (v) # idx_in_chain holds the idx of v in its chain
                
                # Consider the delay from the PoA to the first VNF in this chain
                if (idx_in_chain == 0): # v is the first VNF in this chain
                    for s in range (self.NUM_OF_SERVERS):
                        for y_vs in list (filter (lambda item : item['v'] == v and item['s'] == s, self.ids_of_y_vs) ):
                            for id in y_vs['ids']: 
                                list_of_coefs_in_lin_eq [list_of_decision_vars_in_lin_eq.index(id)] += self.servers_path_delay[self.PoA_of_vnf[v]][s] 
                     
                # Consider the delay from the PoA from the last VNF in this chain to the PoA
                if (idx_in_chain == len(self.vnf_in_chain[chain_num])-1): # v is the last VNF in this chain. Not Elsif, because it can be both the first and last in its chain.
                    for s in range (self.NUM_OF_SERVERS):
                        for y_vs in list (filter (lambda item : item['v'] == v and item['s'] == s, self.ids_of_y_vs) ):
                            for id in y_vs['ids']: 
                                list_of_coefs_in_lin_eq [list_of_decision_vars_in_lin_eq.index(id)] += self.servers_path_delay[s][self.PoA_of_vnf[v]]
                                
                    continue # The last vnf in its chain has no netw' delay to the next VNF in the chain
                
                # Now we know that v isn't the last in its chain --> add the delay from v to vpp
                vpp = self.vpp[v]
                for s in range (self.NUM_OF_SERVERS): # for every possible location of v
                    for s_prime in range (self.NUM_OF_SERVERS): # for every possible location of vpp (the next VM in the chain)
                        if (s == s_prime): # if v and vpp are scheduled to the same server, no netw' delay between them
                            continue
                        
                        for y_s in list (filter (lambda item : item['v'] == v and item['s'] == s, self.ids_of_y_vs) ):
                            for id_v in y_s['ids']:     
                                for y_s_prime in list (filter (lambda item : item['v'] == vpp and item['s'] == s_prime, self.ids_of_y_vs) ):
                                    for id_vpp in y_s_prime['ids']:     
                                
                                        first_decision_vars_in_mult_eq.append   (id_v) 
                                        scnd_decision_vars_in_mult_eq.append    (id_vpp)
                                        list_of_coefs_in_mult_eq.append         (self.servers_path_delay[s][s_prime])
                        
                
                
            self.print_eq ('chain_delay', list_of_coefs_in_lin_eq, list_of_decision_vars_in_lin_eq, self.chain_target_delay [chain_num],
                           list_of_coefs_in_mult_eq, first_decision_vars_in_mult_eq, scnd_decision_vars_in_mult_eq)         


    def gen_link_cap_constraints (self):
        """
        Print the constraints of maximum link's capacity in a LP format
        """
        printf (self.LP_output_file, '\n')
        for l in self.list_of_links:
            link_l_avail_bw                  = self.capacity_of_link[l[0], l[1]]
            list_of_decision_vars_in_lin_eq  = [] # The decision vars that will appear in the relevant lin' constraint 
            list_of_coefs_in_lin_eq          = [] # coefficients of the decision vars that will appear in the relevant lin' constraint           
            first_decision_vars_in_mult_eq   = [] # The first decision vars that will appear in the relevant multiplicative constraint 
            scnd_decision_vars_in_mult_eq    = [] # The scnd decision vars that will appear in the relevant multiplicative constraint
            list_of_coefs_in_mult_eq         = [] # coefficients of the decision vars that will appear in the relevant lin' constraint           
            
            for list_of_paths_using_link_l in list (filter (lambda item : item['link'] == l, self.paths_of_link)):
                list_of_paths_using_link_l = list_of_paths_using_link_l['paths'] 

            # Consider the BW from the PoA to the first VM in each chain
            for v0 in self.v0: # for each VNF which is the first in its chain
                for s in range(self.NUM_OF_SERVERS):
                    if ( not( [self.PoA_of_vnf[v0], s] in list_of_paths_using_link_l)): # The path (PoA(v0), s) doesn't use link l
                        continue
                    
                    # Now we know that the path from V0's PoA to s uses link l
                    if (self.x[v0][s]): # v0 is already located on server s 
                        link_l_avail_bw -= self.traffic_in[v0]
                    else: # v0 is NOT currently located on server s
                         for y_vs in list (filter (lambda item : item['v'] == v0 and item['s'] == s, self.ids_of_y_vs) ):
                             for id in y_vs['ids']:                             
                                 list_of_decision_vars_in_lin_eq.append (id) 
                                 list_of_coefs_in_lin_eq.        append (self.traffic_in[v0])
            
            # Consider the BW from the last VM in each chain to the PoA 
            for chain in range (self.NUM_OF_CHAINS): # for each VNF which is the last in its chain
                v_inf = self.v_inf[chain]
                for s in range(self.NUM_OF_SERVERS):
                    if ( not( [s, self.PoA_of_vnf[v_inf]] in list_of_paths_using_link_l)): # The path (s, PoA(v_inf)) doesn't use link l
                        continue
                    
                    # Now we know that the path from s to v_inf's PoA uses link l
                    traffic_out = self.traffic_out_of_chain [chain]
                    if (self.x[v_inf][s]): # if x_{vs} == 1, namely v_inf is already located on server s 
                        link_l_avail_bw -= traffic_out
                    else: # x[v][s] == 0
                         for y_vs in list (filter (lambda item : item['v'] == v_inf and item['s'] == s, self.ids_of_y_vs) ):
                             for id in y_vs['ids']:               
                                if (id in list_of_decision_vars_in_lin_eq): # Already seen, and wrote a coef', for this decision var, for this inequality
                                    list_of_coefs_in_lin_eq [list_of_decision_vars_in_lin_eq.index(id)] += traffic_out
                                else:
                                    list_of_decision_vars_in_lin_eq.append (id) 
                                    list_of_coefs_in_lin_eq.        append (traffic_out)

            # Consider the bw due to traffic along the chain
            for v in self.v_not_inf: # For every VNF that is not last in its chain
                vpp = self.vpp[v]    # vpp is the next VM in that chain
                for s in range (self.NUM_OF_SERVERS): # for every possible location of v
                    for s_prime in range (self.NUM_OF_SERVERS): # for every possible location of vpp (the next VM in the chain)
                        if (s == s_prime or ( not( [s, s_prime] in list_of_paths_using_link_l))): # # if v and vpp are scheduled to the same server, no bw created for the traffic from v to vpp; if the path (s, s') doesn't use link l, it doesn't generate a new component for this constraint 
                            continue
                
                        if (self.x[v][s] and self.x[vpp][s_prime]): # the path s --> s' is used already in the cur allocation
                            link_l_avail_bw -= self.theta_times_traffic_in[vpp]
                        else:            # the path s --> s' is NOT used by the cur allocation. Need to extract all the decision variables that imply using this path
                            for y_s in list (filter (lambda item : item['v'] == v and item['s'] == s, self.ids_of_y_vs) ): #re
                                for id_v in y_s['ids']:     
                                    for y_s_prime in list (filter (lambda item : item['v'] == vpp and item['s'] == s_prime, self.ids_of_y_vs) ):
                                        for id_vpp in y_s_prime['ids']:     
                                    
                                            first_decision_vars_in_mult_eq.append (id_v) 
                                            scnd_decision_vars_in_mult_eq. append (id_vpp)
                                            list_of_coefs_in_mult_eq.      append (self.theta_times_traffic_in[vpp])

            
            
            # Consider the bw due to migrations
            for v in range (self.NUM_OF_VNFs):
                for mig_src in range (self.NUM_OF_SERVERS):
                    for mig_dst in range (self.NUM_OF_SERVERS):
                         
                        if (mig_dst == mig_src or not ([mig_src, mig_dst] in list_of_paths_using_link_l) or self.x[v][mig_src] == 0):
                            continue
                         
                        for y_vs in list (filter (lambda item : item['v'] == v and item['s'] == mig_dst, self.ids_of_y_vs) ):
                            for id in y_vs['ids']:     
                                if (id in list_of_decision_vars_in_lin_eq): # Already seen, and wrote a coef', for this decision var, for this inequality
                                    list_of_coefs_in_lin_eq [list_of_decision_vars_in_lin_eq.index(id)] += self.mig_bw[v] 
                                else:
                                    list_of_decision_vars_in_lin_eq.append (id) 
                                    list_of_coefs_in_lin_eq.append         (self.mig_bw[v])
                                    

            # Print the constraint obtained for this link
            if (len(list_of_decision_vars_in_lin_eq) == 0 and len(list_of_coefs_in_mult_eq) == 0): #No one uses this link --> no constraints
                continue
            
            self.print_eq ('link_cap', list_of_coefs_in_lin_eq, list_of_decision_vars_in_lin_eq, link_l_avail_bw,
                           list_of_coefs_in_mult_eq, first_decision_vars_in_mult_eq, scnd_decision_vars_in_mult_eq)         
            
    def print_eq (self, constraint_name, list_of_coefs_in_lin_eq, list_of_decision_vars_in_lin_eq, constant, 
                  list_of_coefs_in_mult_eq = None, first_decision_vars_in_mult_eq = None, scnd_decision_vars_in_mult_eq = None):
        """
        Print the obtained inequality into two output files: 
        self.LP_output_file - write to this file the inequality in a Linear-Prog. format, e.g.: 
            3*X1 + 2*X2 <= 5
        self.constraint_check_script - write to this file the inequality as a Python-code that returns false if the inequlity isn't satisfied, e.g.
            if (3*X[1] + 2*X[2] > 5):
                return False
                
        """
        
        printf (self.LP_output_file, 'subject to {}_C{}: ' .format (constraint_name, self.constraint_num))
        printf (self.constraint_check_script, '\t# {}_C{}:\n\tif (' .format (constraint_name, self.constraint_num))
        self.constraint_num += 1

        is_first = True
        if (not (list_of_coefs_in_mult_eq == None)): # If there exist multiplicative components in the inequality
            # Print the mult' constraint obtained for this chain
            for decision_var_idx in range (len(first_decision_vars_in_mult_eq)): 
                if (list_of_coefs_in_mult_eq [decision_var_idx] == 0): # coefficient is 0 --> may skip this component
                    continue

                if (is_first): 
                    is_first = False
                else: # before for any further component, add the "+" sign
                    printf (self.LP_output_file, '+ ')
                    printf (self.constraint_check_script, '+ ')
                printf (self.LP_output_file, '{:.4f}*X{}*X{} ' .format (
                    list_of_coefs_in_mult_eq        [decision_var_idx],  
                    first_decision_vars_in_mult_eq  [decision_var_idx],
                    scnd_decision_vars_in_mult_eq   [decision_var_idx]))           
                printf (self.constraint_check_script, '{:.4f}*X[{}]*X[{}] ' .format (
                    list_of_coefs_in_mult_eq        [decision_var_idx],  
                    first_decision_vars_in_mult_eq  [decision_var_idx],
                    scnd_decision_vars_in_mult_eq   [decision_var_idx]))           
        
        # For convenience, order the decision vars to appear in an increasing ID # order            
        list_of_coefs_in_lin_eq = [list_of_coefs_in_lin_eq[i] for i in np.argsort(list_of_decision_vars_in_lin_eq)]
        list_of_decision_vars_in_lin_eq = np.sort (list_of_decision_vars_in_lin_eq)
        
        for decision_var_idx in range (len(list_of_decision_vars_in_lin_eq)): 
            if (list_of_coefs_in_lin_eq [decision_var_idx] == 0): # coefficient is 0 --> may skip this component
                continue
            if (not(is_first)): # for any component beside the first one, need to add the "+" sign
                printf (self.LP_output_file, '+ ')
                printf (self.constraint_check_script, '+ ')
            else:
                is_first = False
                
            printf (self.LP_output_file, '{:.4f}*X{} ' .format (
                list_of_coefs_in_lin_eq         [decision_var_idx],  
                list_of_decision_vars_in_lin_eq [decision_var_idx]))
            printf (self.constraint_check_script, '{:.4f}*X[{}] ' .format (
                list_of_coefs_in_lin_eq         [decision_var_idx],  
                list_of_decision_vars_in_lin_eq [decision_var_idx]))
               
        printf (self.LP_output_file, '<= {};\n\n' .format (constant))
        printf (self.constraint_check_script, '> {}):\n\t\treturn False\n\n' .format (constant))
        
            
    def gen_cpu_cap_constraints (self):
        """
        Print the constraints of maximum server's CPU capacity in a LP format
        """

        printf (self.LP_output_file, '\n')        
        for s in range (self.NUM_OF_SERVERS):

            # Consider the CPU consumed by all VMs that will be assigned to servers (by n_{vsa} decision variables)
            for item in list (filter (lambda item:  item['s'] == s, self.n )): # for each decision var' related to server s
                item['coef'] = item['a']

            # Consider the CPU consumed by all VMs that are currently assigned to servers (by p_{vsa} parameters)
            server_s_available_cap = self.cpu_capacity_of_server[s]
            for v in range (self.NUM_OF_VNFs):
                if (self.cur_loc_of_vnf[v] == s): # VM v is already using server s
                    a_cur = self.cur_cpu_alloc_of_vnf[v]
                    server_s_available_cap -= a_cur
                    for item in list (filter (lambda item : item['s'] == s and item['v']==v, self.n)):
                        item['coef'] -= a_cur 
                        
            # Print the data we have collected (all the non-zero coefficients)
            is_first = True
            for item in list (filter (lambda item:  item['s'] == s, self.n )): # for each decision var' related to server s
                if (item['coef'] == 0):
                    continue  
                if (is_first):
                    printf (self.LP_output_file, 'subject to max_cpu_C{}: {}*X{} ' .format (self.constraint_num, item['coef'], item['id']))
                    printf (self.constraint_check_script, '\t#max_cpu_C{}\n\tif ({}*X[{}] ' .format (self.constraint_num, item['coef'], item['id']))
                    self.constraint_num += 1
                    is_first = False
                else: 
                    coef = item['coef']
                    sign = '+' if (coef > 0) else '-'
                    abs_coef = abs(coef)
                    printf (self.LP_output_file,          '{} {}*X{} ' .format (sign, abs_coef, item['id']))
                    printf (self.constraint_check_script, '{} {}*X[{}] '  .format (sign, abs_coef, item['id']))
            printf (self.LP_output_file, ' <= {};\n' .format (server_s_available_cap))
            printf (self.constraint_check_script, ' > {}):\n\t\treturn False\n\n' .format (server_s_available_cap))




    def gen_single_alloc_constraints (self):
        """
        Print the constraint of a single allocation for each VM in a LP format
        """
        v = -1 
        for item in self.n:
            if (item['v'] == v): #Already seen decision var' related to this VM
                printf (self.LP_output_file, '+ X{}' .format (item['id']))
                printf (self.constraint_check_script, '+ X[{}]' .format (item['id']))
            else: # First time observing decision var' related to this VM
                if (v > -1):
                    printf (self.LP_output_file, ' = 1;\n' )
                    printf (self.constraint_check_script, ' == 1)):\n\t\treturn False\n' )
                printf (self.LP_output_file, 'subject to single_alloc_C{}:   X{} ' .format (self.constraint_num, item['id'])) 
                printf (self.constraint_check_script, '\tif (not (X[{}] ' .format (item['id']))
                v = item['v']
            self.constraint_num += 1
        printf (self.LP_output_file, ' = 1;\n\n' )
        printf (self.constraint_check_script, ' == 1)):\n\t\treturn False\n\n' )
               
    def print_vars (self):
        """
        Print the decision variables. Each variable is printed with the constraints that it's >=0  
        """
        for __ in self.n:
            printf (self.LP_output_file, 'var X{} >= 0;\n' .format (__['id'], __['id']) )
        printf (self.LP_output_file, '\n')


    def print_obj_function (self):
        """
        Print the objective function in a standard LP form (linear combination of the decision variables)
        """
        printf (self.LP_output_file, 'minimize z:   ')
        printf (self.obj_func_calc_script, 'def obj_func (X):\n')
        printf (self.obj_func_calc_script, '\t"""\n\tCalculate the objective function, given a feasible solution.\n\t"""\n\treturn ')
        is_first_item = True
        for item in self.n:
            if (not (is_first_item)):
                printf (self.LP_output_file, ' + ')
                printf (self.obj_func_calc_script, ' + ')
            printf (self.LP_output_file,            '{:.4f}*X{}' .format (item['cost'], item ['id']) ) 
            printf (self.obj_func_calc_script,   '{:.4f}*X[{}]' .format (item['cost'], item ['id']) ) 
            is_first_item = False
        printf (self.LP_output_file, ';\n\n')
        printf (self.obj_func_calc_script, '\n')
    
    def sol_to_loc_alloc (self, sol):
        """
        Translate a sol' to the optimization prob', given as a multi-dimensional binary vector n_{vsa}, to two vectors:
        nxt_loc_of_vnf[v] - will be s iff VNF v is scheduled to server s
        nxt_cpu_loc_of_vnf[v] - will be a iff a is assigned a units of CPU
        """
        
        self.nxt_loc_of_vnf         = np.empty (self.NUM_OF_VNFs, dtype = np.uint16)
        self.nxt_cpu_alloc_of_vnf   = np.empty (self.NUM_OF_VNFs, dtype = np.uint16)
        
        set_ids = [i for i, entry in enumerate(sol) if entry == 1]

        for id in set_ids:
            for item in list (filter (lambda item : item['id'] == id, self.n)):
                self.nxt_loc_of_vnf       [item['v']] = item['s']
                self.nxt_cpu_alloc_of_vnf [item['v']] = item['a']
                
    
    def gen_p (self):
        """
        Generate self.p (present CPU allocation params) based on the values of self.cur_loc_of_vnf and self.cur_cpu_alloc_of_vnf
        """

        # Generate self.p (present CPU allocation params) as a copy of relevant fields in self.n (scheduled CPU location variables)         
        self.p = []
        for n_item in self.n:
            self.p.append ({'v'     : n_item['v'],
                            's'     : n_item['s'],
                            'a'     : n_item['a'],
                            'id'    : n_item['id'],
                            'val'   : 0,
                            })
         
         # Hard-code an initial allocation
        for v in range (self.NUM_OF_VNFs):
            tmp_list  = list (filter (lambda item : item ['v'] == v and item ['s'] == self.cur_loc_of_vnf[v]  and item['a'] == self.cur_cpu_alloc_of_vnf[v], self.p))
            for item in tmp_list:
                item['val'] = 1  
         
        
        # Generate x and y
        self.gen_x_y ()
                
    
    def gen_x_y (self):
        """
        Generate the location variable (x and y).
        y are generated without specified value.  
        x's values are generated based on the value of self.p
        """    
        self.x = np.zeros (shape = (self.NUM_OF_VNFs, self.NUM_OF_SERVERS), dtype = bool) #[]
        self.y = []
        for v in range (self.NUM_OF_VNFs):
            for s in range (self.NUM_OF_SERVERS):
                self.y.append ({'v' : v, 's' : s} ) 
                tmp_list = list (filter (lambda item : item ['v'] == v and item['s'] == s, self.p))
                if (sum (item['val'] for item in tmp_list)==1):
                    self.x[v][s] = True
#                 self.x.append ({'v' : v, 
#                              's' : s, 
#                              'val' : True if (sum (item['val'] for item in tmp_list)==1) else False}
#                              ) 
            
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
        self.ids_of_y_vs = [] # will hold the IDs of all the n_vsa decision vars related to server s and VM v 
        id = int (0)
        
        # Loop over all combinations of v, s, and a
#         self.list_of_migrating_VMs = []
        for v in range (self.NUM_OF_VNFs):
            for s in range (self.NUM_OF_SERVERS):
                mig = True if (s != self.cur_loc_of_vnf[v]) else False # if s is s different than v's current location, this implies a mig'
                list_of_ids = []
                for a in range (math.ceil (self.theta_times_traffic_in[v]), self.cpu_capacity_of_server[s]+1): # skip too small values of a (capacity allocations), which cause infinite comp' delay
                    denominator = a - self.theta_times_traffic_in[v] 
                    if (denominator <= 0): # skip too small values of a (capacity allocations), which cause infinite comp' delay
                        continue
 
                    comp_delay = 1/denominator
                    # Check for per-VM target delay - currently unused
                    # if (comp_delay > self.VM_target_delay[v]): # Too high perf' degradation
                    #     continue 
                     
                    cost =  a
                    if (mig):
                        cost += self.mig_cost[v]
                    
                    item = {
                        'v'          : int(v), 
                        's'          : int(s),
                        'a'          : int(a),
                        'id'         : id,
                        'comp delay' : comp_delay,
                        'mig'        : mig,
                        'cost'       : cost
                    }
                    
                    self.n.append (item)                
                    
                    list_of_ids.append (id)
                    
                    if (self.verbose == 2):
                        print (item)
                    id +=1 
            
                self.ids_of_y_vs.append ({'v' : v, 's' : s, 'ids' : list_of_ids})
    

        if (self.verbose == 2):
            printf (self.cfg_output_file, 'self.n =\n')
            for item in self.ids_of_y_vs:
                printf (self.cfg_output_file, '{}\n' .format (item))
     

    def brute_force_by_n (self):
        self.min_cost = float ('inf')
        for sol in itertools.product ([0,1],repeat = len (self.n)):
            if ( not (Check_sol.Check_sol (sol)) ):
                continue
            cost = obj_func.obj_func (sol)
            if (cost < self.min_cost):
                self.min_cost = cost
                self.best_n = sol

    def print_sol (self):
        """
        Print the solution found
        """
        
        printf (self.res_output_file, '\t{:.2f} & {:.2f} & VM loc = {} cpu alloc = {} \\tabularnewline \hline \n' .format 
                (self.mig_cost[0], self.min_cost,  self.nxt_loc_of_vnf, self.nxt_cpu_alloc_of_vnf))
#         printf (self.cfg_output_file, 'min cost = {}, VM loc = {} cpu alloc = {} \n'  .format 
#                 (self.min_cost, self.nxt_loc_of_vnf, self.nxt_cpu_alloc_of_vnf))

        
