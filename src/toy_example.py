import networkx as nx
import numpy as np
import math
import itertools 

from printf import printf
import Check_sol
import obj_func
from _overlapped import NULL
from solve_problem_by_Cplex import solve_problem_by_Cplex

class toy_example (object):
    
    def gen_custom_three_nodes_tree (self):
        """
        generate a custom three-nodes tree (root and 2 leaves). 
        """
  
        self.list_of_links = [ [0,1], [1,0], [1,2], [2,1] ]
        #servers_path_delay[i][j] holds the netw' delay of the path from server i to server j
        self.servers_path_delay  = np.array ([
                                   [0, 1, 2],
                                   [1, 0, 1],
                                   [2, 1, 0]
                                  ], dtype = float) 
    
        
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

        self.servers_path_delay = np.array ((self.NUM_OF_SERVERS, self.NUM_OF_SERVERS)) 
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
        """
        Init a toy example - topology (e.g., chains, VMs, target delays etc.).
        """
        
        self.verbose                = verbose
        self.uniform_link_capacity  = 20
        self.uniform_cpu_capacity   = 5

        use_custom_netw = True
        if (use_custom_netw == True):
            self.gen_custom_three_nodes_tree()
        else:
            self.gen_parameterized_tree()
            
        self.PoA_of_user            = 2 * np.ones (self.NUM_OF_USERS, dtype = 'uint16') # np.random.randint(self.NUM_OF_USERS, size = self.NUM_OF_USERS) # PoA_of_user[u] will hold the PoA of the user using chain u       
        self.num_of_vnfs_in_chain   = 2 * np.ones (self.NUM_OF_USERS, dtype ='uint8')
        self.NUM_OF_CHAINS          = self.NUM_OF_USERS
        self.NUM_OF_VNFs            = sum (self.num_of_vnfs_in_chain).astype ('uint')

        self.cur_loc_of_vnf         = [0, 0] #[0, 0, 0, 1]# [0, 0, 0, 0] # np.random.randint(self.NUM_OF_SERVERS, size = self.NUM_OF_VNFs) # Initially, allocate VMs on random VMs
        self.cur_cpu_alloc_of_vnf   = [1, 1] #2 * np.ones (self.NUM_OF_VNFs)                                  # Initially, allocate each VNs uniform amount CPU units

        self.mig_bw                 = 5 * np.ones (self.NUM_OF_VNFs)
        self.cpu_capacity_of_server = self.uniform_cpu_capacity * np.ones (self.NUM_OF_SERVERS, dtype='uint8')     
        self.theta                  = np.ones (self.NUM_OF_VNFs) #cpu units to process one unit of data
        self.traffic_in             = [0.5, 0.9] #traffic_in[v] is the bw of v's input traffic ("\lambda_v"). The last entry is the traffic from the chain back to the user.
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

        self.vnf_in_chain                 = np.empty (shape = self.NUM_OF_CHAINS, dtype = object) # self.vnf_in_chain[c] will hold a list of the VNFs in chain c  
        self.theta_times_traffic_in_chain = np.empty (shape = self.NUM_OF_CHAINS, dtype = object) # self.theta_times_traffic_in_chain[c][j] will hold theta[v] * lambda[v], where v is the j-th VM in chain c
        v = 0
        for chain_num in range (self.NUM_OF_CHAINS):
            self.vnf_in_chain                 [chain_num] = []
            self.theta_times_traffic_in_chain [chain_num] = []
            for idx_in_chain in range (self.num_of_vnfs_in_chain[chain_num]):
                self.vnf_in_chain                [chain_num].append (v)
                self.theta_times_traffic_in_chain[chain_num].append (self.theta_times_traffic_in[v]) 
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

        self.write_to_prb_file = False # When true, will write outputs to a .prb file. - ".prb" - A .prb file may solve an LP problem using the online Eq. solver: https://online-optimizer.appspot.com/?model=builtin:default.mod
        self.write_to_py_file  = False # When true, will write to Py file, checking the feasibility and cost of a suggested sol'.  
        self.write_to_mod_file = False # When true, will write to a .mod file, fitting to IBM CPlex solver       
        self.write_to_cfg_file = True
        self.write_to_lp_file  = True  # When true, will write to a .lp file, which allows running Cplex using a Python's api.       
                
#     def calc_chain_netw_delay (self):
#         """
#         calculate the network delay along a given chain. 
#         This delay is the sum of the netw' delays along the chain only.
#         It doesn't include nor the delay to / from the PoA, neither the computation delay.  
#         """
#         netw_delay_along_chain = sum ()

                   
    def run (self, uniform_link_delay = 1, uniform_mig_cost = 3, chain_target_delay = 15, gen_LP = True, run_brute_force = True):
        """
        generate a LP formulation for a problem, and / or solve it by either a brute-force approach, or by Cplex / approximation alg' / random sol'.
        """
        
        self.chain_target_delay             = chain_target_delay * np.ones (self.NUM_OF_CHAINS)
        self.mig_cost                       = [3, 4] #uniform_mig_cost * np.ones (self.NUM_OF_VNFs)
        self.servers_path_delay             *= uniform_link_delay
        
        if (self.verbose == 1):
            self.debug_output_file = open ('../res/debug.res', 'w')

        if (self.write_to_cfg_file): # Write the static params
            self.cfg_output_file = open ("../res/custom_tree.cfg", "w")           
            printf (self.cfg_output_file, 'lambda_v = {}\n' .format (self.traffic_in))
            printf (self.cfg_output_file, 'uniform cpu capacities = {}\n'  .format (self.uniform_cpu_capacity))
            printf (self.cfg_output_file, 'uniform link capacities = {}\n' .format (self.uniform_link_capacity))
            printf (self.cfg_output_file, 'theta_times_traffic_in = {}\n'  .format (self.theta_times_traffic_in))
            printf (self.cfg_output_file, 'traffic back to user = {}\n'    .format (self.traffic_out_of_chain))
            printf (self.cfg_output_file, 'path delay = \n{}\n'            .format (self.servers_path_delay))
            printf (self.cfg_output_file, 'chain_target_delay = {}\n\n'    .format (self.chain_target_delay))

        self.gen_static_r  ()


        #self.gen_n()
        if (gen_LP):
            
            # prb_output_file will use the format of LP, used in: https://online-optimizer.appspot.com/?model=builtin:default.mod
            # with the addition of quadratic components (e.g. X1 * X3). 
            if (self.write_to_prb_file):
                self.prb_output_file        = open ("../res/custom_tree.prb", "w")           
            if (self.write_to_py_file):
                self.obj_func_by_Py         = open ("obj_func.py", "w")  # Will write to this file a Python function returning the cost of a given feasible sol
                self.constraint_check_by_Py = open ("Check_sol.py", "w") # Will write to this file a Python function which returns true iff a given sol is feasible
            if (self.write_to_mod_file):
                self.mod_output_file        = open ("../../Cplex/short/demo.mod", "w") # Will write to this file an IBM CPlex' .mod file, describing the problem
            if (self.write_to_lp_file):
                self.lp_output_file         = open ("../res/problem.lp", "w") # Will write to this file an IBM CPlex' .mod file, describing the problem

            if (self.write_to_cfg_file):
                printf (self.cfg_output_file, 'PoA = {}\n' .format (self.PoA_of_user))
                printf (self.cfg_output_file, 'cur VM loc = {}\n' .format (self.cur_loc_of_vnf))
                printf (self.cfg_output_file, 'cur CPU alloc = {}\n' .format (self.cur_cpu_alloc_of_vnf))
                printf (self.cfg_output_file, 'mig bw = {}\n' .format (self.mig_bw))
                printf (self.cfg_output_file, 'mig cost = {}\n' .format (self.mig_cost))
            
            self.constraint_num             = int(0)                     # A running number for counting the constraints   

            self.calc_dynamic_r ()
            if (self.write_to_lp_file):
                printf (self.lp_output_file, '\n\nSubject To')
            self.gen_leq1_constraints    ()
            self.gen_cpu_cap_constraints ()
            self.calc_paths_of_links ()
            self.gen_link_cap_constraints_to_lp()
            printf (self.lp_output_file, "\nEnd\n")

            self.gen_p ()                                                
#             if (self.write_to_prb_file or self.write_to_mod_file or self.write_to_py_file):
#                 self.print_vars ()                                          
#                 self.gen_obj_function ()
#                 self.gen_all_constraints ()
#                 self.prb_output_file.close ()
#                 self.constraint_check_by_Py.close ()
        set_vars_in_cpx_sol = asolve_problem_by_Cplex ('../res/problem.lp')
        if (run_brute_force):
            self.min_cost                   = float ('inf')
            self.best_nxt_cpu_alloc_of_vnf  = np.array (self.NUM_OF_VNFs)
            self.best_nxt_loc_of_vnf        = np.array (self.NUM_OF_VNFs)   # nxt_loc_of_vnf[v] will hold the id of the server planned to host VNF v
            self.brute_force_sa_pow_v ()
        
        
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
        Generate all the constraints for the problem, and print them to output files.
        Supported output file formats are:
        - ".mod" - for running by Cplex.
        - ".py"  - for running as a Python program.
        - ".prb" - for running as a LP problem by the online Eq. solver: https://online-optimizer.appspot.com/?model=builtin:default.mod
        - ".lp" - Cplex ".lp" format, that can be run by a Python api. 
        """ 
        if (self.write_to_py_file):
            printf (self.constraint_check_by_Py, 'def Check_sol (X):\n')
            printf (self.constraint_check_by_Py, '\t"""\n\tCheck whether a solution for the LP problem satisfies all the constraints\n\t"""\n')
        
        if (self.write_to_mod_file):
            printf (self.mod_output_file, 'subject to {\n\n')
        
        self.gen_leq1_constraints ()
        self.gen_single_alloc_constraints ()
        self.gen_cpu_cap_constraints ()
        self.calc_paths_of_links ()
        self.gen_link_cap_constraints ()
        self.gen_chain_delay_constraints ()
        if (self.write_to_prb_file):
            printf (self.prb_output_file, '\nend;\n')
        if (self.write_to_py_file):
            printf (self.constraint_check_by_Py, '\n\n\treturn True\n')
        if (self.write_to_mod_file):
            printf (self.mod_output_file, '\n}\n\n')

    def gen_leq1_constraints (self):
        """
        Print the var' ranges constraints: each var should be <=1  
        """
        if (self.write_to_prb_file):
            for __ in self.n:
                printf (self.prb_output_file, 'subject to X_leq1_C{}: 1*X{} <= 1;\n\n' .format (self.constraint_num, __['id'], __['id']) )
                
                # The Python script doesn't really need these constraints, as anyway it checks only binary values for the decision variables. Hence, it's commented out.
                # printf (self.constraint_check_by_Py, '\tif (X[{}] > 1):\n\t\treturn False\n\n' .format (__['id']))
                self.constraint_num += 1
            printf (self.prb_output_file, '\n')
        
        elif (self.write_to_lp_file):
            printf (self.lp_output_file, '\n\n\ single chain allocation constraints\n')
            id = 0  
            for chain_num in range (self.NUM_OF_CHAINS):
                printf (self.lp_output_file, '\n c{}: ' .format (self.constraint_num))
                is_first = True
                while (id < self.last_id_in_chain[chain_num]):
                    if (is_first):
                        is_first = False
                    else:
                        printf (self.lp_output_file, ' + ')
                    printf (self.lp_output_file, 'x{}' .format (id)) 
                    id += 1
                printf (self.lp_output_file, ' = 1')
                self.constraint_num += 1
        

    
    def gen_chain_delay_constraints (self):
        """
        Print the constraints of maximum delay of each chain in a LP format
        """
        
        if (self.write_to_mod_file):
            printf (self.mod_output_file, '\t// Chain delay constraints\n')

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
        if (self.write_to_prb_file):
            printf (self.prb_output_file, '\n')
        if (self.write_to_mod_file):
            printf (self.mod_output_file, '\t //Link capacity constraints\n')
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
        self.prb_output_file - write to this file the inequality in a Linear-Prog. format, e.g.: 
            3*X1 + 2*X2 <= 5
        self.constraint_check_by_Py - write to this file the inequality as a Python-code that returns false if the inequlity isn't satisfied, e.g.
            if (3*X[1] + 2*X[2] > 5):
                return False
                  
        """
          
        if (self.write_to_prb_file):
            printf (self.prb_output_file, 'subject to {}_C{}: ' .format (constraint_name, self.constraint_num))
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
                        printf (self.prb_output_file, '+ ')
                    printf (self.prb_output_file, '{:.4f}*X{}*X{} ' .format (
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
                    printf (self.prb_output_file, '+ ')
                else:
                    is_first = False
                     
                printf (self.prb_output_file, '{:.4f}*X{} ' .format (
                    list_of_coefs_in_lin_eq         [decision_var_idx],  
                    list_of_decision_vars_in_lin_eq [decision_var_idx]))
                    
            printf (self.prb_output_file, '<= {};\n\n' .format (constant))
          
  
        if (self.write_to_py_file):
            printf (self.constraint_check_by_Py, '\t# {}_C{}:\n\tif (' .format (constraint_name, self.constraint_num))
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
                        printf (self.constraint_check_by_Py, '+ ')
                    printf (self.constraint_check_by_Py, '{:.4f}*X[{}]*X[{}] ' .format (
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
                    printf (self.constraint_check_by_Py, '+ ')
                else:
                    is_first = False
                     
                printf (self.constraint_check_by_Py, '{:.4f}*X[{}] ' .format (
                    list_of_coefs_in_lin_eq         [decision_var_idx],  
                    list_of_decision_vars_in_lin_eq [decision_var_idx]))
                    
            printf (self.constraint_check_by_Py, '> {}):\n\t\treturn False\n\n' .format (constant))
  
        if (self.write_to_mod_file):
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
                        printf (self.mod_output_file, '+ ')
                    printf (self.mod_output_file, '{:.4f}*X[{}]*X[{}] ' .format (
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
                    printf (self.mod_output_file, '+ ')
                else:
                    is_first = False
                     
                printf (self.mod_output_file, '{:.4f}*X[{}] ' .format (
                    list_of_coefs_in_lin_eq         [decision_var_idx],  
                    list_of_decision_vars_in_lin_eq [decision_var_idx]))
                    
            printf (self.mod_output_file, '<= {};\n\n' .format (constant))

    def gen_link_cap_constraints_to_lp (self):
        """
        Print link capacity constraints to a .lp cplex format file.
        This function uses the "r" decision variables, in which each single variable determines both the location (servers)
        and the allocation (CPU capacities) of a full chain.
        """
        printf (self.lp_output_file, '\n\n\ Link capacity constraints\n\n')
        
        for l in self.list_of_links:
            list_of_decision_vars_in_link_l_eq = [] # The decision vars that will appear in the relevant lin' constraint 
            list_of_coefs_in_link_l_eq         = [] # coefficients of the decision vars that will appear in the relevant lin' constraint           
            
            for list_of_paths_using_link_l in list (filter (lambda item : item['link'] == l, self.paths_of_link)):
                list_of_paths_using_link_l = list_of_paths_using_link_l['paths'] 

            coef = np.zeros (len(self.dynamic_r)) #coef[i] will hold the coefficient of the variable with id==i in dynami_r
            done = np.zeros (len(self.dynamic_r), dtype = 'bool') # done[i] will be true iff we already calculated the coef' of dynamic_r with id == i for this link's cap' constraint Eq.  
            for r_dict in self.dynamic_r:
                
                id        = r_dict['id']
                if (done[id]):
                    continue
                chain_num = r_dict['chain_num']
                chain_len = self.num_of_vnfs_in_chain[chain_num]

                # Consider the BW requirements of all the paths along the scheduled chain's locations that use link l
                for i in range (chain_len-1):
                    if [r_dict['location'][i], r_dict['location'][i+1]]  in list_of_paths_using_link_l:
                        coef[id] += self.traffic_in [self.vnf_in_chain[chain_num][i+1]]

                # Check wether the scheduled path from the PoA to the first VM in the chain is using link l  
                if ( [self.PoA_of_user[chain_num], r_dict['location'][0]] in list_of_paths_using_link_l): # if the path from the PoA to the first VM in the chain uses link l
                    coef[id] += self.traffic_in[self.vnf_in_chain[chain_num][0]]

                # Consider the scheduled path from the last VM in the chain the PoA  is using link l  
                if ( [r_dict['location'][-1], self.PoA_of_user[chain_num]] in list_of_paths_using_link_l): # if the scheduled path from the last VM in the chain the PoA is using link l  
                    coef[id] += self.traffic_out_of_chain[chain_num]

                # Now account for migrations
                indices_of_migrating_VMs = ( [i for i in range (chain_len) if self.cur_loc_of_vnf [self.vnf_in_chain[chain_num][i]] != r_dict['location'][i] ] ) # indices_of_migrating_VMs will hold a list of the indices of the VMs in that chain, that are scheduled to migrate
                                           

                for i in indices_of_migrating_VMs: # for every VM in the chain
                    v = self.vnf_in_chain[chain_num][i] # refer to this VNF as v 
                    if ([self.cur_loc_of_vnf [v], r_dict['location'][i]] in list_of_paths_using_link_l): # if the path between this VM's current location, and scheduled location, uses link l. 
                        coef[id] += self.mig_bw[v] # Consider the mig' bw
                    if (i < chain_len-1 and ([self.cur_loc_of_vnf [v], self.cur_loc_of_vnf[self.vpp[v]]] in list_of_paths_using_link_l)): # if v is scheduled to migrate, and the path from v to v++ in the current location uses link l  
                        coef[id] += self.traffic_in [self.vpp[v]]

                if (0 in indices_of_migrating_VMs and # if the first VM in the chain migrated
                   ( [self.PoA_of_user[chain_num], self.cur_loc_of_vnf[self.vnf_in_chain[chain_num][0]]] in list_of_paths_using_link_l)): # if its old path uses link l
                    coef[id] += self.traffic_in [self.vnf_in_chain[chain_num][0]]
                    
                if (chain_len-1 in indices_of_migrating_VMs and # if the last VM in the chain migrated
                   ( [self.cur_loc_of_vnf[self.vnf_in_chain[chain_num][-1]], self.PoA_of_user[chain_num]] in list_of_paths_using_link_l)): # if its old path uses link l
                    coef[id] += self.traffic_out_of_chain [chain_num]
                    
                # All r decision variables suggesting the same location for this chain (regardless of the different allocation values)
                # will have identical coefficients. So assign these coefficients, and mark them as 'done'     
                for r_with_identical_location in (list (filter (lambda item : 
                    item['chain_num']==r_dict['chain_num'] and (item['location'] == r_dict['location']).all(), self.dynamic_r))):
                    coef[r_with_identical_location['id']] = coef[id]
                    done[r_with_identical_location['id']] = True
            
                    
            
            # Now print the inequality constraint for this link
            
            list_of_non_zero_coefs = ( [i for i in range (len(coef)) if coef[i] != 0]  )
            
            if (len (list_of_non_zero_coefs) == 0): # No non-zeros coef's, namely, no one uses / will be using this link 
                continue
            
            is_first = True
            for id in list_of_non_zero_coefs:

                if (is_first):
                    printf (self.lp_output_file, ' c{}_link_{}_{}: ' .format (self.constraint_num, l[0], l[1]))
                    self.constraint_num += 1
                    is_first = False
                else:
                    printf (self.lp_output_file, ' + ')
                printf (self.lp_output_file, '{}x{}' .format (coef[id], id)) 
            
            printf (self.lp_output_file, ' <= {}\n\n' .format (self.capacity_of_link[l[0]][l[1]]))

            
    def gen_cpu_cap_constraints (self):
        """
        Print the constraints of maximum server's CPU capacity in various formats
        """

        if (self.write_to_prb_file):
            printf (self.prb_output_file, '\n')        
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
                        printf (self.prb_output_file, 'subject to max_cpu_C{}: {}*X{} ' .format (self.constraint_num, item['coef'], item['id']))
                        self.constraint_num += 1
                        is_first = False
                    else: 
                        coef = item['coef']
                        sign = '+' if (coef > 0) else '-'
                        abs_coef = abs(coef)
                        printf (self.prb_output_file,          '{} {}*X{} ' .format (sign, abs_coef, item['id']))
                printf (self.prb_output_file, ' <= {};\n' .format (server_s_available_cap))

        if (self.write_to_py_file):
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
                        printf (self.constraint_check_by_Py, '\t#max_cpu_C{}\n\tif ({}*X[{}] ' .format (self.constraint_num, item['coef'], item['id']))
                        self.constraint_num += 1
                        is_first = False
                    else: 
                        coef = item['coef']
                        sign = '+' if (coef > 0) else '-'
                        abs_coef = abs(coef)
                        printf (self.constraint_check_by_Py, '{} {}*X[{}] '  .format (sign, abs_coef, item['id']))
                printf (self.constraint_check_by_Py, ' > {}):\n\t\treturn False\n\n' .format (server_s_available_cap))


        if (self.write_to_mod_file):
            printf (self.mod_output_file, '\t// CPU capacity constraints\n')
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
                        printf (self.mod_output_file, '\t{}*X[{}] ' .format (self.constraint_num, item['coef'], item['id']))
                        self.constraint_num += 1
                        is_first = False
                    else: 
                        coef = item['coef']
                        sign = '+' if (coef > 0) else '-'
                        abs_coef = abs(coef)
                        printf (self.mod_output_file, '{} {}*X[{}] '  .format (sign, abs_coef, item['id']))
                printf (self.mod_output_file, ' <= {};\n' .format (server_s_available_cap))
            printf (self.mod_output_file, '\n')
        
        if (self.write_to_lp_file):
            printf (self.lp_output_file, '\n\n\ CPU capacity constraints\n\n')

            for s in range (self.NUM_OF_SERVERS):
                is_first = True
                
                # decision_vars_using_this_server = list (filter (lambda item: s in item['location'], self.dynamic_r))
                for r_dict in self.dynamic_r: # for each decision var' 
                    chain_num   = r_dict['chain_num']
                    coef = 0 # coefficient of the current decision var' in the current CPU cap' equation
                    for i in range (len (r_dict['location'])): # for every VM in the chain scheduled by this decision var
                        # v = self.vnf_in_chain[chain_num][i]  
                        if (r_dict['location'][i] == s): # if this VM is scheduled to use s...
                            coef += r_dict['alloc'][i]      # add the scheduled allocation to the coef'
                        elif (self.cur_loc_of_vnf [self.vnf_in_chain[chain_num][i]] == s): # this VM isn't scheduled to use s, but currently it's using s
                            coef += self.cur_cpu_alloc_of_vnf[self.vnf_in_chain[chain_num][i]]

                    if (coef == 0):
                        continue

                    if (is_first):
                        printf (self.lp_output_file, ' c{}: ' .format (self.constraint_num))
                        self.constraint_num += 1
                        is_first = False
                    else:
                        printf (self.lp_output_file, ' + ')
                    printf (self.lp_output_file, '{}x{}' .format (coef, r_dict['id'])) 
                
                printf (self.lp_output_file, ' <= {}\n\n' .format (self.cpu_capacity_of_server[s]))
                    

    def gen_single_alloc_constraints (self):
        """
        Print the constraint of a single allocation for each VM in a LP format
        """
        if (self.write_to_prb_file):
            v = -1 
            for item in self.n:
                if (item['v'] == v): #Already seen decision var' related to this VM
                    printf (self.prb_output_file, '+ X{}' .format (item['id']))
                else: # First time observing decision var' related to this VM
                    if (v > -1):
                        printf (self.prb_output_file, ' = 1;\n' )
                    printf (self.prb_output_file, 'subject to single_alloc_C{}:   X{} ' .format (self.constraint_num, item['id'])) 
                    v = item['v']
                self.constraint_num += 1
            printf (self.prb_output_file, ' = 1;\n\n' )
               
        if (self.write_to_py_file):
            v = -1 
            for item in self.n:
                if (item['v'] == v): #Already seen decision var' related to this VM
                    printf (self.constraint_check_by_Py, '+ X[{}]' .format (item['id']))
                else: # First time observing decision var' related to this VM
                    if (v > -1):
                        printf (self.constraint_check_by_Py, ' == 1)):\n\t\treturn False\n' )
                    printf (self.constraint_check_by_Py, '\tif (not (X[{}] ' .format (item['id']))
                    v = item['v']
                self.constraint_num += 1
            printf (self.constraint_check_by_Py, ' == 1)):\n\t\treturn False\n\n' )

        if (self.write_to_mod_file):
            printf (self.mod_output_file, '\t// Single allocation constraints\n\t')
            v = -1 
            for item in self.n:
                if (item['v'] == v): #Already seen decision var' related to this VM
                    printf (self.mod_output_file, '+ X[{}]' .format (item['id']))
                else: # First time observing decision var' related to this VM
                    if (v > -1):
                        printf (self.mod_output_file, ' == 1;\n\t' )
                    printf (self.mod_output_file, 'X[{}] ' .format (item['id']))
                    v = item['v']
                self.constraint_num += 1
            printf (self.mod_output_file, ' == 1;\n\n')

    def print_vars (self):
        """
        Print the decision variables. Each variable is printed with the constraints that it's >=0  
        """
        if (self.write_to_prb_file):
            for __ in self.n:
                printf (self.prb_output_file, 'var X{} >= 0;\n' .format (__['id'], __['id']) )
            printf (self.prb_output_file, '\n')
        if (self.write_to_mod_file):
            printf (self.mod_output_file, 'int num_of_dvars = {};\n' .format (len(self.n)))
            printf (self.mod_output_file, 'range dvar_indices = 0..num_of_dvars-1;\n')
            printf (self.mod_output_file, 'dvar boolean X[dvar_indices];\n\n')

    def gen_obj_function (self):
        """
        Print the objective function in a standard LP form (linear combination of the decision variables)
        """
        if (self.write_to_prb_file):
            printf (self.prb_output_file, 'minimize z:   ')
            is_first_item = True
            for item in self.n:
                if (not (is_first_item)):
                    printf (self.prb_output_file, ' + ')
                printf (self.prb_output_file,            '{:.4f}*X{}' .format (item['cost'], item ['id']) ) 
                is_first_item = False
            printf (self.prb_output_file, ';\n\n')

        if (self.write_to_py_file):
            printf (self.obj_func_by_Py, 'def obj_func (X):\n')
            printf (self.obj_func_by_Py, '\t"""\n\tCalculate the objective function, given a feasible solution.\n\t"""\n\treturn ')
            is_first_item = True
            for item in self.n:
                if (not (is_first_item)):
                    printf (self.obj_func_by_Py, ' + ')
                printf (self.obj_func_by_Py,   '{:.4f}*X[{}]' .format (item['cost'], item ['id']) ) 
                is_first_item = False
            printf (self.obj_func_by_Py, '\n')

        if (self.write_to_mod_file):
            printf (self.mod_output_file, 'minimize ')
            is_first_item = True
            for item in self.n:
                if (not (is_first_item)):
                    printf (self.mod_output_file, ' + ')
                printf (self.mod_output_file,   '{:.4f}*X[{}]' .format (item['cost'], item ['id']) ) 
                is_first_item = False
            printf (self.mod_output_file, ';\n\n')
   
    def lp_sol_to_loc_alloc (self, list_of_set_vars):
        """
        Translate a sol' to the optimization prob', given as a list of ids of the set decision binary vars "r_dynamic" set, and translates it to:
        lp_nxt_loc_of_vnf[v]     - will be s iff VNF v is scheduled to server s
        lp_nxt_cpu_loc_of_vnf[v] - will be a iff a is assigned a units of CPU
        """
        
        self.lp_nxt_loc_of_vnf         = np.empty (self.NUM_OF_VNFs, dtype = np.uint16)
        self.lp_nxt_cpu_alloc_of_vnf   = np.empty (self.NUM_OF_VNFs, dtype = np.uint16)
        

        for id in list_of_set_vars:
            for item in list (filter (lambda item : item['id'] == id, self.dynamic_r) ):
                chain_num = item['chain_num']
                for idx_in_chain in range (self.num_of_vnfs_in_chain[chain_num]):                      
                    self.lp_nxt_loc_of_vnf       [self.vnf_in_chain[chain_num][idx_in_chain]] = item['location'][idx_in_chain]
                    self.lp_nxt_cpu_alloc_of_vnf [self.vnf_in_chain[chain_num][idx_in_chain]] = item['alloc']   [idx_in_chain]
                
    def n_vsa_sol_to_loc_alloc (self, sol):
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
            
             
    def gen_static_r (self):
        """
        Generate the r vector decision variable.
        r (H, vec(D), vec(mu)) will indicate that a solution implies allocating VM[j] of chain H on D[j], with CPU allocation mu[j].
        r is a list of dicts, where each dictionary includes the fields:
        - my_chain - the chain of this var.
        - locations - a list (possibly) with repetitions of |H| servers, where locations[i] is destined to host VM i in this chain. 
        - alloc - a list of |H| integers, where alloc[i] is the CPU allocated for host VM i in this chain.
        - static_cost - computation cost for using this allocation, independently of the current state (namely, no mig' cost is counted). 
        - static delay - The delay along the chain + computation delay. The only thing missing for the full chain delay is the delay to / from the PoA, which is unknown statically.
        """         
        self.static_r = []
        self.ids_of_y_vs = [] # will hold the IDs of all the n_vsa decision vars related to server s and VM v 
        
        for chain_num in range (self.NUM_OF_CHAINS):
            
            chain_len = self.num_of_vnfs_in_chain[chain_num]
            min_loc_vals = np.zeros (chain_len, dtype = 'uint16') # minimal possible values for servers' locations: allocate all VMs in this chain in server 0  
            max_loc_vals = np.ones  (chain_len, dtype = 'uint16') * self.NUM_OF_VNFs # maximal value for the loc' var, namely a vector which implies that all VMs in this chain are allocated to the highest-idx server.  
            locations    = max_loc_vals.copy () # Will be reset to the first location to examine upon the first iteration 
            while True:
               
                
                locations = self.inc_array (locations  , min_loc_vals, max_loc_vals)
               
                # static_netw_delay will hold the netw delay along the whole suggested chain's location, from the 1st to the last VM in the chain.
                static_netw_delay = sum (self.servers_path_delay [locations[i]] [locations[i+1]] for i in range (chain_len-1))
                
                if (self.verbose == 1):
                    printf (self.debug_output_file, '\n\nlocations = {}\n********************' .format (locations))

                if (static_netw_delay > self.chain_target_delay[chain_num]): # skip suggested alloc with too long chain delay.
                    if (self.verbose == 1):
                        printf (self.debug_output_file, 'Infeasible netw delay\n' .format (locations))
                    if (np.array_equal (locations, max_loc_vals)): # finished iterating over all possible locations
                        break
                    continue

                # dominant_alloc will hold a list of dominant alloc for the suggested locations. If an allocation is subdominant, we don't need to consider it at all.
                # For instance, if the allocation 
                # dominant_alloc = []                                    
                min_alloc_vals = np.asarray ([math.ceil(self.theta_times_traffic_in_chain[chain_num][i]) for i in range (chain_len)], dtype = 'uint8') # Built-in application of the finite computation delay constraint: the CPU allocation must be at least theta[v] * traffic_in[v]
                max_alloc_vals = np.array   ([self.cpu_capacity_of_server[locations[i]] for i in range (chain_len)], dtype = 'uint8')
                alloc          = max_alloc_vals.copy () # Will be reset to the minimal allocation to examine upon the first iteration

                while True:

                    alloc = self.inc_array (alloc, min_alloc_vals, max_alloc_vals)

                    if (self.verbose == 1):
                        printf (self.debug_output_file, '\nalloc = {} ' .format (alloc))

                    static_delay = sum ( [1 / (alloc[i] - self.theta_times_traffic_in_chain[chain_num][i]) for i in range (chain_len)]) + static_netw_delay
                    if (static_delay > self.chain_target_delay[chain_num]):
                        if (self.verbose == 1):
                            printf (self.debug_output_file, 'infeasible static delay' .format (alloc))
                        if (np.array_equal (alloc, max_alloc_vals)):  # finished iterating over all possible alloc
                            break
                        continue
                    
                    # Discard suggested alloc that disobey the static cpu consumption constraints
                    broke_CPU_cap_constraint = False
                    for s in np.unique(locations): # for each server scheduled to use at least one of the VMs in this chain
                        if (sum ([alloc[i] for i in ([j for j in range (chain_len) if locations[j] == s])]) > self.cpu_capacity_of_server[s]):
                            broke_CPU_cap_constraint = True
                            if (self.verbose == 1):
                                printf (self.debug_output_file, 'infeasible cpu capacity' .format (alloc))
                            break
                    
                    if (broke_CPU_cap_constraint):
                        if (np.array_equal (alloc, max_alloc_vals)):  # finished iterating over all possible alloc
                            break
                        continue 
                        
                    # Now we know that this allocation is statically feasible
                    
                        
                    self.static_r.append (
                        {
                        'chain_num'     : chain_num,
                        'location'     : locations.copy (),
                        'alloc'   : alloc.copy(),
                        'static delay'  : static_delay,
                        'static cost'   : sum (alloc[i]                                                           for i in range (chain_len))                
                        }
                    )                

                    if (np.array_equal (alloc, max_alloc_vals)):  # finished iterating over all possible alloc
                        break
                if (np.array_equal (locations, max_loc_vals)): # finished iterating over all possible locations
                    break
        
    def calc_dynamic_r (self):
        """

        Add to the decision variable r the dynamic parts, that is, the parts relating to the mig and the updated PoA
        """
        id = int (0)
        self.dynamic_r = []
        self.last_id_in_chain = np.zeros (self.NUM_OF_CHAINS, dtype = 'uint16')

        if (self.write_to_lp_file):
            printf (self.lp_output_file, 'Minimize\n obj: ')

        for chain_num in range (self.NUM_OF_CHAINS):
            
            chain_len = self.num_of_vnfs_in_chain[chain_num]
            
            cur_loc = np.array ( [ self.cur_loc_of_vnf [self.vnf_in_chain[chain_num][i]] for i in range (chain_len)])
            
            for r_dict in (list (filter (lambda item : item['chain_num'] == chain_num, self.static_r))):
                
                delay = r_dict['static delay'] + \
                        self.servers_path_delay [self.PoA_of_user [chain_num]] [r_dict['location'][0]] + \
                        self.servers_path_delay [r_dict['location'][-1]]      [self.PoA_of_user [chain_num]]
                
                # Discard decision variables that disobey the chain delay constraint 
                if (delay > self.chain_target_delay[chain_num]):  
                    if (self.verbose == 1):
                        printf (self.debug_output_file, '\nlocation = {}, alloc = {}. infeasible dynamic delay: {:.2f}' .format (r_dict['location'], r_dict['alloc'], delay))
                    continue   

                # Discard decision variables that disobey the CPU capacity constraints 
                for s in range (self.NUM_OF_SERVERS):
                    broke_CPU_cap_constraint = False
                    
                    # decision_vars_using_this_server = list (filter (lambda item: s in item['location'], self.dynamic_r))
                    coef = 0 # coefficient of the current decision var' in the current CPU cap' equation
                    for i in range (len (r_dict['location'])): # for every VM in the chain scheduled by this decision var
                        if (r_dict['location'][i] == s): # if this VM is scheduled to use s...
                            coef += r_dict['alloc'][i]      # add the scheduled allocation to the coef'
                        elif (self.cur_loc_of_vnf [self.vnf_in_chain[chain_num][i]] == s): # this VM isn't scheduled to use s, but currently it's using s
                            coef += self.cur_cpu_alloc_of_vnf[self.vnf_in_chain[chain_num][i]]
                    if (coef > self.cpu_capacity_of_server[s]): # even this decision var' alone, without other decision vars', requires too high CPU capacity from server s
                        broke_CPU_cap_constraint = True
                        break
                    
                if (broke_CPU_cap_constraint):
                    continue 

                cost = r_dict['static cost'] 
                #indices_of_migrating_VMs = ( [i for i in range (chain_len) if cur_loc [self.vnf_in_chain[chain_num][i]] != r_dict['location'][i] ] ) # indices_of_migrating_VMs will hold a list of the indices of the VMs in that chain, that are scheduled to migrate

                for i in [i for i in range (chain_len) if cur_loc [self.vnf_in_chain[chain_num][i]] != r_dict['location'][i] ]: #for every i in the list of the indices of the VMs in that chain, that are scheduled to migrate
                    cost += self.mig_cost[self.vnf_in_chain[chain_num][i]]
                
                if (self.write_to_lp_file):
                    if (id > 0):
                        printf (self.lp_output_file, ' + ')
                    printf (self.lp_output_file, '{}x{}' .format (cost, id))
    
                    #r_dict['migs src dst pairs'] = ( [ [cur_loc[i], r_dict['location'][i]] for i in range (chain_len) if cur_loc[i] != r_dict['location'][i] ] ) 
                
                if (self.verbose == 1):
                    printf (self.debug_output_file, '\nid = {}, location = {}, alloc = {}, cost = {}' 
                            .format (id, r_dict['location'], r_dict['alloc'], cost))
                    
                self.dynamic_r.append (
                    {
                    'id'            : id,
                    'chain_num'     : chain_num,
                    'location'     : r_dict['location'].  copy (),
                    'alloc'         : r_dict['alloc'].copy(),
                    'delay'         : delay,
                    #'indices_of_migrating_VMs' : indices_of_migrating_VMs,
                    }
                )                
                
                id += 1
            
            self.last_id_in_chain[chain_num] = id
            
        
        
    def gen_n (self):
        """"
        Generate the n vector indication variables.
        n(v,s,a) == 1 will indicate allocation VM v on server s with a CPU units. 
        Each entry in the list "n" will be a dictionary, containing the fields:
        - ID
        - v, s, a of this decision var
        - mig - will be True iff such an allocation implies a migration of VM v
        - comp_delay - the computation delay implied by using this var
        - cost - to be used later, when formulating the LP
        """

        self.n = [] # "n" decision variable, defining the scheduled server and CPU alloc' of each VM.
        self.ids_of_y_vs = [] # will hold the IDs of all the n_vsa decision vars related to server s and VM v 
        id = int (0)
        
        # Loop over all combinations of v, s, and a
        for v in range (self.NUM_OF_VNFs):
            for s in range (self.NUM_OF_SERVERS):
                mig = True if (s != self.cur_loc_of_vnf[v]) else False # if s is s different than v's current location, this implies a mig'
                list_of_ids = []
                for a in range (math.ceil (self.theta_times_traffic_in[v]), self.cpu_capacity_of_server[s]+1): # skip too small values of a (capacity alloc), which cause infinite comp' delay
                    denominator = a - self.theta_times_traffic_in[v] 
                    if (denominator <= 0): # skip too small values of a (capacity alloc), which cause infinite comp' delay
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
     
     
     
    def inc_array (self, ar, min_val, max_val):
        """
        input: an array, in which elements[i] is within [min_val[i], max_val[i]] for each i within the array's size
        output: the same array, where the value is incremented by 1 
        """
        for idx in range (ar.size-1, -1, -1):
            if (ar[idx] < max_val[idx]):
                ar[idx] += 1
                return ar
            ar[idx] = min_val[idx]
        return ar 
     
    def brute_force_sa_pow_v (self):
        """
        Brute-force check of all combination which let each VM use exactly one server, and exactly a single value of CPU power.
        That is, each VM selects exactly 1 of its s*a relevant entries in self.n to be 1.
        """
        # Each VNF can have exactly one related decision var set.
        # choice_of_VNF_v[v] will be set iff in the current checked sol
         
        sa                      = self.NUM_OF_SERVERS * self.uniform_cpu_capacity # s times a
        vsa                     = sa * np.arange (self.NUM_OF_VNFs) 
        ar_of_min_vals             = np.zeros (self.NUM_OF_VNFs, dtype = 'uint16')
        ar_of_max_vals          = np.ones  (self.NUM_OF_VNFs, dtype = 'uint16') * (sa-1)
        
        choice_of_VNF_v         = ar_of_min_vals.copy () 
        num_of_decision_vars    = len (self.n)
                                                        
        for __ in range (sa ** self.NUM_OF_VNFs): 
        
            sol = np.zeros (num_of_decision_vars)
            for v in range (self.NUM_OF_VNFs):
                sol[vsa[v] + choice_of_VNF_v[v]] = 1
            if (Check_sol.Check_sol (sol)): # if Solve is not feasible
                cost = obj_func.obj_func (sol)
                if (cost < self.min_cost):
                    self.min_cost = cost
                    self.best_n = sol
                
            choice_of_VNF_v = self.inc_array (choice_of_VNF_v, ar_of_min_vals, ar_of_max_vals) 
        

    def brute_force_by_n (self):
        """
        Perform a brute-force check of all combinations of the decision var's n.
        There're v*s*a of entries for n, henceforth there're 2^{v * s * a} possible combinations.
        """
        self.min_cost = float ('inf')
        for sol in itertools.product ([0,1],repeat = len (self.n)):
            if ( not (Check_sol.Check_sol (sol)) ):
                continue
            cost = obj_func.obj_func (sol)
            if (cost < self.min_cost):
                self.min_cost = cost
                self.best_n = sol

    def print_non_zero_indices_in_best_sol (self):
        """
        Print a list of the indices of "1"s in the best sol' found
        """

    
    
    def print_sol_to_tex (self, running_parameter):
        """
        Print the solution found to a .tex table
        """
        if (self.min_cost == float ('inf')):
            print ('Did not find a feasible sol')
            printf (self.res_output_file, '\t{:.0f} & N/A & No feasible solution \\tabularnewline \hline \n' .format(running_parameter))
            return
        
        self.n_vsa_sol_to_loc_alloc (self.best_n)
        printf (self.res_output_file, '\t{:.1f} & {:.1f} & VM loc = {} cpu alloc = {} \\tabularnewline \hline \n' .format 
                (running_parameter, self.min_cost,  self.nxt_loc_of_vnf, self.nxt_cpu_alloc_of_vnf))
        

if __name__ == "__main__":
    my_toy_example = toy_example (verbose = 1)
    my_toy_example.run (chain_target_delay = 5, gen_LP = True,  run_brute_force = False) # Generate code for the LP
    # if (len(sys.argv) > 2):  # run a parameterized sim'   
    #     chain_target_delay = float (str(sys.argv[2]))
    #         
    #     if (str(sys.argv[1])=="G"): # G --> Gen problem. "S" --> Solve problem
    #         my_toy_example.run (chain_target_delay = chain_target_delay, gen_LP = True,  run_brute_force = False) # Generate code for the LP
    #     else:
    #         my_toy_example.run (chain_target_delay = chain_target_delay, gen_LP = False, run_brute_force = True)  # Brute-force solve the LP
    # else:
    #     if (str(sys.argv[1])=="G"): # G --> Gen problem. "S" --> Solve problem
    #         my_toy_example.run (gen_LP = True,  run_brute_force = False) # Generate code for the LP
    #     else:
    #         my_toy_example.run (gen_LP = False, run_brute_force = True)  # Brute-force solve the LP
    # 
    # alloc = [3, 1, 2]
    # traffic_in = [1, 0.5, 1]
    # 
    # comp_delay = sum ([1 / (alloc[i] * traffic_in[i]) for i in range (3)])
    # 
    # print ('comp_delay = ', comp_delay)
    
