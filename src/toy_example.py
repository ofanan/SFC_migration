import networkx as nx
import numpy as np
import math
import itertools 

from printf import printf
from LP_file_parser import LP_file_parser

class toy_example (object):
    
    def gen_custom_three_nodes_tree (self):
        """
        generate a custom three-nodes tree (root and 2 leaves). 
        """
        self.NUM_OF_SERVERS = 3
        self.NUM_OF_USERS = 2
        self.NUM_OF_PoA = 2
    
        self.list_of_links = [ [0,1], [1,0], [1,2], [2,1] ]
        #servers_path_delay[i][j] holds the netw' delay of the path from server i to server j
        self.servers_path_delay  = [
                                   [0, 1, 2],
                                   [1, 0, 1],
                                   [2, 1, 0]
                                  ]
    
        
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

        self.NUM_OF_LINKS = 4
        self.capacity_of_link       = np.zeros ( (self.NUM_OF_SERVERS, self.NUM_OF_SERVERS))
        self.capacity_of_link[0][1] = self.uniform_link_capacity
        self.capacity_of_link[1][0] = self.uniform_link_capacity
        self.capacity_of_link[1][2] = self.uniform_link_capacity
        self.capacity_of_link[2][1] = self.uniform_link_capacity
        self.NUM_OF_CHAINS          = self.NUM_OF_USERS
        self.PoA_of_user            = np.zeros (self.NUM_OF_USERS) # np.random.randint(self.NUM_OF_PoA, size = self.NUM_OF_USERS) # PoA_of_user[u] will hold the PoA of the user using chain u       
        self.output_file            = open ("../res/custom_tree.res", "a")
        self.LP_output_file         = open ("../res/custom_tree.LP", "a")

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
        self.uniform_link_capacity   = 3
        for edge in self.G.edges:
            self.capacity_of_link[edge[0]][edge[1]] = self.uniform_link_capacity
            self.capacity_of_link[edge[1]][edge[0]] = self.uniform_link_capacity
            self.list_of_links.append ([edge[0], edge[1]])

        self.NUM_OF_CHAINS = self.NUM_OF_USERS
       
        # The Points of Accs will make the first rows in the path_costs matrix
        self.PoA_of_user            = [0,0] #np.random.randint(self.NUM_OF_PoA, size = self.NUM_OF_USERS) #[0,0]
        
        self.output_file            = open ("../param_tree.res", "a")
        self.LP_output_file         = open ("../param_tree.LP", "a")

    def __init__ (self, verbose = -1):
        
        self.verbose = verbose
        self.uniform_link_capacity  = 100

        use_custom_netw = True
        if (use_custom_netw == True):
            self.gen_custom_three_nodes_tree()
        else:
            self.gen_parameterized_tree()
            
        self.num_of_vnfs_in_chain   = np.ones (self.NUM_OF_USERS, dtype='uint8')
        self.NUM_OF_VNFs            = sum (self.num_of_vnfs_in_chain).astype ('uint')

        self.cur_loc_of_vnf         = [2, 1] # np.random.randint(self.NUM_OF_SERVERS, size = self.NUM_OF_VNFs) # Initially, allocate VMs on random VMs
        self.cur_cpu_alloc_of_vnf   = [2, 1] #2 * np.ones (self.NUM_OF_VNFs)                                  # Initially, allocate each VNs uniform amount CPU units

        self.mig_cost               = np.ones (self.NUM_OF_SERVERS) # np.random.rand (self.NUM_OF_SERVERS)         
        self.uniform_cpu_capacity   = 3
        self.cpu_capacity_of_server = self.uniform_cpu_capacity * np.ones (self.NUM_OF_SERVERS, dtype='int8')     
        self.theta                  = 0.5 * np.ones (self.NUM_OF_VNFs) #cpu units to process one unit of data
        self.traffic_in             = np.ones (self.NUM_OF_VNFs+1) #traffic_in[v] is the bw of v's input traffic ("\lambda_v"). traffic_in[-1] will hold the user's input traffic, which is also the output traffic of the last VNF in the chain.
        self.theta_times_traffic_in = self.theta * self.traffic_in [0:self.NUM_OF_VNFs-1]

        self.nxt_loc_of_vnf         = np.array (self.NUM_OF_VNFs)   # nxt_loc_of_vnf[v] will hold the id of the server planned to host VNF v
        self.nxt_cpu_alloc_of_vnf   = np.array (self.NUM_OF_VNFs)
        # self.VM_target_delay           = 10 * np.ones (self.NUM_OF_VNFs)    # the desired (max) delay (aka Delta). Currently unused
        # self.perf_deg_of_vnf        = np.zeros (self.NUM_OF_VNFs). Currently unused
        
        # Calculate v^+ of each VNF v.
        # vpp(v) will hold the idx of the next VNF in the same chain.
        # if v is the last VNF in its chain, then vpp(v) will hold the PoA of this chain's user  
        self.vpp                    = np.zeros (self.NUM_OF_VNFs, dtype = 'uint')
        self.v0                     = [] # self.v0 will hold a list of all the VNFs which are first in their chain
        self.v_inf                  = [] # self.v_inf will hold a list of all the VNFs which are last in their chain
        self.PoA_of_vnf = np.zeros (self.NUM_OF_VNFs, dtype = 'uint') # self.PoA_of_vnf[v] will hold the PoA of the user using VNF v
        v = 0
        for chain in range (self.NUM_OF_CHAINS):
            for idx_in_chain in range (self.num_of_vnfs_in_chain[chain]):
                if (idx_in_chain == 0):
                    self.v0.append(v)
                if (idx_in_chain == self.num_of_vnfs_in_chain[chain]-1): # Not "elif", because in the case of a single-VM chain, the first is also the last
                    self.v_inf.append (v)
                    self.vpp [v] = self.PoA_of_user[chain]
                else: # Not the last VM in the chain
                    self.vpp [v] = v+1 
                self.PoA_of_vnf [v] = self.PoA_of_user[chain]    
                v += 1
       
        # self.mig_comp_delay  = np.ones (self.NUM_OF_VNFs)     # self.mig_comp_delay[v] hold the migration's computational cost of VM v. Currently unused.
        # self.mig_data       = 0 * np.ones (self.NUM_OF_VNFs) # self.mig_data[v] amount of data units to transfer during the migration of VM v. Currently unused.
        self.mig_bw         = 1.1 * np.ones (self.NUM_OF_VNFs)

        self.min_cost = float ('inf')
        self.best_nxt_cpu_alloc_of_vnf = np.array (self.NUM_OF_VNFs)
        self.best_nxt_loc_of_vnf       = np.array (self.NUM_OF_VNFs)   # nxt_loc_of_vnf[v] will hold the id of the server planned to host VNF v

        if (self.verbose == 1):
            print ('PoA = ', self.PoA_of_vnf)
            print ('cur VM loc = ', self.cur_loc_of_vnf)
            print ('cur CPU alloc = ', self.cur_cpu_alloc_of_vnf)

        self.n = [] # "n" decision variable, defining the scheduled server and CPU alloc' of each VM.
        self.gen_n()
        self.const_num = int(0)
        self.print_vars ()
        self.print_obj_function ()
        self.gen_p()
        self.gen_all_constraints ()
        exit ()
        my_LP_file_parser = LP_file_parser ()
        my_LP_file_parser.parse_LP_file ('custom_tree.LP')
#         self.brute_force_by_n ()
#        printf (self.LP_output_file, '\nend;\n')
        self.brute_force()
        self.print_best_sol_as_LP ()
        
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
                        
                     
    def gen_link_cap_constraints (self):
        """
        Print the constraints of maximum link's capacity in a LP format
        $$$ The func' isn't complete yet
        """
        for l in self.list_of_links:
            link_l_avail_bw = self.capacity_of_link[l]
            list_of_decision_vars_in_this_eq = []
            
            for list_of_paths_using_link_l in list (filter (lambda item : item['link'] == l, self.paths_of_link)):
                list_of_paths_using_link_l = list_of_paths_using_link_l['paths'] # for each path (s, s') that uses the link l
                print (list_of_paths_using_link_l)
                exit ()

            for v0 in self.v0: # for each VNF which is the first in its chain
                for s in range(self.NUM_OF_SERVERS):
                    if ( not( [self.PoA_of_vnf[v0], s] in list_of_paths_using_link_l)):
                        continue
                    
                    # Now we know that the path from V0's PoA to s uses link l
                    if (self.x[v0][s]): # if x[v][s] == 1, namely v0 is already located on server s 
                        link_l_avail_bw -= self.traffic_in[v0]
                    else: # x[v][s] == 0
                         for id in self.ids_of_y_vs: 
                             list_of_decision_vars_in_this_eq.append ({'id' : id, 'coef' : self.traffic_in[v0]})
                             
#             
#             for list_of_paths_using_link_l in list (filter (lambda item : item['link'] == l, self.paths_of_link)):
#                 for path in list_of_paths_using_link_l['paths']: # for each path (s, s') that uses the link l
#                     if (path[0] in self.PoA_of_vnf):  #if s==PoA of some VNF 
#                         # = self.PoA_of_vnf.index(path[0]) # PoA(v) = s == path[0])
#                         
#                         for item in list (filter (lambda item:  item['s'] == s, self.n )): # for each decision var' related to server s
# 
#                     print (path)
#                 exit ()
#                 for path in list_of_paths_of_this_link['paths']: # For each path (s, s') that uses this link
#                     for v in range (self.NUM_OF_VNFs):           # For each VM v
#                         if (path[0] == self.PoA_of_vnf[v]):       # if s == PoA(v), 
            
            
#             for item in self.y: # for each decision var' related to server s
#                 item['coef'] = 0

            
#             for v0 in self.v0: # for every VNF v which first in its chain
#                 if l is in links_of_path [self.PoA_of_vnf[v0]][v0]]:
#                 if l['link'] == [[self.PoA_of_vnf[v0] , v0]] 
#                 
#             
#                             for item in list (filter (lambda item  : item['v'] == v and item['s'] == s) ): # For each decision var' related which implies locating VM v on server s
#                                 if (is_first_in_list): 
#                                     printf (self.LP_output_file, '{}*X{} ' .format (self.traffic_in[v], item['id']))
#                                     is_first_in_list = False
#                                 else:                                    printf (self.LP_output_file, '+ {}*X{} ' .format (self.traffic_in[v], item['id']))
#                                      
# 
# 
#             
#         for l in self.list_of_links:
#             printf (self.LP_output_file, 'subject to max_link_bw_C{}: ' .format (self.const_num))
#             is_first_in_list = True
#             for list_of_paths_of_this_link in list (filter (lambda item : item['link'] == link, self.paths_of_link)):
#                 for path in list_of_paths_of_this_link['paths']: # For each path (s, s') that uses this link
#                     for v in range (self.NUM_OF_VNFs):           # For each VM v
#                         if (path[0] == self.PoA_of_vnf[v]):       # if s == PoA(v), 
#                             for item in list (filter (lambda item  : item['v'] == v and item['s'] == s) ): # For each decision var' related which implies locating VM v on server s
#                                 if (is_first_in_list): 
#                                     printf (self.LP_output_file, '{}*X{} ' .format (self.traffic_in[v], item['id']))
#                                     is_first_in_list = False
#                                 else:                                    printf (self.LP_output_file, '+ {}*X{} ' .format (self.traffic_in[v], item['id']))
#                                      
#                     print ('path ', path, ' uses link ', link)
#             self.const_num += 1

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
            is_first_in_list = True
            for item in list (filter (lambda item:  item['s'] == s, self.n )): # for each decision var' related to server s
                if (item['coef'] == 0):
                    continue  
                if (is_first_in_list):
                    printf (self.LP_output_file, 'subject to max_cpu_C{}: {}*X{} ' .format (self.const_num, item['coef'], item['id']))
                    self.const_num += 1
                    is_first_in_list = False
                else: 
                    printf (self.LP_output_file, '+ {}*X{} ' .format (item['coef'], item['id']))
            printf (self.LP_output_file, ' <= {};\n' .format (server_s_available_cap))




    def gen_single_alloc_constraints (self):
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
        Print the decision variables. Each variable is printed with the constraints that it's >=0  
        """
        for __ in self.n:
            printf (self.LP_output_file, 'var X{} >= 0;\n' .format (__['id'], __['id']) )
        printf (self.LP_output_file, '\n')


    def gen_all_constraints (self):
        """
        Generate all the constraints. 
        """
        # self.gen_leq1_constraints ()
        # self.gen_single_alloc_constraints ()
        # self.gen_cpu_cap_constraints ()
        self.calc_paths_of_links ()
        self.gen_link_cap_constraints ()
        

    def gen_leq1_constraints (self):
        """
        Print the var' ranges constraints: each var should be <=1  
        """
        printf (self.LP_output_file, '\n') 
        for __ in self.n:
            #self.constraint.append()
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
            tmp_list  = list (filter (lambda item : item ['v'] == v and item ['s'] == self.cur_loc_of_vnf[v] and item['a'] == self.cur_cpu_alloc_of_vnf, self.p))
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
        self.x = - np.zeros (shape = (self.NUM_OF_VNFs, self.NUM_OF_SERVERS), dtype = bool) #[]
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
                     
                    total_delay = comp_delay + 2 * self.servers_path_delay [s] [self.PoA_of_vnf[v]]
                    # Check for per-VM target delay - currently unused
                    # if (total_delay > self.VM_target_delay[v]): # Too high perf' degradation
                    #     continue
                    
                    cost =  a
                    # To include the VM's target delay in the cost, unco
                    # cost += total_delay / self.VM_target_delay[v] 
                    if (mig):
                        cost += self.mig_cost[v]
                    
                    item = {
                        'v'          : v, 
                        's'          : s,
                        'a'          : a,
                        'id'         : id,
                        'comp delay' : comp_delay,
                        'mig'        : mig,
                        'cost'       : cost
                    }
                    
                    self.n.append (item)                
                    
                    list_of_ids.append (id)
                    
                    if (self.verbose == 1):
                        print (item)
                    id +=1 
            
                self.ids_of_y_vs.append ({'v' : v, 's' : s, 'ids' : list_of_ids})
    

        if (self.verbose == 1):
            for item in self.ids_of_y_vs:
                print (item)
     

    def cost_single_VM_chain (self):
        cost = 0
        for item in self.n:
            cost += item['cost'] * self.sol[item['id']]
        return cost
                    
#     def is_feasible (self):
        
                    
    def brute_force_by_n (self):
        self.min_cost = float ('inf')
        for self.sol in itertools.product ([0,1],repeat = len (self.n)):
#              if (!(is_feasible())):
#                  continue
             cost = self.cost_single_VM_chain ()
             if (cost < self.min_cost):
                 self.min_cost = cost
                 self.best_n = sol
        print ('min cost = {}. best sol is {}\n' .format (self.min_cost, self.best_n))
            

        
    def perf_degradation_not_last_vm (self, v, loc_v, loc_vpp, denominator):
        """
        Calculate the performance degradation of a VM, which isn't the last in its chain
        Inputs
        - loc_v - location of v, 
        - loc_vpp - location of v^+ (the next VM after v in the service chain)
        - cpu_alloc - (suggested) cpu allocation of v
        
        """
        return ( 1 / denominator + self.servers_path_delay[loc_v][loc_vpp] ) / self.VM_target_delay[v]
            
    def perf_degradation_single_vm_chain (self, v, loc_v, denominator):
        """
        Calculate the performance degradation of a VM in a single-vm chain.
        Hence, the path delay includes twice twice the delay to the PoA, as the service path begins and ends in the PoA.
        Inputs
        - loc_v - location of v, 
        - loc_vpp - location of v^+ (the next VM after v in the service chain)
        - cpu_alloc - (suggested) cpu allocation of v
        
        """
        return ( 1 / denominator + 2 * self.servers_path_delay[loc_v][self.PoA_of_vnf[v]] ) / self.VM_target_delay[v]
            
    
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
            
            # Calculation of per chain perf' deg' - currently unused
            # self.perf_deg_of_vnf[v] = self.perf_degradation_single_vm_chain (v, self.nxt_loc_of_vnf[v], denominator)
            # if (self.perf_deg_of_vnf[v] > 1):
            #     return False
            
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

    def print_best_sol_as_LP (self):
        """
        Translate the best sol' obtained by brute-force to n_vsa's decision variables
        """
        printf (self.output_file, '\nSet X in the LP are:\n')
        n = []
        for v in range (self.NUM_OF_VNFs):             
            for item in list (filter (lambda item : item['v'] == v and item['s'] == self.best_nxt_loc_of_vnf[v] and item['a'] == self.best_nxt_cpu_alloc_of_vnf[v], self.n)):
                printf (self.output_file, 'X{}, ' .format (item['id']))
    
    def brute_force (self):    
        self.nxt_loc_of_vnf = np.zeros(self.NUM_OF_VNFs, dtype = 'uint8')

        for i in range (pow (self.NUM_OF_SERVERS, self.NUM_OF_VNFs)): #$$$$ TBD: change to while, stopping at last relevant val
            
            self.nxt_cpu_alloc_of_vnf = 1 * np.ones(self.NUM_OF_VNFs, dtype = 'uint8')

            for j in range (pow (self.uniform_cpu_capacity - 1, self.NUM_OF_VNFs)): #$$$$ TBD: change to while, stopping at last relevant val
                
                if (self.is_feasible()):
                    self.calc_beta (self.nxt_loc_of_vnf, self.nxt_cpu_alloc_of_vnf)
                    printf (self.output_file, 'beta = \n{}\nCost = {:.4}\n\n' .format (self.beta, self.cost ()))
                    # printf (self.output_file, 'cost by n = {:.4}\n\n' .format (self.calc_cost_by_n()))
                
                self.nxt_cpu_alloc_of_vnf = self.inc_array (self.nxt_cpu_alloc_of_vnf, 2, self.uniform_cpu_capacity)
            self.nxt_loc_of_vnf = self.inc_array(self.nxt_loc_of_vnf, 0, self.NUM_OF_SERVERS-1)
        self.calc_beta (self.best_nxt_loc_of_vnf, self.best_nxt_cpu_alloc_of_vnf)
        printf (self.output_file, '\nBest solution is:\nbeta = \n{}\nCost = {:.4}\n\n' .format (self.beta, self.min_cost))

