import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf
import Check_sol
import obj_func
from _overlapped import NULL
# from solve_problem_by_Cplex import solve_problem_by_Cplex
from networkx.algorithms.threshold import shortest_path
from cmath import sqrt
from numpy import int

class SFC_mig_simulator (object):

    def check_greedy_alg (self):
        
        theta_v_lambda_v    = np.array ([0.1, 0.3, 0.5, 0.7, 0.9]) 
        max_alloc           = 2 * np.ones (len (theta_v_lambda_v), dtype = 'int8')
        
        max_budget          = sum (max_alloc) 
        
        greedy_delay        = float ("inf") * np.ones (max_budget+1)
        bf_delay            = float ("inf") * np.ones (max_budget+1)
        
        B0                  = np.array ([math.ceil(theta_v_lambda_v[i]) for i in range (len(theta_v_lambda_v))]) # minimal feasible budget
        mu                  = B0.copy ()
        budget              = sum (mu)
        
        # Calculate bf-delays
        while (budget < max_budget):
            budget   = int(sum (mu))
            delay = sum (1 / (mu[i] - theta_v_lambda_v[i]) for i in range(len(mu))) 
            bf_delay[budget] = min(bf_delay[budget], delay) 
            mu = self.inc_array (mu, B0, max_alloc) 
        
        print (' bf_delay = ', bf_delay)
        
        # Calculate greedy delays
        mu                  = B0.copy ()
        budget              = sum (mu)
        while (budget <= max_budget):
    
            greedy_delay[budget] = sum (1 / (mu[i] - theta_v_lambda_v[i]) for i in range(len(mu)))        
            argmax = np.argmax (np.array ([1 / (mu[i] - theta_v_lambda_v[i]) - 1 / (mu[i] + 1 - theta_v_lambda_v[i]) for i in range(len(mu))]))
            mu[argmax] = mu[argmax] + 1
            budget     = budget     + 1 
            
        print (' greedy_delay = ', greedy_delay)
        

    def calc_CPU_alloc (self): 
        """
        calculate the minimal CPU allocation required for each chain u, when the highest server on which u is located is s.
        The current implementation assumes that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.   
        """
        for u in self.users:
            slack = [u['target delay'] -  self.netw_delay_from_leaf_to_lvl[lvl] for lvl in range (self.tree_height+1)]
            mu    = np.array ([math.floor(u['theta times lambda'][i]) + 1 for i in range (len(u['theta times lambda']))]) # minimal feasible budget
            u['req cpu'] = float ('inf') * np.ones (len(mu))
            req_cpu = sum (mu)
            lvl = 0 
            while lvl in range (self.tree_height+1):
                while (sum (1 / (mu[i] - u['theta times lambda'][i]) for i in range(len(mu))) > slack[lvl]): # while the delay is above the slack delay, and we don't exceed the CPU cap of this lvl 
                    argmax = np.argmax (np.array ([1 / (mu[i] - u['theta times lambda'][i]) - 1 / (mu[i] + 1 - u['theta times lambda'][i]) for i in range(len(mu))]))
                    mu[argmax] = mu[argmax] + 1
                    req_cpu += 1
                    if (req_cpu > self.CPU_cap_at_lvl[lvl]): # the CPU cap at this lvl doesn't suffice for allocating this user's chain -> by the monotone-feasibility assumption, cannot allocate also in any higher lvl
                        lvl = self.tree_height+1
                        break
                # can successfully allocate this user on this lvl
                u['req cpu'][lvl] = req_cpu
            print (u['req cpu'])

        exit ()
    def gen_users_data (self):
        """
        Read the input about the users (target delay, traffic), and write it to the appropriate fields in self.
        The input data is read from the file self.users_loc_file_name. 
        """
        usrs_data_file = open ("../res/" + self.usrs_data_file_name,  "r")
        self.users = np.array (1, dtype=object)
        self.NUM_OF_VMs = 0
        
        for line in usrs_data_file: 
    
            # Ignore comments lines
            if (line.split ("//")[0] == ""):
                continue

            splitted_line = line.split (" ")

            if (splitted_line[0] == "num_of_users"): 
                self.users = np.resize (np.resize, int (line.split("=")[1].rstrip()))
                self.NUM_OF_USERS  = int (line.split("=")[1].rstrip())
                self.NUM_OF_CHAINS = self.NUM_OF_USERS
                continue 

            if (splitted_line[0].split("u")[0] == ""): # line begins by "u"
                u = int(splitted_line[0].split("u")[1])
                
                if (splitted_line[1] == "theta_times_lambda"):              
                    theta_times_lambda = line.split("=")[1].rstrip().split(",")
                    self.users[u] = {'theta times lambda' : [float (theta_times_lambda[i]) for i in range (len (theta_times_lambda)) ]}
                    self.NUM_OF_VMs += len (theta_times_lambda)
        
                elif (splitted_line[1] == "target_delay"):              
                    self.users[u]['target delay'] = float (line.split("=")[1].rstrip())
                  
                elif (splitted_line[1] == "mig_cost"):              
                    self.users[u]['mig cost'] = float (line.split("=")[1].rstrip())
                # elif (splitted_line[1] == "mig_cost"):  
                #     mig_cost = line.split("=")[1].rstrip().split(",")
                #     self.mig_cost_of_usr[u] = [self.mig_cost_of_usr[u].fill (7)]  #(float (line.split("=")[1].rstrip()))
                #     print (self.mig_cost_of_usr[u])
                #     exit ()

                                  
        # The code below may be un-commented and ran only once we know the PoAs, as they're dummy VMs in the chain. 
        # Also, the code is relevant only for non-SS solutions.
        # Calculate v^+ of each VNF v.
        # self.vpp[v] will hold the idx of the next VNF in the same chain.
        # if v is the last VNF in its chain, then self.vpp[v] will hold the PoA of this chain's user  
        # self.vpp                    = np.zeros (self.NUM_OF_VMs, dtype = 'uint')
        #self.v0                     = np.zeros (self.NUM_OF_CHAINS, dtype = 'uint') # self.v0 will hold a list of all the VNFs which are first in their chain
        #self.v_inf                  = np.zeros (self.NUM_OF_CHAINS, dtype = 'uint') # self.v_inf will hold a list of all the VNFs which are last in their chain
        #self.v_not_inf              = [] # list of vnf's that are NOT last in the chain
        #self.PoA_of_vnf = np.zeros (self.NUM_OF_VMs, dtype = 'uint') # self.PoA_of_vnf[v] will hold the PoA of the user using VNF v
        # self.PoA_of_user            = random.choices (self.leaves, k=self.NUM_OF_USERS)
        
        # self.vnf_in_chain                 = np.empty (shape = self.NUM_OF_CHAINS, dtype = object) # self.vnf_in_chain[c] will hold a list of the VNFs in chain c  
        # v = 0
        # for chain_num in range (self.NUM_OF_CHAINS):
        #     self.vnf_in_chain                 [chain_num] = []
        #     # self.theta_times_lambda_v_chain [chain_num] = []
        #     for idx_in_chain in range (self.NUM_OF_VMs_in_chain[chain_num]):
        #         self.vnf_in_chain                [chain_num].append (v)
        #         # self.theta_times_lambda_v_chain[chain_num].append (self.theta_times_lambda_v[v]) 
        #         if (idx_in_chain == 0):
        #             self.v0[chain_num] = v 
        #         if (idx_in_chain == self.NUM_OF_VMs_in_chain[chain_num]-1): # Not "elif", because in the case of a single-VM chain, the first is also the last
        #             self.v_inf[chain_num] = v
        #             self.vpp [v] = self.PoA_of_user[chain_num]
        #         else: # Not the last VM in the chain
        #             self.v_not_inf.append(v)
        #             self.vpp [v] = v+1 
        #         self.PoA_of_vnf [v] = self.PoA_of_user[chain_num]    
        #         v += 1
        
        # A Single-Server (per chain) solution for the problem 
        # self.chain_nxt_loc          = 4 * np.ones (self.NUM_OF_CHAINS, dtype ='uint8') #self.PoA_of_user 
        # self.chain_nxt_total_alloc = 7 * np.ones (self.NUM_OF_CHAINS) # total CPU allocation of the chain
                
    def gen_APs (self):
        """
        Read the input about the users delay, and write the appropriate user-to-PoA connections to the file self.ap_file
        The input data is read from the file self.users_loc_file_name.          
        """
        self.ap_file  = open ("../res/" + self.users_loc_file_name.split(".")[0] + ".ap", "w+")  
        usrs_loc_file = open ("../res/" + self.users_loc_file_name,  "r") 
        printf (self.ap_file, '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of user X at time t\n')
            
        for line in usrs_loc_file: 
        
            # Ignore comments lines
            if (line.split ("//")[0] == ""):
                continue

            splitted_line = line.split (" ")

            if (splitted_line[0] == 'MAX_X'):
                max_X, max_Y = float(splitted_line[1]), float(splitted_line[3])
                if (max_X != max_Y):
                    print("Sorry, currently only square city sizes are supported. Please fix the .loc file\n")
                    exit ()
            
            num_of_APs_in_row = int (math.sqrt (self.num_of_leaves)) #$$$ cast to int, floor  
            cell_X_edge = max_X / num_of_APs_in_row
            cell_Y_edge = cell_X_edge
            
            if (splitted_line[0] == "time"):
                printf(self.ap_file, "\ntime = {}: " .format (splitted_line[2].rstrip()))
                continue
        
            if (splitted_line[0] == "user"):
                X, Y = float(splitted_line[2]), float(splitted_line[3])
                ap = int (math.floor ((Y / cell_Y_edge) ) * num_of_APs_in_row + math.floor ((X / cell_X_edge) )) 
                printf(self.ap_file, "({}, {})," .format (line.split (" ")[1], ap))
                continue
            
        printf(self.ap_file, "\n")

            
    def gen_parameterized_tree (self):
        """
        Generate a parameterized regular three-nodes tree (root and 2 leaves). 
        """
        self.G                  = nx.generators.classic.balanced_tree (r=self.tree_height, h=self.children_per_node) # Generate a tree of height h where each node has r children.
        self.NUM_OF_SERVERS     = self.G.number_of_nodes()
        self.CPU_cap_at_lvl  = [3 * (lvl+1) for lvl in range (self.tree_height+1)]                
        self.CPU_cost_at_lvl = [3 * (self.tree_height - lvl) for lvl in range (self.tree_height+1)]                
        
        self.G = self.G.to_directed()

        shortest_path = nx.shortest_path(self.G)


        # levelize the tree (assuming a balanced tree) 
        root = 0 # In networkx, the ID of the root is 0
        self.num_of_leaves = 0
        self.cpu_cost_at_root = 3^self.tree_height
        for s in self.G.nodes(): # for every server
            if self.G.out_degree(s)==1 and self.G.in_degree(s)==1: # is it a leaf?
                self.num_of_leaves += 1
                self.G.nodes[s]['lvl'] = 0 # Yep --> its lvl is 0
                for lvl in range (self.tree_height):
                    self.G.nodes[shortest_path[s][root][lvl]]['lvl']      = lvl # assume here a balanced tree
                    self.G.nodes[shortest_path[s][root][lvl]]['CPU cap']  = self.CPU_cap_at_lvl[lvl]                
                    self.G.nodes[shortest_path[s][root][lvl]]['CPU cost'] = self.CPU_cost_at_lvl[lvl]                
        # # Iterate over all children of node i
        # for n in self.G.neighbors(i):
        #     if (n > i):
        #         print (n)
        # exit
                    
        # Calculate edge propagation delays    
        for edge in self.G.edges: 
            self.G[edge[0]][edge[1]]['delay'] = self.Lmax / self.uniform_link_capacity + self.uniform_Tpd
            self.G[edge[0]][edge[1]]['cost']  = self.uniform_link_cost
            # paths_using_this_edge = []
            # for src in range (self.NUM_OF_SERVERS):
                # for dst in range (self.NUM_OF_SERVERS): 
                    # if ((edge[0],edge[1]) in links_of_path[src][dst]): # Does link appear in the path from src to dst
                        # paths_using_this_edge.append ((src, dst)) # Yep --> append it to the list of paths in which this link appears
            # self.G[edge[0]][edge[1]]['paths using this edge'] = paths_using_this_edge

        # self.path_delay[s][d] will hold the prop' delay of the path from server s to server d
        self.path_delay   = np.zeros ([self.NUM_OF_SERVERS, self.NUM_OF_SERVERS]) 
        self.path_bw_cost = np.zeros ([self.NUM_OF_SERVERS, self.NUM_OF_SERVERS])
        for s in range (self.NUM_OF_SERVERS):
            for d in range (self.NUM_OF_SERVERS):
                if (s == d):
                    continue
                self.path_delay   [s][d] = sum (self.G[shortest_path[s][d][hop]][shortest_path[s][d][hop+1]]['delay'] for hop in range (len(shortest_path[s][d])-1))
                self.path_bw_cost [s][d] = sum (self.G[shortest_path[s][d][hop]][shortest_path[s][d][hop+1]]['cost']  for hop in range (len(shortest_path[s][d])-1))
                
        # calculate the network delay from a leaf to a node in each level,  
        # assuming that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.   
        leaf = self.G.number_of_nodes()-1 # when using networkx and a balanced tree, self.path_delay[self.G[nodes][-1]] is surely a leaf (it's the node with highest ID).
        self.netw_delay_from_leaf_to_lvl = [ self.path_delay[leaf][shortest_path[leaf][root][lvl]] for lvl in range (0, self.tree_height+1)]

    def __init__ (self, verbose = -1):
        """
        Init a toy example - topology (e.g., chains, VMs, target delays etc.).
        """
        
        self.verbose                = verbose
        
        # Network parameters
        self.tree_height            = 2
        self.children_per_node      = 4 # num of children of every non-leaf node
        self.uniform_link_capacity  = 100
        self.Lmax                   = 1
        self.uniform_Tpd            = 5
        self.uniform_link_cost      = 1
        self.max_chain_len          = 2
        self.usrs_data_file_name = "res.usr" #input file containing the target delays and traffic of all users
        self.users_loc_file_name = "my_mob_sim.loc"  #input file containing the locations of all users along the simulation
        self.gen_parameterized_tree ()
        self.gen_users_data ()
        self.gen_APs ()
        self.calc_CPU_alloc()
        self.write_to_prb_file = False # When true, will write outputs to a .prb file. - ".prb" - A .prb file may solve an LP problem using the online Eq. solver: https://online-optimizer.appspot.com/?model=builtin:default.mod
        self.write_to_py_file  = True # When true, will write to Py file, checking the feasibility and cost of a suggested sol'.  
        self.write_to_mod_file = False # When true, will write to a .mod file, fitting to IBM CPlex solver       
        self.write_to_cfg_file = True
        self.write_to_lp_file  = True  # When true, will write to a .lp file, which allows running Cplex using a Python's api.       

    def calc_SS_sol_total_cost (self):
        """
        Calculate the total cost of an SS (single-server pver-chain) full solution.
        """
        total_cost = 0
        for chain in range(self.NUM_OF_CHAINS):
            total_cost += self.CPU_cost[self.chain_nxt_loc[chain]] * self.chain_nxt_total_alloc[chain] + \
                        self.path_bw_cost[self.PoA_of_user[chain]]  [self.chain_nxt_loc[chain]] * self.lambda_v[chain][0] + \
                        self.path_bw_cost[self.chain_nxt_loc[chain]][self.PoA_of_user[chain]]   * self.lambda_v[chain][self.NUM_OF_VMs_in_chain[chain]] + \
                        (self.chain_cur_loc[chain] != self.chain_nxt_loc[chain]) * self.chain_mig_cost[chain]
        exit
            
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
     
          
if __name__ == "__main__":
    lp_time_summary_file = open ("../res/lp_time_summary.res", "a") # Will write to this file an IBM CPlex' .mod file, describing the problem
    
    # Gen static LP problem
    t = time.time()
    my_simulator = SFC_mig_simulator (verbose = 0)
    # my_simulator.calc_SS_sol_total_cost ()
    # my_simulator.check_greedy_alg ()
    # exit (0)
    #
    # my_simulator.init_problem  ()
    #
    # # Gen dynamic LP problem
    # t = time.time()
    # for leaf in my_simulator.leaves:
        # my_simulator.gen_lp_problem(leaf)
        #
        #
    # my_simulator.gen_lp_problem_old ()
    # printf (lp_time_summary_file, 'Gen dynamic LP took {:.4f} seconds\n' .format (float (time.time() - t)))
    #
    # # Solve the LP problem
    # t = time.time()
    # my_simulator.run_lp_Cplex()
    # printf (lp_time_summary_file, 'Solving the LP took {:.4f} seconds\n' .format (float (time.time() - t)))

