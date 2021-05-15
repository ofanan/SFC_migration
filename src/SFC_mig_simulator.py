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

    def CPUAll_once (self): 
        """
        CPUAll algorithm:
        calculate the minimal CPU allocation required for each chain u, when the highest server on which u is located is s.
        The current implementation assumes that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.
        This version of the alg' assumes a balanced homogeneous tree, 
        so that the netw' delay between every two servers is unequivocally defined by their levels.
        """
        for u in self.usr:
            slack = [u['target delay'] -  u['delay to PoA'] - 2*self.netw_delay_from_leaf_to_lvl[lvl] for lvl in range (self.tree_height+1)]
            slack = [slack[lvl] for lvl in range(self.tree_height+1) if slack[lvl] > 0] # trunc all servers with negative slack, which are surely delay-INfeasible
            #u['L'] = -1 # Highest server which is delay-feasible for u
            u['B'] = [] # u['B'] will hold a list of the budgets required for placing u on each level 
            mu = np.array ([math.floor(u['theta times lambda'][i]) + 1 for i in range (len(u['theta times lambda']))]) # minimal feasible budget
            lvl = 0 
            for lvl in range(len(slack)):
                while (sum(mu) <= u['C_u']): # The SLA still allows increasing this user's CPU allocation
                    if (sum (1 / (mu[i] - u['theta times lambda'][i]) for i in range(len(mu))) <= slack[lvl]):  
                        u['B'].append(sum(mu))
                        # Can save now the exact vector mu; for now, no need for that, as we're interested only in the sum
                        break
                    argmax = np.argmax (np.array ([1 / (mu[i] - u['theta times lambda'][i]) - 1 / (mu[i] + 1 - u['theta times lambda'][i]) for i in range(len(mu))]))
                    mu[argmax] = mu[argmax] + 1

    def rd_usr_data (self):
        """
        Read the input about the users (target delay, traffic), and write it to the appropriate fields in self.
        The input data is read from the file self.usr_loc_file_name. 
        """
        usrs_data_file = open ("../res/" + self.usrs_data_file_name,  "r")
        self.usr = np.empty (self.MAX_NUM_OF_USRS, dtype=object)
        self.NUM_OF_VMs = 0
        
        for line in usrs_data_file: 
    
            # Ignore comments lines
            if (line.split ("//")[0] == ""):
                continue

            splitted_line = line.split (" ")

            if (splitted_line[0].split("u")[0] == ""): # line begins by "u"
                u = int(splitted_line[0].split("u")[1])
                
                if (splitted_line[1] == "theta_times_lambda"):              
                    theta_times_lambda = line.split("=")[1].rstrip().split(",")
                    self.usr[u] = {'theta times lambda' : [float (theta_times_lambda[i]) for i in range (len (theta_times_lambda)) ]}
                    self.NUM_OF_VMs += len (theta_times_lambda)

                # The mig_cost here is for the whole chain. For specifying per-VM mig' cost,
                # need to parse a vector, as done for 'theta times lambda' above. 
                elif (splitted_line[1] == "mig_cost"):              
                    self.usr[u]['mig cost'] = int (line.split("=")[1].rstrip())
                    
                elif (splitted_line[1] == "target_delay"):              
                    self.usr[u]['target delay'] = float (line.split("=")[1].rstrip())
                  
                elif (splitted_line[1] == "delay_to_PoA"):              
                    self.usr[u]['delay to PoA'] = float (line.split("=")[1].rstrip())
                    
                elif (splitted_line[1] == "C_u"):              
                    self.usr[u]['C_u'] = int (line.split("=")[1].rstrip())
                    
        self.usr = np.delete (self.usr, [i for i in range(self.MAX_NUM_OF_USRS) if i>u])
                                                     
    def loc2ap (self):
        """
        Currently unused.
        Read the input about the users locations, 
        and write the appropriate user-to-PoA connections to the file self.ap_file
        """
        self.ap_file  = open ("../res/" + self.usr_loc_file_name.split(".")[0] + ".ap", "w+")  
        usrs_loc_file = open ("../res/" + self.usr_loc_file_name,  "r") 
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
        
            elif (splitted_line[0] == "user"):
                X, Y = float(splitted_line[2]), float(splitted_line[3])
                ap = int (math.floor ((Y / cell_Y_edge) ) * num_of_APs_in_row + math.floor ((X / cell_X_edge) )) 
                printf(self.ap_file, "({}, {})," .format (line.split (" ")[1], ap))
                continue
            
        printf(self.ap_file, "\n")

            
    def gen_parameterized_tree (self):
        """
        Generate a parameterized tree with specified height and children-per-non-leaf-node. 
        """
        self.G                  = nx.generators.classic.balanced_tree (r=self.tree_height, h=self.children_per_node) # Generate a tree of height h where each node has r children.
        self.NUM_OF_SERVERS     = self.G.number_of_nodes()
        self.CPU_cap_at_lvl  = [3 * (lvl+1) for lvl in range (self.tree_height+1)]                
        self.CPU_cost_at_lvl = [3 * (self.tree_height - lvl) for lvl in range (self.tree_height+1)]                
        
        self.G = self.G.to_directed()

        shortest_path = nx.shortest_path(self.G)


        # levelize the tree (assuming a balanced tree) 
        root                  = 0 # In networkx, the ID of the root is 0
        self.num_of_leaves    = 0
        self.cpu_cost_at_root = 3^self.tree_height
        for s in self.G.nodes(): # for every server
            if self.G.out_degree(s)==1 and self.G.in_degree(s)==1: # is it a leaf?
                self.G.nodes[s]['lvl']   = 0 # Yep --> its lvl is 0
                self.G.nodes[s]['AP id'] = self.num_of_leaves
                self.num_of_leaves += 1
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
        self.Lmax                   = 0
        self.uniform_Tpd            = 2
        self.uniform_link_cost      = 1
        self.max_chain_len          = 2
        self.MAX_NUM_OF_USRS        = 1000
        self.usrs_data_file_name    = "res.usr" #input file containing the target delays and traffic of all users
        self.usr_loc_file_name      = "my_mob_sim.loc"  #input file containing the locations of all users along the simulation
        self.usr_ap_file_name       = 'mob_sim.ap' #input file containing the APs of all users along the simulation
        self.gen_parameterized_tree ()
        self.write_to_prb_file = False # When true, will write outputs to a .prb file. - ".prb" - A .prb file may solve an LP problem using the online Eq. solver: https://online-optimizer.appspot.com/?model=builtin:default.mod
        self.write_to_py_file  = True # When true, will write to Py file, checking the feasibility and cost of a suggested sol'.  
        self.write_to_mod_file = False # When true, will write to a .mod file, fitting to IBM CPlex solver       
        self.write_to_cfg_file = True
        self.write_to_lp_file  = True  # When true, will write to a .lp file, which allows running Cplex using a Python's api.       

    def simulate (self):
        self.rd_usr_data ()
        self.CPUAll_once  ()       
        
        self.ap_file  = open ("../res/" + self.usr_ap_file_name, "r")  

        for line in self.ap_file: 

            # Ignore comments lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue

            splitted_line = line.split (" ")

            if (splitted_line[0] == "time"):
                continue
        
            self.rd_AP_line(splitted_line)
            self.alg_top ()
    
    def alg_top (self):
        """
        Top-level alg'
        """
        
        self.bu ()
        return
    
    def bu (self):
        """
        Bottom-up alg'
        """
    
    def rd_AP_line (self, line):
        splitted_line = line[0].split ("\n")[0].split (")")
        for tuple in splitted_line:
            tuple = tuple.split("(")
            if (len(tuple) > 1):
                tuple   = tuple[1].split (",")
                usr_num = int(tuple[0])
                AP      = int(tuple[1])
                if (usr_num > len (self.usr)-1):
                    print ('error: encountered usr num {}, where by res.usr file, there are only {} users' .format (usr_num, len(self.usr)))
                    exit  ()
                self.usr[usr_num]['nxt AP'] = AP
        

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
    my_simulator.simulate()
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

