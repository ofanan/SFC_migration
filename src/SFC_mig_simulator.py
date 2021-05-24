import networkx as nx
import numpy as np
import math
import itertools 
import time
import random
import heapq

from usr_c import usr_c
from printf import printf
# import Check_sol
# import obj_func
# from _overlapped import NULL
# from solve_problem_by_Cplex import solve_problem_by_Cplex
from networkx.algorithms.threshold import shortest_path
from cmath import sqrt
from numpy import int

class SFC_mig_simulator (object):


    #############################################################################
    # Inline functions
    #############################################################################
    # An inline function that returns the parent of a given server
    parent_of = lambda self, s : self.G.nodes[s]['prnt']

    # # Find the server on which a given user is located, by its lvl
    loc_of_user = lambda self, u : self.usrs[u].S_u[self.usrs[u].lvl]

    # calculate the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    calc_chain_cost_homo = lambda self, usr, lvl: self.link_cost_of_SSP_at_lvl[lvl] + (not (self.X[usr.id][usr.S_u[usr.lvl]])) * self.uniform_mig_cost * len (usr.theta_times_lambda) + self.CPU_cost_at_lvl[lvl] * usr.B[lvl]    
          
    # Calculate the (maximal) rsrc aug' used by the current solution
    calc_sol_rsrc_aug = lambda self, R : np.max ([(R * self.G.nodes[s]['cpu cap'] - self.G.nodes[s]['a']) / self.G.nodes[s]['cpu cap'] for s in self.G.nodes()])
         
    # Valculate the CPU used in practice in each server in the current solution
    used_cpu_in = lambda self, R : [(R * self.G.nodes[s]['cpu cap'] - self.G.nodes[s]['a']) for s in self.G.nodes()]

    # calculate the cost of the current solution (self.Y)
    calc_sol_cost = lambda self: sum ([self.calc_chain_cost_homo (usr, usr.lvl) for usr in self.usrs])
   
    # calculate the proved upper bnd on the rsrc aug that bottomUp may need to find a feasible sol, given such a sol exists for the non-augmented prob'
    calc_upr_bnd_rsrc_aug = lambda self: np.max ([usr.C_u for usr in self.usrs]) / np.min ([np.min (usr.B) for usr in self.usrs])

    # Returns True iff the solution self.Y schedules all the chain.
    # Note: the function does NOT check whether self.Y adheres to all the constraints (e.g., CPU cap', link cap', delay).
    # This is assumed to be True by construction.
    found_sol = lambda self: sum (sum (self.Y)) >= len (self.usrs)

    def reset_sol (self):
        """"reset the solution, including:
        - self.Y (init it to a matrix of "False")
        - usr.lvl
        """
        self.Y = np.zeros ([len (self.usrs), len (self.G.nodes())], dtype = 'bool')
        for usr in self.usrs:
            usr.lvl = -1


    def print_cost_per_usr (self):
        """
        For debugging / analysing only.
        print the cost of each chain. 
        """
        
        for usr in self.usrs:
            chain_cost = self.calc_chain_cost_homo (usr, usr.lvl)  
            print ('cost of usr {} = {}' .format (u, chain_cost))
        

    def init_log_file (self):
        """
        Open the log file for writing and write initial comments line on it
        """
        with open('../res/' + self.log_file_name, 'w') as self.log_output_file:
            self.log_output_file.write('')                
        self.log_output_file  = open ('../res/' + self.log_file_name,  "w")
        printf (self.log_output_file, '// format: s : used / C_s   chains[u1, u2, ...]\n')
        printf (self.log_output_file, '// where: s = number of server. used = capacity used by the sol on server s. C_S = non-augmented capacity of s. u1, u2, ... = chains placed on s.\n' )

    def print_sol (self, R):
        """
        print a solution for DPM to the output log file 
        """
        printf (self.log_output_file, 'R = {}, phi = {}\n' .format (self.calc_sol_rsrc_aug (R), self.calc_sol_cost()))
        used_cpu_in = self.used_cpu_in (R)
        for s in self.G.nodes():
            printf (self.log_output_file, '{}: {} / {}\t chains {}\n' .format (s, used_cpu_in[s], self.G.nodes[s]['cpu cap'], [u for u in range (len(self.usrs)) if self.Y[u][s] ] ))
        
    def push_up (self):
        """
        Push chains up: take a feasible solution, and greedily try pushing each chain as high as possible in the tree. 
        Do that when chains are sorted in a decreasing order of the # of CPU units they're currently using.
        """
        
        # Assume here that the available cap' at each server 'a' is already calculated by the alg' that was run 
        # before calling push-up ()
        usrs_heap = []
        for usr in self.usrs: #range (len(self.usrs)):
            heapq.heappush(usrs_heap, usr)

        # Assume here that the available cap' at each server 'a' is already calculated by the alg' that was run 
        # before calling reduce_cost ()
        stop_cntr = 0 # will cnt continuum number of usrs that we didn't succeed to push-up  
        while True:
            usr = heapq.nlargest(1, usrs_heap)[0]
            for lvl in range (len(usr.B)-1, usr.lvl+1, -1): 
                if (self.G.nodes[usr.S_u[lvl]]['a'] >= usr.B[lvl]): # if there's enough available space to move u to level lvl                     
                    self.G.nodes [usr.S_u[usr.lvl]] ['a'] += usr.B[usr.lvl] # inc the available CPU at the prev loc of the moved usr  
                    self.G.nodes [usr.S_u[lvl]]     ['a'] -= usr.B[lvl]     # dec the available CPU at the new  loc of the moved usr
                    usr.lvl = lvl # update usr.lvl accordingly. Note: we don't update self.Y for now.
                    if (lvl == len(usr.B)-1): # If we pushed the usr to the highest  is still not the highest delay-feasible server for this server...
                        heapq.heappush(usrs_heap, usr) # push usr back to the heap; after more users move maybe it will be possible to push this user further up
                    stop_cntr = 0 # succeeded to push-up a user, so reset the cntr
            stop_cntr += 1
            if (stop_cntr == len (self.usrs)): # didn't suucceed to push-up any user
                break
                    
    def reduce_cost (self):
        """
        Reduce cost alg': take a feasible solution, and greedily decrease the cost 
        by changing the placement of a single chain, using a gradient method, as long as this is possible.
        """        
        # Assume here that the available cap' at each server 'a' is already calculated by the alg' that was run 
        # before calling reduce_cost ()
        while (1):
            lvl_star = -1; max_reduction = 0 # init default values for the maximal reduction cost func', and for the argmax indices 
            for usr in self.usrs:
                for lvl in range(len (usr.B)): # for each level in which there's a delay-feasible server for this usr
                    if (self.G.nodes[usr.S_u[lvl]]['a'] >= usr.B[lvl]): # if there's enough available space to move u to level lvl 
                        reduction = self.calc_chain_cost_homo (usr, usr.lvl) - self.calc_chain_cost_homo (usr, lvl)
                        if (reduction > max_reduction):  
                            usr2mov       = usr
                            lvl_star      = lvl
                            max_reduction = reduction 
            if (max_reduction == 0): # cannot decrease cost anymore
                break
            print ('max_reduction = ', max_reduction)
            
            # move usr2mov from its current lvl to lvl_star, and update Y, a, accordingly
            self.G.nodes [usr2mov.S_u[usr2mov.lvl]] ['a'] += usr2mov.B[usr2mov.lvl] # inc the available CPU at the prev loc of the moved usr  
            self.G.nodes [usr2mov.S_u[lvl_star]]   ['a'] -= usr2mov .B[lvl_star]           # dec the available CPU at the new  loc of the moved usr
            print ('id of usr2mov = {}, old lvl = {}, lvl_star = {}, max_reduction = {}' .format(usr2mov.id, usr2mov.lvl, lvl_star, max_reduction))
            usr2mov.lvl                                       = lvl_star # update usr.lvl accordingly. Note: we don't update self.Y for now.
        
        # Update self.Y
        self.Y = np.zeros ([len (self.usrs), len (self.G.nodes())], dtype = 'bool')
 
        for usr in self.usrs:
            self.Y[usr.id][usr.S_u[usr.lvl]] = True   
    
        
    def CPUAll_once (self): 
        """
        CPUAll algorithm:
        calculate the minimal CPU allocation required for each chain u, when the highest server on which u is located is s.
        The current implementation assumes that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.
        This version of the alg' assumes a balanced homogeneous tree, 
        so that the netw' delay between every two servers is unequivocally defined by their levels.
        """
        for usr in self.usrs:
            slack = [usr.target_delay - self.link_delay_of_SSP_at_lvl[lvl] for lvl in range (self.tree_height+1)]
            slack = [slack[lvl] for lvl in range(self.tree_height+1) if slack[lvl] > 0] # trunc all servers with negative slack, which are surely delay-INfeasible
            #u.L = -1 # Highest server which is delay-feasible for u
            usr.B = [] # usr.B will hold a list of the budgets required for placing u on each level 
            mu = np.array ([math.floor(usr.theta_times_lambda[i]) + 1 for i in range (len(usr.theta_times_lambda))]) # minimal feasible budget
            lvl = 0 
            for lvl in range(len(slack)):
                while (sum(mu) <= usr.C_u): # The SLA still allows increasing this user's CPU allocation
                    if (sum (1 / (mu[i] - usr.theta_times_lambda[i]) for i in range(len(mu))) <= slack[lvl]):  
                        usr.B.append(sum(mu))
                        # Can save now the exact vector mu; for now, no need for that, as we're interested only in the sum
                        break
                    argmax = np.argmax (np.array ([1 / (mu[i] - usr.theta_times_lambda[i]) - 1 / (mu[i] + 1 - usr.theta_times_lambda[i]) for i in range(len(mu))]))
                    mu[argmax] = mu[argmax] + 1
            usr.L = lvl # usr.L holds the highest lvl on which it's possible to locate this usr
            
        # for u in range(len (self.usrs)):
        #     print ('mu[{}] = {}' .format (u, self.usrs[u]B))

    def rd_usr_data (self):
        """
        Read the input about the users (target_delay, traffic), and write it to the appropriate fields in self.
        The input data is read from the file self.usrs_loc_file_name. 
        """
        usrs_data_file = open ("../res/" + self.usrss_data_file_name,  "r")
        self.usrs = []
        
        for line in usrs_data_file: 
    
            # Ignore comments lines
            if (line.split ("//")[0] == ""):
                continue

            splitted_line = line.split (" ")

            if (splitted_line[0].split("u")[0] == ""): # line begins by "u"
                id = int(splitted_line[0].split("u")[1])
                self.usrs.append (usr_c(id = id))

            elif (splitted_line[0] == "theta_times_lambda"):
                theta_times_lambda = line.split("=")[1].rstrip().split(",")
                self.usrs[id].theta_times_lambda = [float (theta_times_lambda[i]) for i in range (len (theta_times_lambda)) ]

            elif (splitted_line[0] == "target_delay"):              
                self.usrs[id].target_delay = float (line.split("=")[1].rstrip())
              
            elif (splitted_line[0] == "mig_cost"):              
                mig_cost = line.split("=")[1].rstrip().split(",")
                self.usrs[id].mig_cost = [float (mig_cost[i]) for i in range (len (mig_cost)) ]
                
            elif (splitted_line[0] == "C_u"):              
                self.usrs[id].C_u = int (line.split("=")[1].rstrip())
                            
    def loc2ap (self):
        """
        Currently unused.
        Read the input about the users locations, 
        and write the appropriate user-to-PoA connections to the file self.ap_file
        """
        self.ap_file  = open ("../res/" + self.usrs_loc_file_name.split(".")[0] + ".ap", "w+")  
        usrs_loc_file = open ("../res/" + self.usrs_loc_file_name,  "r") 
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
        self.G                 = nx.generators.classic.balanced_tree (r=self.tree_height, h=self.children_per_node) # Generate a tree of height h where each node has r children.
        self.NUM_OF_SERVERS    = self.G.number_of_nodes()
        self.CPU_cap_at_lvl    = [3 * (lvl+1) for lvl in range (self.tree_height+1)]                
        self.CPU_cost_at_lvl   = [1 * (self.tree_height + 1 - lvl) for lvl in range (self.tree_height+1)]
        self.link_cost_at_lvl  = np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of using a link from level i to level i+1, or vice versa.
        self.link_delay_at_lvl = np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of using a link from level i to level i+1, or vice versa.
        
        # overall link cost and link capacity of a Single-Server Placement of a chain at each lvl
        self.link_cost_of_SSP_at_lvl  = [2 * sum([self.link_cost_at_lvl[i]  for i in range (lvl)]) for lvl in range (self.tree_height+1)] 
        self.link_delay_of_SSP_at_lvl = [2 * sum([self.link_delay_at_lvl[i] for i in range (lvl)]) for lvl in range (self.tree_height+1)] 
        
        self.G = self.G.to_directed()

        shortest_path = nx.shortest_path(self.G)

        # levelize the tree (assuming a balanced tree)
        self.ap2s             = np.zeros (len (self.G.nodes), dtype='int16') 
        root                  = 0 # In networkx, the ID of the root is 0
        self.num_of_leaves    = 0
        self.cpu_cost_at_root = 3^self.tree_height
        for s in self.G.nodes(): # for every server
            self.G.nodes[s]['id'] = s
            if self.G.out_degree(s)==1 and self.G.in_degree(s)==1: # is it a leaf?
                self.G.nodes[s]['lvl']   = 0 # Yep --> its lvl is 0
                # self.G.nodes[s]['AP id'] = self.num_of_leaves
                self.ap2s[self.num_of_leaves] = s
                self.num_of_leaves += 1
                for lvl in range (self.tree_height+1):
                    self.G.nodes[shortest_path[s][root][lvl]]['lvl']       = lvl # assume here a balanced tree
                    self.G.nodes[shortest_path[s][root][lvl]]['cpu cap']   = self.CPU_cap_at_lvl[lvl]                
                    self.G.nodes[shortest_path[s][root][lvl]]['cpu cost']  = self.CPU_cost_at_lvl[lvl]                
                    self.G.nodes[shortest_path[s][root][lvl]]['link cost'] = self.link_cost_of_SSP_at_lvl[lvl]
                    # # Iterate over all children of node i
                    # for n in self.G.neighbors(i):
                    #     if (n > i):
                    #         print (n)
        
        # Find parents of all nodes (except of the root)
        for s in range (1, len(self.G.nodes())):
            self.G.nodes[s]['prnt'] = shortest_path[s][root][1]

        # # Calculate delays and costs for the fully-hetero' case, where each link may have a unique cost / delay.    
        # for edge in self.G.edges: 
        #     self.G[edge[0]][edge[1]]['delay'] = self.Lmax / self.uniform_link_capacity + self.uniform_Tpd
            # paths_using_this_edge = []
            # for src in range (self.NUM_OF_SERVERS):
                # for dst in range (self.NUM_OF_SERVERS): 
                    # if ((edge[0],edge[1]) in links_of_path[src][dst]): # Does link appear in the path from src to dst
                        # paths_using_this_edge.append ((src, dst)) # Yep --> append it to the list of paths in which this link appears
            # self.G[edge[0]][edge[1]]['paths using this edge'] = paths_using_this_edge

        # self.path_delay[s][d] will hold the prop' delay of the path from server s to server d
        # self.path_delay   = np.zeros ([self.NUM_OF_SERVERS, self.NUM_OF_SERVERS]) 
        # self.path_bw_cost = np.zeros ([self.NUM_OF_SERVERS, self.NUM_OF_SERVERS])
        # for s in range (self.NUM_OF_SERVERS):
        #     for d in range (self.NUM_OF_SERVERS):
        #         if (s == d):
        #             continue
        #         self.path_delay   [s][d] = sum (self.G[shortest_path[s][d][hop]][shortest_path[s][d][hop+1]]['delay'] for hop in range (len(shortest_path[s][d])-1))
        #         self.path_bw_cost [s][d] = sum (self.G[shortest_path[s][d][hop]][shortest_path[s][d][hop+1]]['cost']  for hop in range (len(shortest_path[s][d])-1))
                
        # calculate the network delay from a leaf to a node in each level,  
        # assuming that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.   
        # leaf = self.G.number_of_nodes()-1 # when using networkx and a balanced tree, self.path_delay[self.G[nodes][-1]] is surely a leaf (it's the node with highest ID).
        # self.netw_delay_from_leaf_to_lvl = [ self.path_delay[leaf][shortest_path[leaf][root][lvl]] for lvl in range (0, self.tree_height+1)]

    def __init__ (self, verbose = -1):
        """
        Init a toy example - topology (e.g., chains, VMs, target_delays etc.).
        """
        
        self.verbose                = verbose
        
        # Network parameters
        self.tree_height           = 2
        self.children_per_node     = 2 # num of children of every non-leaf node
        self.uniform_mig_cost      = 1
        self.Lmax                  = 0
        self.uniform_Tpd           = 2
        self.uniform_link_cost     = 1
        
        # Names of input files for the users' data, locations and / or current access points
        self.usrss_data_file_name  = "res.usr" #input file containing the target_delays and traffic of all users
        self.usrs_loc_file_name    = "my_mob_sim.loc"  #input file containing the locations of all users along the simulation
        self.usrs_ap_file_name     = 'mob_sim.ap' #input file containing the APs of all users along the simulation
        
        # Names of output files
        self.log_file_name         = "run.log" 
        self.res_file_name         = "run.res" 
        self.gen_parameterized_tree ()

        # Flags indicators for writing output to results and log files
        self.write_to_cfg_file = False
        self.write_to_log_file = True
        
        # Flags indicators for writing output to various LP solvers
        self.write_to_prb_file = False # When true, will write outputs to a .prb file. - ".prb" - A .prb file may solve an LP problem using the online Eq. solver: https://online-optimizer.appspot.com/?model=builtin:default.mod
        self.write_to_mod_file = False # When true, will write to a .mod file, fitting to IBM CPlex solver       
        self.write_to_lp_file  = True  # When true, will write to a .lp file, which allows running Cplex using a Python's api.       

    def simulate (self):
        self.rd_usr_data ()
        self.CPUAll_once ()       

        # reset S_u, Hs        
        for usr in self.usrs:
            usr.S_u = [] 
        for s in self.G.nodes():
            self.G.nodes[s]['Hs'] = []

        # Open input and output files
        self.ap_file  = open ("../res/" + self.usrs_ap_file_name, "r")  
        if (self.write_to_log_file):
            self.init_log_file()

        # init self.X (current placement): self.X[u][s] = True will indicate that user u is placed on server s     
        self.X = np.zeros ([len (self.usrs), len (self.G.nodes())], dtype = 'bool')

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
        
        self.not_X = np.invert (self.X)
        
 
        calc_upr_bnd_rsrc_aug = lambda self: np.max ([usr.C_u for usr in self.usrs]) / np.min ([np.min (usr.B) for usr in self.usrs])

        R = self.calc_upr_bnd_rsrc_aug () 
        self.bottom_up (R)
        if (not(self.found_sol())):
            print ('did not find a feasible sol even with maximal rsrc aug')
            exit ()
                   
        ub = R # upper-bnd on the rsrc aug' that may be required
        lb = 1
        
        while ub > lb + 0.5:
            
            R = (ub+lb)/2
            self.bottom_up(R) 
            
            if (self.found_sol()):
                # self.print_sol(R)
                # print ('B4 reduceCost: R = {}, phi = {}' .format (self.calc_sol_rsrc_aug (R), self.calc_sol_cost()) )
                # self.reduce_cost ()
                # print ('after reduceCost: R = {}, phi = {}' .format (self.calc_sol_rsrc_aug (R), self.calc_sol_cost()) )
                printf (self.log_output_file, 'B4 push-up\n')
                self.print_sol(R)
                self.push_up ()
                printf (self.log_output_file, 'After push-up\n')
                self.print_sol(R)
                ub = R
        
            else:
                print ('R = {}' .format (R))
                lb = R
                
        return
    
    def bottom_up (self, R):
        """
        Bottom-up alg'. 
        Looks for a feasible sol'.
        Input: R, a mult' factor on the amount of CPU units in each server.
        """
        
        self.reset_sol()

        # init a(s), the number of available CPU in each server 
        for s in self.G.nodes():
            self.G.nodes[s]['a'] = R * self.G.nodes[s]['cpu cap']
                
        for s in range (len (self.G.nodes())-1, -1, -1): # for each server s, in an increasing order of levels (DFS).
            lvl = self.G.nodes[s]['lvl']
            Hs = [self.usrs[u] for u in self.G.nodes[s]['Hs'] if (self.usrs[u].lvl == -1)]
            for usr in sorted (Hs, key = lambda usr : usr.L): # for each chain in Hs, in an increasing order of level ('L')                   
                if (self.G.nodes[s]['a'] > usr.B[lvl]): 
                    self.Y[usr.id][s] = True
                    usr.lvl = lvl
                    self.G.nodes[s]['a'] -= usr.B[lvl]
                elif (len (usr.B) == lvl):
                    self.Y = np.zeros ([len (self.usrs), len (self.G.nodes())], dtype = 'bool')
                    return
                 
   
    def rd_AP_line (self, line):
        """
        Read a line in an ".ap" file.
        An AP file details for each time t, the current AP of each user.
        The user number and its current AP are written as a tuple. e.g.: 
        (3, 2) means that user 3 is currently covered by user 2.
        After reading the tuples, the function assigns each chain to its relevant list of chains, H-s.  
        """
        splitted_line = line[0].split ("\n")[0].split (")")
        for tuple in splitted_line:
            tuple = tuple.split("(")
            if (len(tuple) > 1):
                tuple   = tuple[1].split (",")
                usr_id  = int(tuple[0])
                if (usr_id > len (self.usrs)-1):
                    print ('error: encountered usr num {}, where by res.usr file, there are only {} users' .format (tuple[0], len(self.usrs)))
                    exit  ()
                usr   = self.usrs[usr_id]
                AP_id = int(tuple[1])
                if (AP_id > self.num_of_leaves):
                    print ('error: encountered AP num {} in the input file, but in the tree there are only {} leaves' .format (AP_id, self.num_of_leaves))
                    exit  ()

                s = self.ap2s[AP_id]
                usr.S_u.append (s)
                self.G.nodes[s]['Hs'].append(usr_id)
                for lvl in (range (len(usr.B)-1)):
                    s = self.parent_of(s)
                    usr.S_u.append (s)
                    self.G.nodes[s]['Hs'].append(usr_id)
                        

    def calc_SS_sol_total_cost (self):
        """
        Calculate the total cost of an SS (single-server pver-chain) full solution.
        """
        total_cost = 0
        for chain in range(self.NUM_OF_CHAINS):
            total_cost += self.CPU_cost[self.chain_nxt_loc[chain]] * self.chain_nxt_total_alloc[chain] + \
                        self.path_bw_cost[self.PoA_of_user[chain]]  [self.chain_nxt_loc[chain]] * self.lambda_v[chain][0] + \
                        self.path_bw_cost[self.chain_nxt_loc[chain]][self.PoA_of_user[chain]]   * self.lambda_v[chain][self._in_chain[chain]] + \
                        (self.chain_cur_loc[chain] != self.chain_nxt_loc[chain]) * self.chain_mig_cost[chain]
            
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

