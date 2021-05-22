import networkx as nx
import numpy as np
import math
import itertools 
import time
import random
import heapq

from printf import printf
# import Check_sol
# import obj_func
from _overlapped import NULL
# from solve_problem_by_Cplex import solve_problem_by_Cplex
from networkx.algorithms.threshold import shortest_path
from cmath import sqrt
from numpy import int

class usr_c (object):
    """
    class for of "user" 
    """ 
    def __init__ (self, id, cur_cpu = -1):
        self.id      = id
        self.cur_cpu = cur_cpu
    def __lt__ (self, other):
        return (self.cur_cpu < other.cur_cpu)



class SFC_mig_simulator (object):

    # An inline function that returns the parent of a given server
    parent_of = lambda self, s : self.G.nodes[s]['prnt']

    # # Find the server on which a given user is located, by the Y_lvl_of
    loc_of_user = lambda self, u : self.usr[u]['S_u'][self.Y_lvl_of[u]]

    # An inline func' for calculating the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    calc_chain_cost_homo = lambda self, lvl, not_X, usr: self.link_cost_of_SSP_at_lvl[lvl] + not_X * self.uniform_mig_cost * len (usr['theta times lambda']) + self.CPU_cost_at_lvl[lvl] * usr['B'][lvl]    
     
    # An inline func' for calculating the (maximal) rsrc aug' used by the current solution
    calc_sol_rsrc_aug = lambda self, R : np.max ([(R * self.G.nodes[s]['cpu cap'] - self.G.nodes[s]['a']) / self.G.nodes[s]['cpu cap'] for s in self.G.nodes()])
         
    # An inline func' for calculating the CPU used in practice in each server in the current solution
    used_cpu_in = lambda self, R : [(R * self.G.nodes[s]['cpu cap'] - self.G.nodes[s]['a']) for s in self.G.nodes()]

    # reset the solution, self.Y (init it to a matrix of "False")
    reset_sol   =  lambda self : np.zeros ([len (self.usr), len (self.G.nodes())], dtype = 'bool')

    # calculate the cost of the current solution (self.Y)
    calc_sol_cost = lambda self: sum ([self.calc_chain_cost_homo (self.Y_lvl_of[u], not (self.X[u][self.loc_of_user(u)]), self.usr[u]) for u in range(len(self.usr))])
   
    # calculate the proved upper bnd on the rsrc aug that bottomUp may need to find a feasible sol, given such a sol exists for the non-augmented prob'
    calc_upr_bnd_rsrc_aug = lambda self: np.max ([usr['C_u'] for usr in self.usr]) / np.min ([np.min (usr['B']) for usr in self.usr])

    # Returns True iff the solution self.Y schedules all the chain.
    # Note: the function does NOT check whether self.Y adheres to all the constraints (e.g., CPU cap', link cap', delay).
    # This is assumed to be True by construction.
    found_sol = lambda self: sum (sum (self.Y)) >= len (self.usr)

    def print_cost_per_usr (self):
        """
        For debugging / analysing only.
        print the cost of each chain. 
        """
        
        for u in range (len (self.usr)):
            chain_cost = self.calc_chain_cost_homo (self.Y_lvl_of[u], not (self.X[u][self.loc_of_user(u)]), self.usr[u])  
            print ('cost of usr {} = {}' .format (u, chain_cost))
        

    def print_sol (self, R):
        """
        print a formatted solution for the allocation and placement prob' 
        """
        used_cpu_in = self.used_cpu_in (R)
        print ('\\\ format: s : used / C_s   chains[u1, u2, ...]')
        print ('\\\ where: s = number of server. used = capacity used by the sol on server s. C_S = non-augmented capacity of s. u1, u2, ... = chains placed on s.' )
        print ('Rsrc aug = {:.2f}', R)
        for s in self.G.nodes():
            print ('{}: {} / {}\t chains {}' .format (s, used_cpu_in[s], self.G.nodes[s]['cpu cap'], [u for u in range (len(self.usr)) if self.Y[u][s] ] ))
        
    def push_up (self):
        """
        Push chains up: take a feasible solution, and greedily try pushing each chain as high as possible in the tree. 
        Do that when chains are sorted in a decreasing order of the # of CPU units they currently use.
        """
        
        # Assume here that the available cap' at each server 'a' is already calculated by the alg' that was run 
        # before calling push-up ()
        usrs_heap = []
        for usr in self.usr: #range (len(self.usr)):
            heapq.heappush(usrs_heap, usr_c(id = usr['id'], cur_cpu = usr['B'][self.Y_lvl_of[usr['id']]])) 

        while (1):
            usr = heapq.nlargest (1, usrs_heap)
            usr = usr[0]
            # for lvl in range ()
            # print (usr.id)
            exit ()
        #     for u in range(len(self.usr)):
        #         usr = self.usr[u]       
        #         for lvl in range(len (usr['B'])): # for each level in which there's a delay-feasible server for this usr
        #             if (self.G.nodes[usr['S_u'][lvl]]['a'] >= usr['B'][lvl]): # if there's enough available space to move u to level lvl 
        #                 reduction = self.calc_chain_cost_homo (self.Y_lvl_of[u], self.not_X[u][usr['S_u'][self.Y_lvl_of[u]]], usr) - self.calc_chain_cost_homo (lvl, self.not_X[u][usr['S_u'][lvl]], usr)
        #                 if (reduction > max_reduction):  
        #                     u_star = u
        #                     lvl_star = lvl
        #                     max_reduction = reduction 
        #     if (max_reduction == 0): # cannot decrease cost anymore
        #         break
        #
        #     # move u_star from lvl to lvl_star, and update Y, a, accordingly
        #     usr2mov            = self.usr[u_star]
        #     old_lvl_of_usr2mov = self.Y_lvl_of[u_star]
        #     self.G.nodes [usr2mov['S_u'][old_lvl_of_usr2mov]] ['a'] += usr2mov ['B'][old_lvl_of_usr2mov] # inc the available CPU at the prev loc of the moved usr  
        #     self.G.nodes [usr2mov['S_u'][lvl_star]]           ['a'] -= usr2mov ['B'][lvl_star]           # dec the available CPU at the new  loc of the moved usr  
        #     self.Y_lvl_of[u_star] = lvl_star # update self.Y_lvl_of accordingly. Note: we don't update self.Y for now.
        #     # print ('u_star = {}, lvl_star = {}, max_reduction = {}' .format(u_star, lvl_star, max_reduction))
        #
        # # Update self.Y
        # self.Y = self.reset_sol () 
        # for u in range(len(self.usr)):
        #     usr = self.usr[u]       
        #     self.Y[u] [usr['S_u'][self.Y_lvl_of[u]] ] = True   
         
    def reduce_cost (self):
        """
        Reduce cost alg': take a feasible solution, and greedily decrease the cost 
        by changing the placement of a single chain, using a gradient method, as long as this is possible.
        """
        
        # Assume here that the available cap' at each server 'a' is already calculated by the alg' that was run 
        # before calling reduce_cost ()
        while (1):
            u_star = -1; lvl_star = -1; max_reduction = 0 # init default values for the maximal reduction cost func', and for the argmax indices 
            for u in range(len(self.usr)):
                usr = self.usr[u]       
                for lvl in range(len (usr['B'])): # for each level in which there's a delay-feasible server for this usr
                    if (self.G.nodes[usr['S_u'][lvl]]['a'] >= usr['B'][lvl]): # if there's enough available space to move u to level lvl 
                        reduction = self.calc_chain_cost_homo (self.Y_lvl_of[u], self.not_X[u][usr['S_u'][self.Y_lvl_of[u]]], usr) - self.calc_chain_cost_homo (lvl, self.not_X[u][usr['S_u'][lvl]], usr)
                        if (reduction > max_reduction):  
                            u_star = u
                            lvl_star = lvl
                            max_reduction = reduction 
            if (max_reduction == 0): # cannot decrease cost anymore
                break
            
            # move u_star from lvl to lvl_star, and update Y, a, accordingly
            usr2mov            = self.usr[u_star]
            old_lvl_of_usr2mov = self.Y_lvl_of[u_star]
            self.G.nodes [usr2mov['S_u'][old_lvl_of_usr2mov]] ['a'] += usr2mov ['B'][old_lvl_of_usr2mov] # inc the available CPU at the prev loc of the moved usr  
            self.G.nodes [usr2mov['S_u'][lvl_star]]           ['a'] -= usr2mov ['B'][lvl_star]           # dec the available CPU at the new  loc of the moved usr  
            self.Y_lvl_of[u_star] = lvl_star # update self.Y_lvl_of accordingly. Note: we don't update self.Y for now.
            # print ('u_star = {}, lvl_star = {}, max_reduction = {}' .format(u_star, lvl_star, max_reduction))
        
        # Update self.Y
        self.Y = self.reset_sol () 
        for u in range(len(self.usr)):
            usr = self.usr[u]       
            self.Y[u] [usr['S_u'][self.Y_lvl_of[u]] ] = True   
    
    # def calc_chain_cost_hetro (self):
    #     """
    #     Calculate the cost of locating chain u on server s, per each pair u, s
    #     """
    #     self.cost_per_usr = np.empty (len(self.usr), dtype = object)
    #     for u in range(len(self.usr)):
    #         self.cost_per_usr[u] = []
    #         usr = self.usr[u]       
    #         for lvl in range(len (usr['B'])): # for each level in which there's a delay-feasible server for this usr
    #             self.cost_per_usr[u].append (self.calc_chain_cost_homo (lvl, self.not_X[u][usr['S_u'][lvl]], usr))
    #             # Below is a calculation of the cost for the fully-hetero' case. 
    #             # s = usr['S_u'][lvl]
    #             # self.cost_per_usr[u].append (usr['B'][lvl] * self.G.nodes[s]['cpu cost'] + # comp' cost 
    #             #                              usr['mig cost'][lvl] * self.not_X[u][s] + #mig' cost
    #             #                              self.G.nodes[s]['link cost']) # netw' cost
        
    def CPUAll_once (self): 
        """
        CPUAll algorithm:
        calculate the minimal CPU allocation required for each chain u, when the highest server on which u is located is s.
        The current implementation assumes that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.
        This version of the alg' assumes a balanced homogeneous tree, 
        so that the netw' delay between every two servers is unequivocally defined by their levels.
        """
        for usr in self.usr:
            slack = [usr['target delay'] -  usr['delay to PoA'] - self.link_delay_of_SSP_at_lvl[lvl] for lvl in range (self.tree_height+1)]
            slack = [slack[lvl] for lvl in range(self.tree_height+1) if slack[lvl] > 0] # trunc all servers with negative slack, which are surely delay-INfeasible
            #u['L'] = -1 # Highest server which is delay-feasible for u
            usr['B'] = [] # usr['B'] will hold a list of the budgets required for placing u on each level 
            mu = np.array ([math.floor(usr['theta times lambda'][i]) + 1 for i in range (len(usr['theta times lambda']))]) # minimal feasible budget
            lvl = 0 
            for lvl in range(len(slack)):
                while (sum(mu) <= usr['C_u']): # The SLA still allows increasing this user's CPU allocation
                    if (sum (1 / (mu[i] - usr['theta times lambda'][i]) for i in range(len(mu))) <= slack[lvl]):  
                        usr['B'].append(sum(mu))
                        # Can save now the exact vector mu; for now, no need for that, as we're interested only in the sum
                        break
                    argmax = np.argmax (np.array ([1 / (mu[i] - usr['theta times lambda'][i]) - 1 / (mu[i] + 1 - usr['theta times lambda'][i]) for i in range(len(mu))]))
                    mu[argmax] = mu[argmax] + 1
            usr['L'] = lvl # usr['L'] holds the highest lvl on which it's possible to locate this usr
            
        # for u in range(len (self.usr)):
        #     print ('mu[{}] = {}' .format (u, self.usr[u]['B']))

    def rd_usr_data (self):
        """
        Read the input about the users (target delay, traffic), and write it to the appropriate fields in self.
        The input data is read from the file self.usr_loc_file_name. 
        """
        usrs_data_file = open ("../res/" + self.usrs_data_file_name,  "r")
        self.usr = np.empty (self.MAX_NUM_OF_USRS, dtype=object)
        
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

                elif (splitted_line[1] == "delay_to_PoA"):              
                    self.usr[u]['delay to PoA'] = float (line.split("=")[1].rstrip())
                    
                elif (splitted_line[1] == "target_delay"):              
                    self.usr[u]['target delay'] = float (line.split("=")[1].rstrip())
                  
                elif (splitted_line[1] == "mig_cost"):              
                    mig_cost = line.split("=")[1].rstrip().split(",")
                    self.usr[u]['mig cost'] = [float (mig_cost[i]) for i in range (len (mig_cost)) ]
                    
                elif (splitted_line[1] == "C_u"):              
                    self.usr[u]['C_u'] = int (line.split("=")[1].rstrip())
                    
                self.usr[u]['id'] = u
                    
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
        Init a toy example - topology (e.g., chains, VMs, target delays etc.).
        """
        
        self.verbose                = verbose
        
        # Network parameters
        self.tree_height            = 2
        self.children_per_node      = 2 # num of children of every non-leaf node
        self.uniform_link_capacity  = 100
        self.uniform_mig_cost       = 1
        self.Lmax                   = 0
        self.uniform_Tpd            = 2
        self.uniform_link_cost      = 1
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
        self.CPUAll_once ()       

        # reset S_u, Hs        
        for usr in self.usr:
            usr['S_u'] = [] 
        for s in self.G.nodes():
            self.G.nodes[s]['Hs'] = []

        self.ap_file  = open ("../res/" + self.usr_ap_file_name, "r")  

        # init self.X (current placement).
        # self.X[u][s] = True will indicate that user u is placed on server s     
        self.X = np.zeros ([len (self.usr), len (self.G.nodes())], dtype = 'bool')

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
            
            if (not(self.found_sol())):
                lb = R
                
            else:
                # self.print_sol(R)
                # print ('B4 reduceCost: R = {}, phi = {}' .format (self.calc_sol_rsrc_aug (R), self.calc_sol_cost()) )
                self.push_up ()
                # print ('after reduceCost: R = {}, phi = {}' .format (self.calc_sol_rsrc_aug (R), self.calc_sol_cost()) )
                ub = R
        
        return
    
    def bottom_up (self, R):
        """
        Bottom-up alg'. 
        Looks for a feasible sol'.
        Input: R, a mult' factor on the amount of CPU units in each server.
        """
        
        # init a(s), the number of available CPU in each server 
        for s in self.G.nodes():
            self.G.nodes[s]['a'] = R * self.G.nodes[s]['cpu cap']
    
        # init self.Y (the placement to be found).
        # self.Y[u][s] = True will indicate that user u is placed on server s     
        self.Y = self.reset_sol ()
        self.Y_lvl_of = (-1) * np.ones  (len (self.usr), dtype = 'int8') #self.Y_lvl_of[u] will hold the level on which chain u is placed by sol' Y. 
        
        # Mark all users as not placed yet
        for usr in self.usr:
            usr['placed'] = False
    
        for s in range (len (self.G.nodes())-1, -1, -1): # for each server s, in an increasing order of levels
            lvl = self.G.nodes[s]['lvl']
            Hs = [self.usr[u] for u in self.G.nodes[s]['Hs'] if (not(self.usr[u]['placed']))]
            for usr in sorted (Hs, key = lambda usr : usr['L']): # for each chain in Hs, in an increasing order of level ('L')                   
                if (self.G.nodes[s]['a'] > usr['B'][lvl]): 
                    self.Y[usr['id']][s] = True
                    self.Y_lvl_of[usr['id']] = lvl 
                    usr['lvl'] = lvl
                    usr['placed'] = True
                    self.G.nodes[s]['a'] -= usr['B'][lvl]
                elif (len (usr['B']) == lvl):
                    self.Y = np.zeros ([len (self.usr), len (self.G.nodes())], dtype = 'bool')
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
                if (usr_id > len (self.usr)-1):
                    print ('error: encountered usr num {}, where by res.usr file, there are only {} users' .format (tuple[0], len(self.usr)))
                    exit  ()
                usr   = self.usr[usr_id]
                AP_id = int(tuple[1])
                if (AP_id > self.num_of_leaves):
                    print ('error: encountered AP num {} in the input file, but in the tree there are only {} leaves' .format (AP_id, self.num_of_leaves))
                    exit  ()

                s = self.ap2s[AP_id]
                usr['S_u'].append (s)
                self.G.nodes[s]['Hs'].append(usr_id)
                for lvl in (range (len(usr['B'])-1)):
                    s = self.parent_of(s)
                    usr['S_u'].append (s)
                    self.G.nodes[s]['Hs'].append(usr_id)
                        

    def calc_SS_sol_total_cost (self):
        """
        Calculate the total cost of an SS (single-server pver-chain) full solution.
        """
        total_cost = 0
        for chain in range(self.NUM_OF_CHAINS):
            total_cost += self.CPU_cost[self.chain_nxt_loc[chain]] * self.chain_nxt_total_alloc[chain] + \
                        self.path_bw_cost[self.PoA_of_user[chain]]  [self.chain_nxt_loc[chain]] * self.lambda_v[chain][0] + \
                        self.path_bw_cost[self.chain_nxt_loc[chain]][self.PoA_of_user[chain]]   * self.lambda_v[chain][self.
_in_chain[chain]] + \
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

