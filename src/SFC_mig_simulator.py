import networkx as nx
import numpy as np
import math
import itertools 
import time
import random
import heapq

from usr_c import usr_c # class of the users
from decision_var_c import decision_var_c # class of the decision variables
from printf import printf
# import Check_sol
# import obj_func
# from _overlapped import NULL
# from solve_problem_by_Cplex import solve_problem_by_Cplex
from scipy.optimize import linprog
# from networkx.algorithms.threshold import shortest_path
from cmath import sqrt

# Levels of verbose (which output is generated)
VERBOSE_NO_OUTPUT             = 0
VERBOSE_ONLY_RES              = 1 # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_RES_AND_LOG           = 2 # Write to a file the total cost and rsrc aug. upon every event + write a detailed ".log" file

class SFC_mig_simulator (object):


    #############################################################################
    # Inline functions
    #############################################################################
    # Returns the parent of a given server
    parent_of = lambda self, s : self.G.nodes[s]['prnt']

    # # Find the server on which a given user is located, by its lvl
    loc_of_user = lambda self, u : self.usrs[u].S_u[self.usrs[u].lvl]

    # calculate the cost of the current solution (self.Y)
    calc_sol_cost = lambda self: sum ([self.calc_chain_cost_homo (usr, usr.lvl) for usr in self.usrs])
   
    # calculate the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    calc_chain_cost_homo = lambda self, usr, lvl: self.link_cost_of_SSP_at_lvl[lvl] + (not (self.X[usr.id][usr.S_u[usr.lvl]])) * self.uniform_mig_cost * len (usr.theta_times_lambda) + self.CPU_cost_at_lvl[lvl] * usr.B[lvl]    
          
    # Calculate the (maximal) rsrc aug' used by the current solution, using a single (scalar) R
    calc_sol_rsrc_aug = lambda self, R : np.max ([(R * self.G.nodes[s]['cpu cap'] - self.G.nodes[s]['a']) / self.G.nodes[s]['cpu cap'] for s in self.G.nodes()])
         
    # Calculate the CPU used in practice in each server in the current solution
    used_cpu_in = lambda self, R : [(R * self.G.nodes[s]['cpu cap'] - self.G.nodes[s]['a']) for s in self.G.nodes()]

    # calculate the proved upper bnd on the rsrc aug that bottomUp may need to find a feasible sol, given such a sol exists for the non-augmented prob'
    calc_upr_bnd_rsrc_aug = lambda self: np.max ([usr.C_u for usr in self.usrs]) / np.min ([np.min (usr.B) for usr in self.usrs])

    # Returns True iff the solution self.Y schedules all the chain.
    # Note: the function does NOT check whether self.Y adheres to all the constraints (e.g., CPU cap', link cap', delay).
    # This is assumed to be True by construction.
    found_sol = lambda self: sum (sum (self.Y)) >= len (self.usrs)

    def solveByLp (self, changed_usrs):
        """
        Example of solving a problem using Python's LP capabilities.
        """
        self.decision_vars  = []
        id                  = 0
        for usr in changed_usrs:
            for lvl in range(len(usr.B)):
                self.decision_vars.append (decision_var_c (id     = id, 
                                                           usr    = usr.id, 
                                                           lvl    = lvl, 
                                                           server = usr.S_u[lvl]))
                id += 1

        # Adding the CPU cap' constraints
        # A will hold the decision vars' coefficients. b will hold the bound: the constraints are: Ax<=b 
        A = np.zeros ([len (self.G.nodes), len(self.decision_vars)], dtype = 'uint16')
        for s in self.G.nodes():
            for decision_var in filter (lambda item : item.server == s, self.decision_vars):
                A[s][decision_var.id] = self.usrs[decision_var.usr][decision_var.lvl]
            # if (A_s_non_zeros = []) # no constraint on this server (the total demand by users that may use this server < its CPU capacity)
            #     self.decision_vars = filter (lambda item: item.server != s, self.decision_vars) # remove all decision vars related to this server, as they're 
        print ('A = ', A)
        # print ('A = ', np.array(A))
        # print ('A shape = ', np.array(A).shape)
        # print ('b = ', b)
        # b = np.empty (len(self.G.nodes), dytpe='uint16')
        #     b[s] (self.G.nodes[s]['cpu cap'])
        # print ('c = ', [decision_var.cost for decision_var in self.decision_vars])

        # c = [-1, 4]
        # #A = [[-3, 1], [1, 2]]
        # A = np.ones ([2, 2])
        # # print (A)
        # b = [6, 4]
        # x0_bounds = (None, None)
        # x1_bounds = (-3, None)
        # res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
        exit ()
        res = linprog ([decision_var.cost for decision_var in self.decision_vars], 
                       A_ub   = A, 
                       b_ub   = [s['cpu cap'] for s in self.G.nodes[s]], 
                       bounds = [np.zeros (len(self.decision_vars)), np.ones (len(self.decision_vars))])


    def reset_sol (self):
        """"
        reset the solution, including:
        - self.Y (init it to a matrix of "False")
        - usr.lvl
        """
        self.Y = np.zeros ([len (self.usrs), len (self.G.nodes())], dtype = 'bool')
        for usr in self.usrs:
            usr.lvl = -1

    def print_cost_per_usr (self):
        """
        For debugging / analysis:
        print the cost of each chain. 
        """
        
        for usr in self.usrs:
            chain_cost = self.calc_chain_cost_homo (usr, usr.lvl)  
            print ('cost of usr {} = {}' .format (u, chain_cost))
        
    def init_output_file (self, file_name):
        """
        Open an output file for writing, overwriting previous content in that file 
        """
        with open('../res/' + file_name, 'w') as FD:
            FD.write('')                
        FD  = open ('../res/' + file_name,  "w")
        return FD

    def init_res_file (self):
        """
        Open the res file for writing and write initial comments line on it
        """
        self.res_output_file = self.init_output_file(self.res_file_name)

    def init_log_file (self):
        """
        Open the log file for writing and write initial comments lines on it
        """
        self.log_output_file = self.init_output_file(self.log_file_name)
        printf (self.log_output_file, '// format: s : used / C_s   chains[u1, u2, ...]\n')
        printf (self.log_output_file, '// where: s = number of server. used = capacity used by the sol on server s.\n//C_s = non-augmented capacity of s. u1, u2, ... = chains placed on s.\n' )

    def print_sol (self):
        """
        print a solution for DPM to the output log file 
        """
        printf (self.log_output_file, 'phi = {:.0f}\n' .format (self.calc_sol_cost()))
        used_cpu_in = np.array ([self.G.nodes[s]['cur RCs'] - self.G.nodes[s]['a'] for s in self.G.nodes])
        
        for s in self.G.nodes():
            printf (self.log_output_file, 's{} : used cpu={:.0f}, Cs={}\t chains {}\n' .format (
                    s,
                    sum ([usr.B[usr.lvl] for usr in self.usrs if self.Y[usr.id][s] ] ),
                    self.G.nodes[s]['cpu cap'],
                    [usr.id for usr in self.usrs if self.Y[usr.id][s] ]))

    def print_heap (self):
        """
        print the id, level and CPU of each user in a heap.
        Used for debugginign only.
        """
        for usr in self.usrs:
            print ('id = {}, lvl = {}, CPU = {}' .format (usr.id, usr.lvl, usr.B[usr.lvl]))
        print ('')
        
    def push_up (self):
        """
        Push chains up: take a feasible solution, and greedily try pushing each chain as high as possible in the tree. 
        Do that when chains are sorted in a decreasing order of the # of CPU units they're currently using.
        """
        
        # Assume here that the available cap' at each server 'a' is already calculated by the alg' that was run 
        # before calling push-up ()
        heapq._heapify_max(self.usrs)
 
        n = 0  
        while n < len (self.usrs):
            usr = self.usrs[n]
            for lvl in range (len(usr.B)-1, usr.lvl, -1): #  
                if (self.G.nodes[usr.S_u[lvl]]['a'] >= usr.B[lvl]): # if there's enough available space to move u to level lvl                     
                    self.G.nodes [usr.S_u[usr.lvl]] ['a'] += usr.B[usr.lvl] # inc the available CPU at the prev loc of the moved usr  
                    self.G.nodes [usr.S_u[lvl]]     ['a'] -= usr.B[lvl]     # dec the available CPU at the new  loc of the moved usr
                    usr.lvl = lvl # update usr.lvl accordingly. Note: we don't update self.Y for now.
                    
                    # update the moved usr's location in the heap
                    self.usrs[n] = self.usrs[-1] # replace the usr to push-up with the last usr in the heap
                    self.usrs.pop()
                    heapq.heappush(self.usrs, usr)
                    n = -1 # succeeded to push-up a user, so next time should start from the max (which may now succeed to move)
                    break
            n += 1
        self.update_Y()
                            
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
    
    def update_Y (self):
        """
        Update the solution Y according to the values of the field ".lvl" in each usr
        """
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
            
    def rd_usr_data (self):
        """
        Read the input about the users (target_delay, traffic), and write it to the appropriate fields in self.
        The input data is read from the file self.usrs_loc_file_name. 
        """
        usrs_data_file = open ("../res/" + self.usrs_data_file_name,  "r")
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
        
        # INPUT / OUTPUT FILES
        self.verbose = VERBOSE_RES_AND_LOG
        
        # Names of input files for the users' data, locations and / or current access points
        self.usrs_data_file_name  = "res.usr" #input file containing the target_delays and traffic of all users
        self.usrs_loc_file_name   = "my_mob_sim.loc"  #input file containing the locations of all users along the simulation
        self.usrs_ap_file_name    = 'mob_sim.ap' #input file containing the APs of all users along the simulation
        
        # Names of output files
        if (self.verbose == VERBOSE_ONLY_RES or self.verbose == VERBOSE_RES_AND_LOG):
            self.res_file_name     = "run.res" 
        if (self.verbose == VERBOSE_RES_AND_LOG):
            self.log_file_name = "run.log" 
        self.gen_parameterized_tree ()

        # Flags indicators for writing output to results and log files
        self.write_to_cfg_file = False
        
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
        if (self.verbose == VERBOSE_RES_AND_LOG):
            self.init_log_file()

        # init self.X (current placement): self.X[u][s] = True will indicate that user u is placed on server s     
        self.X = np.zeros ([len (self.usrs), len (self.G.nodes())], dtype = 'bool')

        for line in self.ap_file: 

            # Ignore comments lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue

            splitted_line = line.split (" ")

            if (splitted_line[0] == "time"):
                if (self.verbose == VERBOSE_RES_AND_LOG):
                    printf (self.log_output_file, '\ntime = {}\n**************************************\n' .format (splitted_line[2]))
                continue
        

            self.rd_AP_line(splitted_line)
            self.solveByLp(self.usrs)
            exit ()
            self.alg_top ()
    
    def alg_top (self):
        """
        Top-level alg'
        """
        
        max_R = self.calc_upr_bnd_rsrc_aug () 
        
        # init cur RCs and a(s) to the number of available CPU in each server, assuming maximal rsrc aug' 
        for s in self.G.nodes():
            self.G.nodes[s]['cur RCs'] = math.ceil (max_R * self.G.nodes[s]['cpu cap']) 
            self.G.nodes[s]['a']       = self.G.nodes[s]['cur RCs'] #currently-available rsrcs at server s  

        self.bottom_up ()
        if (not(self.found_sol())):
            print ('did not find a feasible sol even with maximal rsrc aug')
            exit ()
        if (self.verbose == VERBOSE_RES_AND_LOG):
            printf (self.log_output_file, 'Initial solution:\n')
            self.print_sol()
                   
        ub = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()]) # upper-bnd on the (augmented) cpu cap' that may be required
        lb = np.array([self.G.nodes[s]['cpu cap'] for s in self.G.nodes()]) # lower-bnd on the (augmented) cpu cap' that may be required
        while True: 
            
            for s in self.G.nodes():
                self.G.nodes[s]['a'] = math.ceil (0.5*(ub[s] + lb[s])) 
            if (np.array([self.G.nodes[s]['a'] for s in self.G.nodes()]) == np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()])).all():
                break
            
            for s in self.G.nodes():
                self.G.nodes[s]['cur RCs'] = self.G.nodes[s]['a'] 

            self.bottom_up()
            # printf (self.log_output_file, 'After BU:\n') 
            # self.print_sol()

            if (self.found_sol()):
                ub = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()])
        
            else:
                lb = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()])
                
        if (self.verbose == VERBOSE_RES_AND_LOG):
            printf (self.log_output_file, 'B4 push-up:\n')
            self.print_sol()
            
        # update the available capacity a at each server
        for s in self.G.nodes():
            self.G.nodes[s]['a'] = self.G.nodes[s]['cur RCs'] - sum ([usr.B[usr.lvl] for usr in self.usrs if self.Y[usr.id][s] ])
        self.push_up ()
        if (self.verbose == VERBOSE_RES_AND_LOG):
            printf (self.log_output_file, 'After push-up:\n')
            self.print_sol()

    def bottom_up (self):
        """
        Bottom-up alg'. 
        Looks for a feasible sol'.
        Input: R, a mult' factor on the amount of CPU units in each server.
        """
        
        self.reset_sol()

        for s in range (len (self.G.nodes())-1, -1, -1): # for each server s, in an increasing order of levels (DFS).
            lvl = self.G.nodes[s]['lvl']
            Hs = [usr for usr in self.G.nodes[s]['Hs']  if (usr.lvl == -1)]
            for usr in sorted (Hs, key = lambda usr : len(usr.B)): # for each chain in Hs, in an increasing order of level ('L')
                #print ('s = ', s, 'a = ', self.G.nodes[s]['a'], 'B = ', usr.B[lvl])                   
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
                self.G.nodes[s]['Hs'].append(usr)
                for lvl in (range (len(usr.B)-1)):
                    s = self.parent_of(s)
                    usr.S_u.append (s)
                    self.G.nodes[s]['Hs'].append(usr)                       
                    

    def calc_sol_cost_SS (self):
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
    Y = [0, 2, 14, 25, 29, 36, 42, 47, 54, 61, 62, 68, 74, 80, 86, 91, 100, 111, 115, 122, 126, 136, 140, 147, 153, 158, 164, 172, 177, 183, 188, 198, 206, 214, 220, 224, 227, 228, 233, 236, 238, 240, 241, 245, 246, 249, 254, 256, 263, 269, 275, 277, 278, 280, 284, 288, 289, 297, 300, 303, 305, 301, 303, 311, 317, 317, 321, 326, 328, 334, 338, 345, 349, 353, 349, 352, 357, 358, 365, 369, 370, 369, 374, 379, 387, 395, 395, 395, 399, 404, 404, 408, 411, 414, 419, 423, 425, 428, 432, 436, 440, 440, 444, 448, 450, 454, 456, 456, 459, 462, 467, 469, 470, 472, 478, 484, 486, 489, 495, 497, 499, 500, 507, 519, 521, 526, 527, 532, 535, 542, 548, 554, 557, 557, 559, 555, 558, 553, 554, 561, 565, 566, 573, 574, 578, 584, 584, 590, 591, 590, 594, 597, 597, 601, 603, 604, 607, 609, 607, 609, 608, 609, 613, 616, 620, 622, 622, 627, 630, 635, 637, 640, 640, 644, 649, 649, 653, 656, 657, 655, 657, 661, 665, 675, 680, 679, 685, 689, 691, 694, 696, 694, 695, 696, 696, 697, 698, 701, 705, 707, 705, 705, 706, 707, 707, 706, 709, 715, 717, 718, 721, 720, 714, 717, 718, 720, 725, 724, 731, 733, 734, 736, 740, 741, 746, 747, 747, 740, 743, 743, 743, 748, 749, 749, 755, 757, 765, 760, 762, 762, 761, 764, 766, 773, 771, 776, 779, 784, 783, 785, 793, 796, 799, 799, 803, 804, 807, 808, 807, 809, 815, 815, 817, 823, 824, 824, 821, 824, 833, 834, 839, 838, 839, 840, 840, 844, 845, 852, 857, 857, 860, 858, 858, 864, 869, 873, 877, 874, 876, 878, 876, 877, 880, 888, 888, 891, 892, 887, 888, 893, 893, 901, 904, 908, 912, 915, 916, 914, 920, 918, 920, 921, 918, 920, 923, 920, 921, 927, 930, 931, 934, 934, 935, 940, 940, 944, 942, 942, 941, 940, 941, 946, 949, 948, 946, 946, 954, 954, 959, 959, 962, 962, 965, 964, 965, 968, 969, 971, 969, 967, 967, 965, 963, 962, 962, 965, 969, 966, 966, 969, 962, 965, 972, 976, 973, 973, 976, 975, 982, 985, 986, 988, 988, 998, 1002, 1009, 1009, 1008, 1011, 1010, 1010, 1015, 1019, 1019, 1016, 1015, 1016, 1015, 1016, 1012, 1016, 1020, 1022, 1018, 1016, 1019, 1011, 1014, 1011, 1013, 1006, 1010, 1014, 1012, 1009, 1010, 1014, 1014, 1011, 1012, 1012, 1008, 1014, 1006, 1009, 1006, 1004, 1008, 1006, 1007, 1012, 1017, 1021, 1021, 1028, 1026, 1022, 1029, 1035, 1038, 1038, 1039, 1043, 1046, 1043, 1044, 1047, 1050, 1051, 1049, 1052, 1053, 1051, 1059, 1060, 1059, 1063, 1065, 1066, 1060, 1061, 1061, 1060, 1060, 1057, 1055, 1050, 1055, 1057, 1052, 1057, 1060, 1061, 1060, 1058, 1059, 1063, 1070, 1069, 1077, 1081, 1081, 1082, 1090, 1090, 1088, 1086, 1087, 1084, 1082, 1084, 1083, 1118, 1142, 1148, 1150, 1155, 1156, 1154, 1157, 1159, 1163, 1164, 1171, 1172, 1167, 1168, 1171, 1175, 1171, 1176, 1174, 1173, 1174, 1175, 1175, 1176, 1180, 1180, 1190, 1194, 1202, 1201, 1201, 1205, 1205, 1217, 1220, 1227, 1228, 1231, 1241, 1241, 1245, 1248, 1254, 1260, 1263, 1264, 1264, 1268, 1269, 1271, 1274, 1275, 1277, 1285, 1288, 1293, 1295, 1303, 1306, 1337, 1365, 1369, 1371, 1385, 1386, 1391, 1401, 1399, 1405, 1417, 1431, 1427, 1433, 1436, 1444, 1442, 1443, 1454, 1459, 1464, 1467, 1479, 1489, 1499, 1503, 1516, 1524, 1534, 1530, 1540, 1549, 1554, 1559, 1565, 1568, 1578, 1585, 1601, 1598, 1611, 1621, 1620, 1634, 1640, 1648, 1653, 1676, 1687, 1699, 1707, 1720, 1741, 1743, 1758, 1761, 1773, 1785, 1792, 1805, 1836, 1856, 1876, 1885, 1898, 1917, 1937, 1960, 1971, 1982, 1997, 2008, 2020, 2030, 2042, 2064, 2076, 2087, 2107, 2126, 2130, 2147, 2164, 2179, 2194, 2211, 2231, 2239, 2259, 2276, 2291, 2301, 2320, 2329, 2348, 2378, 2403, 2423, 2435, 2448, 2464, 2487, 2508, 2517, 2540, 2561, 2579, 2605, 2624, 2650, 2671, 2685, 2704, 2720, 2749, 2773, 2796, 2821, 2850, 2876, 2922, 2948, 2981, 2990, 3023, 3063, 3081, 3114, 3143, 3176, 3194, 3227, 3259, 3291, 3315, 3338, 3363, 3381, 3395, 3439, 3456, 3486, 3511, 3546, 3574, 3615, 3654, 3666, 3689, 3715, 3744, 3791, 3817, 3839, 3864, 3886, 3923, 3940, 3981, 4018, 4041, 4064, 4098, 4130, 4155, 4193, 4228, 4269, 4299, 4317, 4349, 4392, 4429, 4451, 4483, 4532, 4569, 4619, 4662, 4698, 4759, 4794, 4826, 4861, 4889, 4915, 4968, 5001, 5018, 5065, 5118, 5181, 5216, 5243, 5292, 5338, 5392, 5452, 5504, 5540, 5574, 5616, 5639, 5676, 5703, 5744, 5793, 5872, 5920, 5971, 6008, 6049, 6102, 6146, 6194, 6236, 6284, 6341, 6398, 6464, 6511, 6552, 6601, 6654, 6715, 6780, 6843, 6908, 6961, 7011, 7065, 7105, 7174, 7226, 7285, 7342, 7401, 7471, 7534, 7578, 7666, 7738, 7798, 7836, 7916, 7971, 8010, 8066, 8132, 8180, 8285, 8337, 8392, 8471, 8521, 8602, 8664, 8750, 8783, 8829, 8897, 8944, 9013, 9082, 9146, 9191, 9257, 9321, 9399, 9500, 9563, 9676, 9724, 9820, 9884, 9924, 9979, 10054, 10134, 10250, 10284, 10347, 10422, 10499, 10557, 10605, 10669, 10746, 10792, 10876, 10928, 10971, 11028, 11090, 11198, 11270, 11355, 11420, 11457, 11511, 11630, 11703, 11801, 11913, 11949, 12035, 12096, 12151, 12244, 12267, 12333, 12442, 12506, 12564, 12642, 12698, 12757, 12840, 12915, 13001, 13074, 13144, 13255, 13299, 13341, 13372, 13435, 13517, 13595, 13656, 13750, 13802, 13865, 13956, 14043, 14151, 14239, 14264, 14313, 14402, 14485, 14542, 14638, 14705, 14796, 14942, 15009, 15042, 15111, 15213, 15257, 15394, 15492, 15559, 15605, 15695, 15771, 15842, 15888, 15955, 16076, 16175, 16216, 16281, 16405, 16476, 16556, 16644, 16740, 16815, 16877, 16955, 17011, 17070, 17155, 17246, 17313, 17389, 17457, 17521, 17562, 17634, 17719, 17819, 17912, 17984, 18041, 18137, 18191, 18316, 18364, 18423, 18493, 18584, 18648, 18716, 18801, 18916, 19036, 19125, 19179, 19262, 19324, 19408, 19495, 19534, 19561, 19635, 19742, 19827, 19884, 19971, 20020, 20144, 20158, 20223, 20295, 20353, 20435, 20501, 20613, 20686, 20719, 20818, 20939, 21046, 21131, 21146, 21230, 21291, 21402, 21497, 21599, 21668, 21716, 21792, 21937, 22042, 22099, 22187, 22262, 22344, 22416, 22509, 22584, 22639, 22675, 22752, 22768, 22847, 22953, 23043, 23089, 23156, 23216, 23329, 23428, 23530, 23620, 23720, 23795, 23807, 23872, 23911, 23937, 24022, 24054, 24124, 24226, 24342, 24424, 24469, 24513, 24530, 24577, 24690, 24744, 24828, 24885, 25020, 25022, 25115, 25186, 25276, 25352, 25391, 25437, 25527, 25552, 25601, 25660, 25731, 25795, 25837, 25904, 25966, 26088, 26206, 26242, 26284, 26360, 26419, 26427, 26480, 26527, 26561, 26662, 26672, 26713, 26774, 26814, 26892, 26943, 27024, 27084, 27177, 27214, 27209, 27246, 27301, 27386, 27479, 27527, 27622, 27610, 27672, 27740, 27814, 27856, 27929, 28011, 28023, 28087, 28102, 28101, 28180, 28270, 28285, 28306, 28359, 28400, 28433, 28505, 28533, 28563, 28625, 28618, 28653, 28707, 28735, 28782, 28879, 28952, 28935, 29008, 29058, 29160, 29156, 29232, 29295, 29318, 29376, 29454, 29526, 29577, 29594, 29634, 29664, 29695, 29647, 29652, 29604, 29643, 29735, 29731, 29715, 29759, 29790]
    print ('len Y = ', len (Y))
    exit ()
    #lp_time_summary_file = open ("../res/lp_time_summary.res", "a") # Will write to this file an IBM CPlex' .mod file, describing the problem
    
    # Gen static LP problem
    t = time.time()
    my_simulator = SFC_mig_simulator (verbose = 0)
    my_simulator.simulate()
    # my_simulator.calc_sol_cost_SS ()
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

