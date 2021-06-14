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
from scipy.optimize import linprog
from cmath import sqrt

# Levels of verbose (which output is generated)
VERBOSE_NO_OUTPUT             = 0
VERBOSE_ONLY_RES              = 1 # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_RES_AND_LOG           = 2 # Write to a file the total cost and rsrc aug. upon every event + write a detailed ".log" file
VERBOSE_RES_AND_DETAILED_LOG  = 3 # Write to a file the total cost and rsrc aug. upon every event + write a detailed ".log" file

class SFC_mig_simulator (object):


    #############################################################################
    # Inline functions
    #############################################################################
    # Returns the parent of a given server
    parent_of = lambda self, s : self.G.nodes[s]['prnt']

    # # Find the server on which a given user is located, by its lvl
    loc_of_user = lambda self, usr : usr.S_u[usr.lvl]

    # calculate the total cost of a solution
    calc_sol_cost = lambda self: sum ([self.calc_chain_cost_homo (usr, usr.lvl) for usr in self.usrs])
   
    # calculate the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    calc_chain_cost_homo = lambda self, usr, lvl: self.link_cost_of_SSP_at_lvl[lvl] + (usr.S_u[usr.lvl] != usr.cur_s) * self.uniform_mig_cost * len (usr.theta_times_lambda) + self.CPU_cost_at_lvl[lvl] * usr.B[lvl]    
          
    # Calculate the (maximal) rsrc aug' used by the current solution, using a single (scalar) R
    calc_sol_rsrc_aug = lambda self, R : np.max ([(R * self.G.nodes[s]['cpu cap'] - self.G.nodes[s]['a']) / self.G.nodes[s]['cpu cap'] for s in self.G.nodes()])
         
    # Calculate the CPU used in practice in each server in the current solution
    used_cpu_in = lambda self, R : [(R * self.G.nodes[s]['cpu cap'] - self.G.nodes[s]['a']) for s in self.G.nodes()]

    # calculate the proved upper bnd on the rsrc aug that bottomUp may need to find a feasible sol, given such a sol exists for the non-augmented prob'
    calc_upr_bnd_rsrc_aug = lambda self: np.max ([usr.C_u for usr in self.usrs]) / np.min ([np.min (usr.B) for usr in self.usrs])

    # # Returns the AP covering a given (x,y) location, assuming that the cells are identical fixed-size squares
    loc2ap_sq = lambda self, x, y: int (math.floor ((y / self.cell_Y_edge) ) * self.num_of_APs_in_row + math.floor ((x / self.cell_X_edge) )) 

    # Returns the server to which a given user is currently assigned
    cur_server_of = lambda usr: usr.S_u[usr.lvl] 

    def solveByLp (self, changed_usrs):
        """
        Example of solving a problem using Python's LP capabilities.
        """
        self.decision_vars  = []
        id                  = 0
        for usr in changed_usrs:
            for lvl in range(len(usr.B)):
                self.decision_vars.append (decision_var_c (id=id, usr=usr, lvl=lvl, s=usr.S_u[lvl]))
                id += 1

        # Adding the CPU cap' constraints
        # A will hold the decision vars' coefficients. b will hold the bound: the constraints are: Ax<=b 
        A = np.zeros ([len (self.G.nodes) + len (self.usrs), len(self.decision_vars)], dtype = 'int16')
        for s in self.G.nodes():
            for decision_var in filter (lambda item : item.s == s, self.decision_vars):
                A[s][decision_var.id] = decision_var.usr.B[decision_var.lvl]

        for decision_var in self.decision_vars:
            A[len(self.G.nodes) + decision_var.usr.id][decision_var.id] = -1
        b_ub = - np.ones (len(self.G.nodes) + len(self.usrs), dtype='int16')  
        b_ub[self.G.nodes()] = [self.G.nodes[s]['cpu cap'] for s in range(len(self.G.nodes))]
        # print (A)
        # print (b_ub)
        # print (self.usrs)
        # exit ()
        res = linprog ([self.calc_chain_cost_homo (decision_var.usr, decision_var.lvl) for decision_var in self.decision_vars], 
                       A_ub   = A, 
                       b_ub   = b_ub, 
                       bounds = [[0.0, 1.0] for line in range (len(self.decision_vars))])
        
        if (self.verbose in [VERBOSE_ONLY_RES, VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            printf (self.res_output_file, 't{}.LP.stts{} {:.2f}' .format(self.t, res.status, res.fun))
            printf (self.log_output_file, 't{}.LP.stts{} {:.2f}' .format(self.t, res.status, res.fun))
            if (res.success): # successfully solved
                printf (self.log_output_file, 'cost by LP = {:.2f}\n' .format(res.fun))
                if (self.verbose == VERBOSE_RES_AND_DETAILED_LOG):
                    for i in [i for i in range(len(res.x)) if res.x[i]>0]:
                        printf (self.log_output_file, 'u {} lvl {:.0f} loc {:.0f} val {:.2f}' .format(
                               self.decision_vars[i].usr.id,self.decision_vars[i].lvl,self.decision_vars[i].s,res.x[i]))
                        printf (self.log_output_file, '\ntime = {}\n**************************************\n' .format (splitted_line[2]))                    
        exit ()

    def reset_sol (self):
        """"
        reset the solution - namely, the placement of each user to a concrete level in the tree, and to a concrete server
        - usr.lvl
        """
        for usr in self.usrs:
            usr.lvl   = -1
            usr.nxt_s = -1

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
        self.res_file_name = "../res/" + self.usrs_ap_file_name.split(".")[0] + ".res"  
        self.res_output_file = self.init_output_file(self.res_file_name)

    def init_log_file (self):
        """
        Open the log file for writing and write initial comments lines on it
        """
        self.log_file_name = "../res/" + self.usrs_ap_file_name.split(".")[0] + ".log"  
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
                    sum ([usr.B[usr.lvl] for usr in self.usrs if usr.nxt_s==s] ),
                    self.G.nodes[s]['cpu cap'],
                    [usr.id for usr in self.usrs if usr.nxt_s==s]))

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
                    
                    # update usr.lvl accordingly and usr.nxt_s
                    usr.lvl      = lvl               
                    usr.nxt_s    = usr.S_u[usr.lvl]    
                    
                    # update the moved usr's location in the heap
                    self.usrs[n] = self.usrs[-1] # replace the usr to push-up with the last usr in the heap
                    self.usrs.pop()
                    heapq.heappush(self.usrs, usr)
                    n = -1 # succeeded to push-up a user, so next time should start from the max (which may now succeed to move)
                    break
            n += 1
                            
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
            dst_server = usr2mov.S_u[lvl_star] 
            self.G.nodes [dst_server]   ['a'] -= usr2mov .B[lvl_star]           # dec the available CPU at the new loc of the moved usr
            print ('id of usr2mov = {}, old lvl = {}, lvl_star = {}, max_reduction = {}' .format(usr2mov.id, usr2mov.lvl, lvl_star, max_reduction))
            usr2mov.lvl   = lvl_star
            usr2mov.nxt_s = dst_server  
   
    def CPUAll_single_usr (self, usr): 
        """
        CPUAll algorithm:
        calculate the minimal CPU allocation required by a given usr, when the highest server on which u is located is s.
        The current implementation assumes that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.
        This version of the alg' assumes a balanced homogeneous tree, 
        so that the netw' delay between every two servers is unequivocally defined by their levels.
        """
        slack = [usr.target_delay - self.link_delay_of_SSP_at_lvl[lvl] for lvl in range (self.tree_height+1)]
        slack = [slack[lvl] for lvl in range(self.tree_height+1) if slack[lvl] > 0] # trunc all servers with negative slack, which are surely delay-INfeasible
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
                    
    def CPUAll (self, usrs): 
        """
        CPUAll algorithm:
        calculate the minimal CPU allocation required for each chain u, when the highest server on which u is located is s.
        The current implementation assumes that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.
        This version of the alg' assumes a balanced homogeneous tree, 
        so that the netw' delay between every two servers is unequivocally defined by their levels.
        """
        for usr in usrs:
            self.CPUAll_single_usr(usr)
            
    def fix_usr_params (self, usrs):
        """
        For each of the given users, fix the following parameters:
        target_delay, mig_cost, C_u
        """
            
    def rd_usr_data (self):
        """
        Currently unused.
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
                            
    # def loc2ap (self):
    #     """
    #     Read the input about the users locations, 
    #     and write the appropriate user-to-PoA connections to the file self.ap_file
    #     Assume that each AP covers a square area
    #     """
    #     self.ap_file  = open ("../res/" + self.usrs_loc_file_name.split(".")[0] + ".ap", "w+")  
    #     usrs_loc_file = open ("../res/" + self.usrs_loc_file_name,  "r") 
    #     printf (self.ap_file, '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of user X at time t\n')
    #
    #     self.max_X, self.max_Y = 1000, 1000 # size of the square cell of each AP, in meters. 
    #     self.num_of_APs_in_row = int (math.sqrt (self.num_of_leaves)) #$$$ cast to int, floor  
    #     self.cell_X_edge = self.max_X / self.num_of_APs_in_row
    #     self.cell_Y_edge = self.cell_X_edge
    #
    #     cur_ap_of_usr = [] # will hold pairs of (usr_id, cur_ap). 
    #     for line in usrs_loc_file: 
    #
    #         # remove the new-line character at the end (if any), and ignore comments lines 
    #         line = line.split ('\n')[0] 
    #         if (line.split ("//")[0] == ""):
    #             continue
    #
    #         splitted_line = line.split (" ")
    #
    #         # print (splitted_line[0])
    #         if (splitted_line[0] == "t" or splitted_line[0] == 'usrs_that_left:'):
    #             printf(self.ap_file, '\n{}' .format (line))
    #             continue
    #
    #         elif (splitted_line[0] == 'new_or_moved:'): # new vehicle
    #             printf(self.ap_file, '\nnew_or_moved: ')
    #
    #         else: # now we know that this line details a user that either joined, or moved.
    #             print (splitted_line)
    #             type   = splitted_line[0] # type will be either 'n', or 'o' (new, old user, resp.).
    #             usr_id = splitted_line[1]
    #             nxt_ap = self.loc2ap_sq (float(splitted_line[2]), float(splitted_line[3]))
    #             if (type == 'n'): # new vehicle
    #                 printf(self.ap_file, "({},{},{})," .format (type,usr_id, nxt_ap))                
    #                 cur_ap_of_usr.append({'id' : usr_id, 'ap' : nxt_ap})
    #             else: # old vehicle
    #                 list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, cur_ap_of_usr))
    #                 if (len (list_of_usr)== 0):
    #                     print ('Inaal raback')
    #                     exit ()
    #                 if (list_of_usr[0]['ap'] == nxt_ap): # The user is moving within area covered by the cur AP
    #                     continue
    #                 printf(self.ap_file, "({},{},{})" .format (type,usr_id, nxt_ap))                
    #                 list_of_usr[0]['ap'] = nxt_ap       
    #             continue
    #
    #     printf(self.ap_file, "\n")   

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
                    self.G.nodes[shortest_path[s][root][lvl]]['cpu cost']  = self.CPU_cost_at_lvl[lvl]                
                    self.G.nodes[shortest_path[s][root][lvl]]['link cost'] = self.link_cost_of_SSP_at_lvl[lvl]
                    self.G.nodes[shortest_path[s][root][lvl]]['cpu cap']   = self.CPU_cap_at_lvl[lvl]                
                    self.G.nodes[shortest_path[s][root][lvl]]['a']         = self.CPU_cap_at_lvl[lvl] # initially, there is no rsrc augmentation, and the available capacity of each server is exactly its CPU capacity.                
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
        self.tree_height                = 2
        self.children_per_node          = 2 # num of children of every non-leaf node
        self.uniform_mig_cost           = 1
        self.Lmax                       = 0
        self.uniform_Tpd                = 2
        self.uniform_link_cost          = 1
        self.uniform_theta_times_lambda = [1,1]
        self.uniform_Cu                 = 15
        self.uniform_target_delay       = 10
        
        # INPUT / OUTPUT FILES
        self.verbose = VERBOSE_RES_AND_LOG
        
        # Names of input files for the users' data, locations and / or current access points
        self.usrs_data_file_name  = "res.usr" #input file containing the target_delays and traffic of all users
        self.usrs_loc_file_name   = "short.loc"  #input file containing the locations of all users along the simulation
        self.usrs_ap_file_name    = 'short.ap' #input file containing the APs of all users along the simulation
        
        # Names of output files
        if (self.verbose in [VERBOSE_ONLY_RES, VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            self.init_res_file() 
        if (self.verbose in [VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            self.init_log_file() 
             
        self.gen_parameterized_tree ()

        # Flags indicators for writing output to results and log files
        self.write_to_cfg_file = False
        
        # Flags indicators for writing output to various LP solvers
        self.write_to_prb_file = False # When true, will write outputs to a .prb file. - ".prb" - A .prb file may solve an LP problem using the online Eq. solver: https://online-optimizer.appspot.com/?model=builtin:default.mod
        self.write_to_mod_file = False # When true, will write to a .mod file, fitting to IBM CPlex solver       
        self.write_to_lp_file  = True  # When true, will write to a .lp file, which allows running Cplex using a Python's api.       

    def simulate (self):

        # reset Hs        
        for s in self.G.nodes():
            self.G.nodes[s]['Hs'] = [] 
            self.G.nodes[s]['cur RCs'] = self.G.nodes[s]['cpu cap'] # Initially, no rsrc aug --> at each server, we've exactly his non-augmented capacity. 

        # Open input and output files
        self.ap_file  = open ("../res/" + self.usrs_ap_file_name, "r")  
        if (self.verbose == VERBOSE_RES_AND_LOG):
            self.init_log_file()
            
        self.usrs = []

        for line in self.ap_file: 

            # Ignore comments lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue

            line = line.split ('\n')[0]
            splitted_line = line.split (" ")

            if (splitted_line[0] == "t"):
                self.t = splitted_line[2]
                if (self.verbose in [VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
                    printf (self.log_output_file, '\ntime = {}\n**************************************\n' .format (self.t))
                continue
        
            elif (splitted_line[0] == "users_that_left:"):
                for usr in list (filter (lambda usr : usr.id in splitted_line[1:], self.usrs)):                     
                    self.G.nodes[usr.C_u[usr.lvl]]['a'] += usr.B[usr.lvl] # free the resources held by that usr
                continue
        
            elif (splitted_line[0] == "new_or_moved:"):
                self.rd_AP_line(splitted_line[1:])
                self.alg_top()
                for usr in self.usrs: # prepare for the next iteration
                     usr.cur_s = usr.nxt_s
                self.solveByLp (self.usrs)
                continue
        
    def binary_search (self):
        """
        Binary search for a feasible sol that minimizes the resource augmentation R.
        The search is done by calling bottom_up ().
        """
        self.reset_sol() # dis-allocate all users
        self.CPUAll(self.usrs) 
        max_R = self.calc_upr_bnd_rsrc_aug () 
        
        # init cur RCs and a(s) to the number of available CPU in each server, assuming maximal rsrc aug' 
        for s in self.G.nodes():
            self.G.nodes[s]['cur RCs'] = math.ceil (max_R * self.G.nodes[s]['cpu cap']) 
            self.G.nodes[s]['a']       = self.G.nodes[s]['cur RCs'] #currently-available rsrcs at server s  

        if (not (self.bottom_up ())):
            print ('did not find a feasible sol even with maximal rsrc aug')
            exit ()
        
        # Now we know that we found an initial feasible sol 
        if (self.verbose == VERBOSE_RES_AND_LOG):
            printf (self.log_output_file, 'Initial solution:\n')
            self.print_sol()
                   
        ub = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()]) # upper-bnd on the (augmented) cpu cap' that may be required
        lb = np.array([self.G.nodes[s]['cpu cap'] for s in self.G.nodes()]) # lower-bnd on the (augmented) cpu cap' that may be required
        while True: 
            
            # Update the available capacity at each server according to the value of resource augmentation for this iteration            
            for s in self.G.nodes():
                self.G.nodes[s]['a'] = math.ceil (0.5*(ub[s] + lb[s])) 
            if (np.array([self.G.nodes[s]['a'] for s in self.G.nodes()]) == np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()])).all():
                # The binary search converged - update 'a' to the real available value at each server, considering the allocation to users, and return
                self.update_available_cpu_by_sol()
                return 
            
            # Update the total capacity at each server according to the value of resource augmentation for this iteration            
            for s in self.G.nodes():
                self.G.nodes[s]['cur RCs'] = self.G.nodes[s]['a'] 

            # Solve using bottom-up
            if (self.bottom_up()):
                ub = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()])        
            else:
                lb = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()])
                
    
    
    def alg_top (self):
        """
        Top-level alg'
        """
        
        # Try to solve the problem by changing the placement or CPU allocation only for the new / moved users
        if (not(self.bottom_up())):
            self.binary_search()

        # By hook or by crook, now we have a feasible solution        
        if (self.verbose == VERBOSE_RES_AND_LOG):
            printf (self.log_output_file, 'B4 push-up:\n')
            self.print_sol()
            
        self.update_available_cpu_by_sol () # update the available capacity a at each server $$$ - isn't this already done by bottom-up?
        self.push_up ()
        if (self.verbose == VERBOSE_RES_AND_LOG):
            printf (self.log_output_file, 'After push-up:\n')
            self.print_sol()

    def update_available_cpu_by_sol (self):
        """
        Update the available capacity at each server to: 
        the (possibly augmented) CPU capacity at this server - the total CPU assigned to users by the solution 
        """
        for s in self.G.nodes():
            self.G.nodes[s]['a'] = self.G.nodes[s]['cur RCs'] - sum ([usr.B[usr.lvl] for usr in self.usrs if usr.nxt_s == s])                

    
    def bottom_up (self):
        """
        Bottom-up alg'. 
        Looks for a feasible sol'.
        Returns true iff a feasible sol was found
        """
        
        for s in range (len (self.G.nodes())-1, -1, -1): # for each server s, in an increasing order of levels (DFS).
            lvl = self.G.nodes[s]['lvl']
            Hs = [usr for usr in self.G.nodes[s]['Hs']  if (usr.lvl == -1)]
            for usr in sorted (Hs, key = lambda usr : len(usr.B)): # for each chain in Hs, in an increasing order of level ('L')
                #print ('s = ', s, 'a = ', self.G.nodes[s]['a'], 'B = ', usr.B[lvl])                   
                if (self.G.nodes[s]['a'] > usr.B[lvl]): 
                    usr.nxt_s = s
                    usr.lvl = lvl
                    self.G.nodes[s]['a'] -= usr.B[lvl]
                elif (len (usr.B) == lvl):
                    return False
        return True
   
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
            if (len(tuple) <= 1):
                break
            tuple = tuple.split("(")
            # print ('b4 split', tuple)
            tuple   = tuple[1].split (',')

            if (tuple[0] == 'n'): # new user
                usr = usr_c (id = int(tuple[1]))
                self.CPUAll_single_usr (usr)
                self.usrs.append (usr)
            else: # old, existing user, who moved
                list_of_usr = list(filter (lambda usr : usr.id == int(tuple[1]), self.usrs))
                usr = list_of_usr[0]
                usr_cur_cpu = usr.B[usr.lvl]
                self.CPUAll_single_usr (usr)
                if (usr.lvl >= len (usr.B) and usr.B[usr.lvl] <= usr_cur_cpu): # can satisfy delay constraint while leaving the chain in its cur location and CPU budget 
                        continue
                
                # Now we know that this is a critical usr, namely a user that needs more CPU and/or migration for satisfying its target delay constraint 
                self.G.nodes[cur_server_of(usr)]['a'] += usr.B[usr.lvl] # free the CPU units used by the user in the old location

            AP_id = int(tuple[2])
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
        Calculate the total cost of an SS (single-server per-chain) full solution.
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
    #lp_time_summary_file = open ("../res/lp_time_summary.res", "a") # Will write to this file an IBM CPlex' .mod file, describing the problem
    
    # Gen static LP problem
    t = time.time()
    my_simulator = SFC_mig_simulator (verbose = 1)
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

