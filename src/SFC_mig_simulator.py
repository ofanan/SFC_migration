# Bugs: 
# improve syntax of .loc, and change loc2ap_c.py accordingly.
# change plp. Possibly use for it a different "usr_c" type.
# Check the inter-time-slots issue for alg top. Is the calc of mig' cost correct?

import networkx as nx
import numpy as np
import math
import itertools 
import time
import random
import heapq
import pulp as plp

from usr_c import usr_c # class of the users
from decision_var_c import decision_var_c # class of the decision variables
from printf import printf
# from scipy.optimize import linprog
from cmath import sqrt

# Levels of verbose (which output is generated)
VERBOSE_NO_OUTPUT             = 0
VERBOSE_ONLY_RES              = 1 # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_RES_AND_LOG           = 2 # Write to a file the total cost and rsrc aug. upon every event + write a detailed ".log" file
VERBOSE_RES_AND_DETAILED_LOG  = 3 # Write to a file the total cost and rsrc aug. upon every event + write a detailed ".log" file

class SFC_mig_simulator (object):
    """
    Run a simulation of the Service Function Chains migration problem.
    """
    #############################################################################
    # Inline functions
    #############################################################################
    # Returns the parent of a given server
    parent_of = lambda self, s : self.G.nodes[s]['prnt']

    # calculate the total cost of a solution
    calc_sol_cost = lambda self: sum ([self.calc_chain_cost_homo (usr, usr.lvl) for usr in self.usrs])
   
    # calculate the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    calc_chain_cost_homo = lambda self, usr, lvl: self.link_cost_of_SSP_at_lvl[lvl] + self.CPU_cost_at_lvl[lvl] * usr.B[lvl] + self.calc_mig_cost (usr, lvl)    
    
    # # calculate the migration cost incurred for a usr if located on a given lvl
    calc_mig_cost = lambda self, usr, lvl : (usr.S_u[usr.lvl] != usr.cur_s and usr.cur_s!=-1) * self.uniform_mig_cost * len (usr.theta_times_lambda)
          
    # Calculate the number of CPU units actually used in each server
    used_cpu_in = lambda self: np.array ([self.G.nodes[s]['cur RCs'] - self.G.nodes[s]['a'] for s in self.G.nodes])      
          
    # calculate the proved upper bnd on the rsrc aug that bottomUp may need to find a feasible sol, given such a sol exists for the non-augmented prob'
    calc_upr_bnd_rsrc_aug = lambda self: np.max ([usr.C_u for usr in self.usrs]) / np.min ([np.min (usr.B) for usr in self.usrs])

    # # Returns the AP covering a given (x,y) location, assuming that the cells are identical fixed-size squares
    loc2ap_sq = lambda self, x, y: int (math.floor ((y / self.cell_Y_edge) ) * self.num_of_APs_in_row + math.floor ((x / self.cell_X_edge) )) 

    # Returns the server to which a given user is currently assigned
    cur_server_of = lambda self, usr: usr.S_u[usr.lvl] 
   
    def calc_rsrc_aug (self):
        """
        Calculate the (maximal) rsrc aug' used by the current solution, using a single (scalar) R
        """
        used_cpu_in = self.used_cpu_in ()
        return max (np.max ([(used_cpu_in[s] / self.G.nodes[s]['cpu cap']) for s in self.G.nodes()]), 1)    
         
    def reset_sol (self):
        """
        Reset the solution, namely, Dis-place all users. This is done by: 
        1. Resetting the placement of each user to a concrete level in the tree, and to a concrete server.
        2. Init the available cpu at each server to its (possibly augmented) cpu capacity. 
        """
        for usr in self.usrs:
            usr.lvl   = -1
            usr.nxt_s = -1
        for s in self.G.nodes():
            self.G.nodes[s]['a'] = self.G.nodes[s]['cur RCs']
        # print ('in rst sol: RCS={}, a={}' .format (self.G.nodes[0]['cur RCs'], self.G.nodes[0]['a']))

    # def solve_by_scipy_linprog (self):
    #     """
    #     Currently unused, because we switched to using Python's pulp linear solver, which is much faster.
    #     Find a fractional optimal solution using Python's scipy.optimize.linprog library.
    #     """
    #     self.decision_vars  = []
    #     id                  = 0
    #     for usr in self.usrs:
    #         for lvl in range(len(usr.B)):
    #             self.decision_vars.append (decision_var_c (id=id, usr=usr, lvl=lvl, s=usr.S_u[lvl]))
    #             id += 1
    #
    #     # Adding the CPU cap' constraints
    #     # A will hold the decision vars' coefficients. b will hold the bound: the constraints are: Ax<=b 
    #     A = np.zeros ([len (self.G.nodes) + len (self.usrs), len(self.decision_vars)], dtype = 'int16')
    #     for s in self.G.nodes():
    #         for decision_var in filter (lambda item : item.s == s, self.decision_vars):
    #             A[s][decision_var.id] = decision_var.usr.B[decision_var.lvl]
    #
    #     for decision_var in self.decision_vars:
    #         A[len(self.G.nodes) + decision_var.usr.id][decision_var.id] = -1
    #     b_ub = - np.ones (len(self.G.nodes) + len(self.usrs), dtype='int16')  
    #     b_ub[self.G.nodes()] = [self.G.nodes[s]['cpu cap'] for s in range(len(self.G.nodes))]
    #     res = linprog ([self.calc_chain_cost_homo (decision_var.usr, decision_var.lvl) for decision_var in self.decision_vars], 
    #                    A_ub   = A, 
    #                    b_ub   = b_ub, 
    #                    bounds = [[0.0, 1.0] for line in range (len(self.decision_vars))])
    #
    #     if (self.verbose in [VERBOSE_ONLY_RES, VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
    #         printf (self.res_output_file, 't{}.LP.stts{} {:.2f}\n' .format(self.t, res.status, res.fun))
    #         printf (self.log_output_file, 't{}.LP.stts{} {:.2f}\n' .format(self.t, res.status, res.fun))
    #         if (res.success == True): # successfully solved
    #             if (self.verbose == VERBOSE_RES_AND_DETAILED_LOG):
    #                 for i in [i for i in range(len(res.x)) if res.x[i]>0]:
    #                     printf (self.log_output_file, '\nu {} lvl {:.0f} loc {:.0f} val {:.2f}' .format(
    #                            self.decision_vars[i].usr.id,self.decision_vars[i].lvl,self.decision_vars[i].s,res.x[i]))
    #         else: 
    #             printf (self.log_output_file, '// status codes: 1: Iteration limit reached. 2. Infeasible. 3. Unbounded. 4. Numerical difficulties.\n')

    # def calc_decision_var_cost (self):
    #     """
    #     Not complete yet.
    #     Caluclate the cost of setting a decision var, when using the linear prog'
    #     """
    #     # consider the BW and CPU cost
    #     cost = self.link_cost_of_SSP_at_lvl[decision_var.lvl] + self.CPU_cost_at_lvl[decision_var.lvl] * decision_var.usr.B[decision_var.lvl] 
    #
    #     # Add the mig cost
    #     list_of_cur_st_param = list(filter (lambda cur_st : cur_st.usr==decision_var.usr and cur_st.s == decision_var.s, self.cur_state))
    #     cost += self.uniform_mig_cost * len (usr.theta_times_lambda) * list_of_cur_st_param[0].value 

    def solve_by_plp (self):
        """
        Find an optimal fractional solution using Python's pulp LP library.
        pulp library can use commercial tools (e.g., Gurobi, Cplex) to efficiently solve the prob'.
        """
        printf (self.log_output_file, 'Starting LP\n')
        model = plp.LpProblem(name="SFC_mig", sense=plp.LpMinimize)
        self.d_vars  = [] # decision variables  
        obj_func     = [] # objective function
        id           = 0  # cntr for the id of the decision variables 
        for usr in self.usrs:
            single_place_const = [] # will hold constraint assuring that each chain is placed in a single server
            for lvl in range(len(usr.B)): # will check all delay-feasible servers for this user
                plp_lp_var = plp.LpVariable (lowBound=0, upBound=1, name='x_{}' .format (id))
                decision_var = decision_var_c (id=id, usr=usr, lvl=lvl, s=usr.S_u[lvl], plp_lp_var=plp_lp_var) # generate a decision var, containing the lp var + details about its meaning 
                self.d_vars.append (decision_var)
                single_place_const += plp_lp_var
                obj_func           += self.calc_chain_cost_homo (decision_var) * plp_lp_var # add the cost of this decision var to the objective func 
                id += 1
            model += (single_place_const == 1) # demand that each chain is placed in a single server
        model += obj_func

        # Generate CPU capacity constraints
        for s in self.G.nodes():
            cpu_cap_const = []
            for d_var in list (filter (lambda item : item.s == s, self.d_vars)): # for every decision variable meaning placing a chain on this server 
                cpu_cap_const += (d_var.usr.B[d_var.lvl] * d_var.plp_lp_var) # Add the overall cpu of this chain, if located on s
            if (cpu_cap_const != []):
                model += (cpu_cap_const <= self.G.nodes[s]['cpu cap']) 

        # solve using another solver: solve(GLPK(msg = 0))
        model.solve(plp.PULP_CBC_CMD(msg=0)) # solve the model, without printing output
        
        if (self.verbose in [VERBOSE_ONLY_RES, VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            printf (self.res_output_file, 't{}.plp.stts{} cost={:.2f}\n' .format(self.t, model.status, model.objective.value())) 
            if (self.verbose in [VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
                printf (self.log_output_file, 't{}.plp.stts{} cost={:.2f}\n' .format(self.t, model.status, model.objective.value())) 
                                                                            #plp.LpStatus[model.status]))
        if (self.verbose == VERBOSE_RES_AND_DETAILED_LOG): # successfully solved
            if (model.status == 1): 
                for d_var in self.d_vars: 
                    if d_var.plp_lp_var.value() > 0:
                        printf (self.log_output_file, '\nu {} lvl {:.0f} s {:.0f} val {:.2f}' .format(
                               d_var.usr.id, d_var.lvl, d_var.s, d_var.plp_lp_var.value()))
            else:
                printf (self.log_output_file, 'failed. status={}\n' .format(plp.LpStatus[model.status]))

        # Make the solution the "current state", for the next time slot  
        for d_var in self.d_vars: 
            printf (self.log_output_file, '\nu {} lvl {:.0f} s {:.0f} val {:.2f}' .format(
                   d_var.usr.id, d_var.lvl, d_var.s, d_var.plp_lp_var.value()))
        

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
        Open an output file for writing, overwriting previous content in that file. 
        Input: output file name.
        Output: file descriptor.  
        """
        with open('../res/' + file_name, 'w') as FD:
            FD.write('')                
        FD  = open ('../res/' + file_name,  "w")
        return FD

    def init_res_file (self):
        """
        Open the res file for writing.
        """
        self.res_file_name = "../res/" + self.ap_file_name.split(".")[0] + ".res"  
        self.res_output_file = self.init_output_file(self.res_file_name)

    def init_log_file (self):
        """
        Open the log file for writing and write initial comments lines on it
        """
        self.log_file_name = "../res/" + self.ap_file_name.split(".")[0] + '.' + self.alg.split("_")[1] + '.log'  
        self.log_output_file = self.init_output_file(self.log_file_name)
        printf (self.log_output_file, '// format: s : used / C_s   chains[u1, u2, ...]\n')
        printf (self.log_output_file, '// where: s = number of server. used = capacity used by the sol on server s.\n//C_s = non-augmented capacity of s. u1, u2, ... = chains placed on s.\n' )

    def print_sol_to_res (self):
        """
        print a solution for the problem to the output log file 
        """
        used_cpu_in = self.used_cpu_in ()
        printf (self.res_output_file, 't{}.alg cost={:.2f} rsrc_aug={:.2f}\n' .format(
            self.t, 
            self.calc_sol_cost(),
            self.calc_rsrc_aug())) 

    def print_sol_to_log (self):
        """
        print a detailed solution for to the output log file 
        """
        used_cpu_in = self.used_cpu_in ()
        printf (self.log_output_file, 't{}.alg cost={:.2f} rsrc_aug={:.2f}\n' .format(
            self.t, 
            self.calc_sol_cost(),
            self.calc_rsrc_aug())) 
        
        for s in self.G.nodes():
            printf (self.log_output_file, 's{} : Rcs={}, a={}, used cpu={:.0f}, Cs={}\t chains {}\n' .format (
                    s,
                    self.G.nodes[s]['cur RCs'],
                    self.G.nodes[s]['a'],
                    sum ([usr.B[usr.lvl] for usr in self.usrs if usr.nxt_s==s] ),
                    #used_cpu_in[s],
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
        Push-up chains: take a feasible solution, and greedily try pushing each chain as high as possible in the tree. 
        Do that when chains are sorted in a decreasing order of the # of CPU units they're currently using.
        """
        # Assume here that the available cap' at each server 'a' is already calculated by the alg' that was run 
        # before calling push-up ()
        heapq._heapify_max(self.usrs)
 
        n = 0  # num of failing push-up tries in sequence; when this number reaches the number of users, return

        while n < len (self.usrs):
            usr = self.usrs[n]
            for lvl in range (len(usr.B)-1, usr.lvl, -1): #
                if (self.G.nodes[usr.S_u[lvl]]['a'] >= usr.B[lvl]): # if there's enough available space to move u to level lvl
                    self.G.nodes [usr.S_u[usr.lvl]] ['a'] += usr.B[usr.lvl] # inc the available CPU at the prev loc of the moved usr  
                    self.G.nodes [usr.S_u[lvl]]     ['a'] -= usr.B[lvl]     # dec the available CPU at the new  loc of the moved usr
                    
                    # update usr.lvl and usr.nxt_s accordingly 
                    usr.lvl      = lvl               
                    usr.nxt_s    = usr.S_u[usr.lvl]    
                    
                    # update the moved usr's location in the heap
                    self.usrs[n] = self.usrs[-1] # replace the usr to push-up with the last usr in the heap
                    self.usrs.pop() # pop the last user from the heap
                    heapq.heappush(self.usrs, usr) # push back to the heap the user we have just pushed-up
                    n = 0 # succeeded to push-up a user, so next time should start from the max (which may now succeed to move)
                    break
            else:
                n += 1
                            
    def reduce_cost (self):
        """
        Currently unused
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
            
            # move usr2mov from its current lvl to lvl_star, and update Y, a, accordingly
            self.G.nodes [usr2mov.S_u[usr2mov.lvl]] ['a'] += usr2mov.B[usr2mov.lvl] # inc the available CPU at the prev loc of the moved usr
            dst_server = usr2mov.S_u[lvl_star] 
            self.G.nodes [dst_server]   ['a'] -= usr2mov .B[lvl_star]           # dec the available CPU at the new loc of the moved usr
            print ('id of usr2mov = {}, old lvl = {}, lvl_star = {}, max_reduction = {}' .format(usr2mov.id, usr2mov.lvl, lvl_star, max_reduction))
            usr2mov.lvl   = lvl_star
            usr2mov.nxt_s = dst_server  
   
    def CPUAll_single_usr (self, usr): 
        """
        CPUAll algorithm, for a single usr:
        calculate the minimal CPU allocation required by the given usr, when the highest server on which u is located is s.
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
        Currently unused.
        For each of the given users, fix the following parameters:
        target_delay, mig_cost, C_u
        """
            
    def gen_parameterized_tree (self):
        """
        Generate a parameterized tree with specified height and children-per-non-leaf-node. 
        """
        self.G                 = nx.generators.classic.balanced_tree (r=self.children_per_node, h=self.tree_height) # Generate a tree of height h where each node has r children.
        self.cpu_cap_at_lvl    = np.array ([1000 * (lvl+1) for lvl in range (self.tree_height+1)], dtype='uint16')
        self.CPU_cost_at_lvl   = [1 * (self.tree_height + 1 - lvl) for lvl in range (self.tree_height+1)]
        self.link_cost_at_lvl  = np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of using a link from level i to level i+1, or vice versa.
        self.link_delay_at_lvl = np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of using a link from level i to level i+1, or vice versa.
        
        # overall link cost and link capacity of a Single-Server Placement of a chain at each lvl
        self.link_cost_of_SSP_at_lvl  = [2 * sum([self.link_cost_at_lvl[i]  for i in range (lvl)]) for lvl in range (self.tree_height+1)]
        self.link_delay_of_SSP_at_lvl = [2 * sum([self.link_delay_at_lvl[i] for i in range (lvl)]) for lvl in range (self.tree_height+1)] 
        
        self.G = self.G.to_directed()

        shortest_path = nx.shortest_path(self.G)

        # levelize the tree (assuming a balanced tree)
        self.ap2s             = []  
        root                  = 0 # In networkx, the ID of the root is 0
        self.num_of_leaves    = 0
        for s in self.G.nodes(): # for every server
            self.G.nodes[s]['id'] = s
            if self.G.out_degree(s)==1 and self.G.in_degree(s)==1: # is it a leaf?
                self.G.nodes[s]['lvl']   = 0 # Yep --> its lvl is 0
                self.ap2s.append (s) #[self.num_of_leaves] = s
                self.num_of_leaves += 1
                for lvl in range (self.tree_height+1):
                    self.G.nodes[shortest_path[s][root][lvl]]['lvl']       = np.uint8(lvl) # assume here a balanced tree
                    self.G.nodes[shortest_path[s][root][lvl]]['cpu cap']   = self.cpu_cap_at_lvl[lvl]                
                    self.G.nodes[shortest_path[s][root][lvl]]['a']         = self.cpu_cap_at_lvl[lvl] # initially, there is no rsrc augmentation, and the available capacity of each server is exactly its CPU capacity.
                    # # The lines below are for case one likes to vary the link and cpu costs of distinct servers on the same level. 
                    # self.G.nodes[shortest_path[s][root][lvl]]['cpu cost']  = self.CPU_cost_at_lvl[lvl]                
                    # self.G.nodes[shortest_path[s][root][lvl]]['link cost'] = self.link_cost_of_SSP_at_lvl[lvl]
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

    def __init__ (self, ap_file_name = 'shorter.ap', verbose = -1, tree_height = 3, children_per_node = 4):
        """
        """
               
        self.verbose                    = verbose
        # Network parameters
        self.tree_height                = tree_height
        self.children_per_node          = children_per_node # num of children of every non-leaf node
        self.uniform_mig_cost           = 1
        self.Lmax                       = 0
        self.uniform_Tpd                = 2
        self.uniform_link_cost          = 1
        self.uniform_theta_times_lambda = [1, 1, 1]
        self.uniform_Cu                 = 15 
        self.uniform_target_delay       = 20
        self.warned_about_too_large_ap  = False
        self.ap_file_name               = ap_file_name #input file containing the APs of all users along the simulation
        
        # Names of output files
        if (self.verbose in [VERBOSE_ONLY_RES, VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            self.init_res_file() 
        self.gen_parameterized_tree ()

    def simulate (self, alg):
        """
        Simulate the whole simulation:
        At each time step:
        - Read and parse from an input ".ap" file the AP cells of each user who moved. 
        - solve the problem (the "nxt_st), using the chosen alg: LP, or ALG_TOP (our alg).
        - Write outputs results and/or logs to files.
        - update cur_st = nxt_st
        """

        self.alg = alg
        if (self.verbose in [VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            self.init_log_file()
             
        print ('Simulating. num of leaves = {}. ap file = {}' .format (self.num_of_leaves, self.ap_file_name))
        # reset Hs        
        for s in self.G.nodes():
            self.G.nodes[s]['Hs'] = [] 
            self.G.nodes[s]['cur RCs'] = self.G.nodes[s]['cpu cap'] # Initially, no rsrc aug --> at each server, we've exactly his non-augmented capacity. 

        # Open input and output files
        self.ap_file  = open ("../res/" + self.ap_file_name, "r")  
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
                self.t = int(splitted_line[2])
                if (self.verbose in [VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
                    printf (self.log_output_file, '\ntime = {}\n**************************************\n' .format (self.t))
                continue
        
            elif (splitted_line[0] == "usrs_that_left:"):

                for usr in list (filter (lambda usr : usr.id in [int(usr_id) for usr_id in splitted_line[1:] if usr_id!=''], self.usrs)):

                    self.G.nodes[usr.cur_s]['a'] += usr.B[usr.lvl] # free the resources held by that usr       
                    
                    # Remove the usr from the list of "descended users" on any server s 
                    del (usr)
                continue
        
            elif (splitted_line[0] == "new_usrs:"):              
                self.rd_new_usrs_line (splitted_line[1:])
            elif (splitted_line[0] == "old_usrs:"):              
                self.rd_old_usrs_line (splitted_line[1:])
                
                self.solve_mig_prob ()
                continue

    def solve_mig_prob (self):
        """
        Solve the mig' problem for this time slot, using the self.alg algorithm (e.g., lp, or our algorithm).
        """
        
        if (self.verbose == VERBOSE_RES_AND_DETAILED_LOG):
            self.last_rt = time.time()
        if (self.alg == 'alg_lp'): 
            self.solve_by_plp () 
        elif (self.alg ==  'alg_top'):
            self.alg_top ()
        else: 
            print ('sorry, but the requested algorithm {} is not supported' .format (self.alg))
            exit ()

        if (self.verbose in [VERBOSE_ONLY_RES, VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            self.print_sol_to_res()
            if (self.verbose == VERBOSE_RES_AND_DETAILED_LOG):
                printf (self.log_output_file, '\nSolved in {:.3f} [sec]\n' .format (time.time() - self.last_rt))
        for usr in self.usrs: # The solution found at this time slot is the "cur_state" for next slot
             usr.cur_s = usr.nxt_s        
        for s in self.G.nodes(): # Clean the list of 'unallocated usrs, for them server s is delay-feasible" (Hs) on each server 
            self.G.nodes[s]['Hs'] = []
        
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
                   
        ub = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()]) # upper-bnd on the (augmented) cpu cap' that may be required
        lb = np.array([self.G.nodes[s]['cpu cap'] for s in self.G.nodes()]) # lower-bnd on the (augmented) cpu cap' that may be required
        while True: 
            
            # Check if the binary search converged 
            # check_ar = np.array ([ub[s] <= lb[s]+1 for s in self.G.nodes()], dtype='bool') 
            if ( np.array([ub[s] <= lb[s]+1 for s in self.G.nodes()], dtype='bool').all()):
                used_cpu = self.used_cpu_in ()
                print ('used cpu in s[0] = {}' .format (used_cpu[0]))
                return 

            # Update the available capacity at each server according to the value of resource augmentation for this iteration            
            for s in self.G.nodes():
                self.G.nodes[s]['cur RCs'] = math.floor (0.5*(ub[s] + lb[s]))  
            self.reset_sol()

            # Solve using bottom-up
            if (self.bottom_up()):
                ub = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()])        
            else:
                lb = np.array([self.G.nodes[s]['cur RCs'] for s in self.G.nodes()])
    
    def alg_top (self):
        """
        Our top-level alg'
        """
        if (self.verbose in [VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            printf (self.log_output_file, 'beginning alg top\n')
        # Try to solve the problem by changing the placement or CPU allocation only for the new / moved users
        if (not(self.bottom_up())):
            if (self.verbose in [VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
                printf (self.log_output_file, 'By binary search:\n')
            self.binary_search()

        # By hook or by crook, now we have a feasible solution        
        if (self.verbose in [VERBOSE_RES_AND_DETAILED_LOG]):
            printf (self.log_output_file, 'b4 push-up\n')
            self.print_sol_to_log()
        
        self.push_up ()
        if (self.verbose in [VERBOSE_RES_AND_LOG, VERBOSE_RES_AND_DETAILED_LOG]):
            printf (self.log_output_file, 'after push-up\n')
            self.print_sol_to_log()
        printf (self.log_output_file, 'finished alg top\n')

    # def update_available_cpu_by_sol (self):
    #     """
    #     Currently unused.
    #     Update the available capacity at each server to: 
    #     the (possibly augmented) CPU capacity at this server - the total CPU assigned to users by the solution.
    #     NOTE: this function assumes that every user is already exclusively located in its "nxt_s" location! 
    #     """
    #     for s in self.G.nodes():
    #         self.G.nodes[s]['a'] = self.G.nodes[s]['cur RCs'] - sum ([usr.B[usr.lvl] for usr in self.usrs if usr.nxt_s == s])                
   
    def bottom_up (self):
        """
        Bottom-up alg'. 
        Assigns all self.usrs that weren't assigned yet (either new usrs, or old usrs that moved, and now they don't satisfy the target delay).
        Looks for a feasible sol'.
        Returns true iff a feasible sol was found
        """        
        for s in range (len (self.G.nodes())-1, -1, -1): # for each server s, in an increasing order of levels (DFS).
            lvl = self.G.nodes[s]['lvl']
            Hs = [usr for usr in self.G.nodes[s]['Hs'] if (usr.lvl == -1)] # usr.lvl==-1 verifies that this usr wasn't placed yet
           
            for usr in sorted (Hs, key = lambda usr : len(usr.B)): # for each chain in Hs, in an increasing order of level ('L')
                printf (self.log_output_file, 's = {}. usr {} is critical. S_u = {}\n' .format (s, usr.id, usr.S_u)) 
                if (self.G.nodes[s]['a'] > usr.B[lvl]):
                    usr.nxt_s = s
                    usr.lvl   = lvl
                    self.G.nodes[s]['a'] -= usr.B[lvl]
                elif (len (usr.B)-1 == lvl):
                    return False
        return True

    def rd_new_usrs_line (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file.
        The input includes a list of tuples of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the tuples, the function assigns each chain to its relevant list of chains, Hs.  
        """
        
        if (line ==[]):
            return # no new users

        splitted_line = line[0].split ("\n")[0].split (")")

        for tuple in splitted_line:
            if (len(tuple) <= 1):
                break
            tuple = tuple.split("(")
            tuple   = tuple[1].split (',')
            
            usr = usr_c (id = int(tuple[0]), # generate a new usr, which is assigned as "un-placed" yet (usr.lvl==-1) 
                         theta_times_lambda=self.uniform_theta_times_lambda,
                         C_u = 10)
            self.CPUAll_single_usr (usr)
            self.usrs.append (usr)
            AP_id = int(tuple[1])
            if (AP_id >= self.num_of_leaves):
                if (self.warned_about_too_large_ap == False):
                    print ('Encountered AP num {} in the input file, but in the tree there are only {} leaves. Changing the ap to {}' .format (AP_id, self.num_of_leaves, self.num_of_leaves-1))
                    self.warned_about_too_large_ap = True
                AP_id = self.num_of_leaves-1
            s = self.ap2s[AP_id]
            usr.S_u.append (s)
            if (self.alg != 'alg_lp'): # the LP solver doesn't need the 'Hs' (list of chains that may be located on each server while satisfying the delay constraint)
                self.G.nodes[s]['Hs'].append(usr) # Hs is the list of chains that may be located on each server while satisfying the delay constraint
                for lvl in (range (len(usr.B)-1)):
                    s = self.parent_of(s)
                    usr.S_u.append (s)
                    self.G.nodes[s]['Hs'].append(usr)                       
                    
    def rd_old_usrs_line (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file.
        The input includes a list of tuples of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the tuples, the function assigns each chain to its relevant list of chains, Hs.  
        """
        if (line == []): # if the list of old users that moved is empty
            return
        
        splitted_line = line[0].split ("\n")[0].split (")")

        for tuple in splitted_line:
            if (len(tuple) <= 1):
                break
            tuple = tuple.split("(")
            tuple   = tuple[1].split (',')
            
            list_of_usr = list(filter (lambda usr : usr.id == int(tuple[0]), self.usrs))
            usr = list_of_usr[0]
            usr_cur_cpu = usr.B[usr.lvl]
            AP_id       = int(tuple[1])
            if (AP_id > self.num_of_leaves):
                AP_id = self.num_of_leaves-1
                if (self.warned_about_too_large_ap == False):
                    print ('Encountered AP num {} in the input file, but in the tree there are only {} leaves. Changing the ap to {}' .format (AP_id, self.num_of_leaves, self.num_of_leaves-1))
                    self.warned_about_too_large_ap = True
            self.CPUAll_single_usr (usr)

            # update the list of delay-feasible servers for this usr
            s = self.ap2s[AP_id]
            usr.S_u = []
            usr.S_u.append (s)
            self.G.nodes[s]['Hs'].append(usr)
            for lvl in (range (len(usr.B)-1)):
                s = self.parent_of(s)
                usr.S_u.append (s)
                self.G.nodes[s]['Hs'].append(usr)                               
    
            if (usr.cur_s in usr.S_u and usr_cur_cpu <= usr.B[usr.lvl]): # Can satisfy delay constraint while staying in the cur location and keeping the CPU budget 
                continue
            
            # printf (self.log_output_file, 'usr {} is critical. S_u = {}\n' .format (usr.id, usr.S_u)) $$$
            # Now we know that this is a critical usr, namely a user that needs more CPU and/or migration for satisfying its target delay constraint 
            # dis-place this user (mark it as having nor assigned level, neither assigned server), and free its assigned CPU 
            self.G.nodes[usr.cur_s]['a'] += usr.B[usr.lvl] # free the CPU units used by the user in the old location
            usr.lvl   = -1
            usr.nxt_s = -1

    def calc_sol_cost_CLP (self):
        """
        Calculate the total cost of a CLP (Co-Located Placement), where all the VMs of each chain are co-located on a single server.
        """
        total_cost = 0
        for chain in range(self.NUM_OF_CHAINS):
            total_cost += self.CPU_cost[self.chain_nxt_loc[chain]] * self.chain_nxt_total_alloc[chain] + \
                        self.path_bw_cost[self.PoA_of_user[chain]]  [self.chain_nxt_loc[chain]] * self.lambda_v[chain][0] + \
                        self.path_bw_cost[self.chain_nxt_loc[chain]][self.PoA_of_user[chain]]   * self.lambda_v[chain][self._in_chain[chain]] + \
                        (self.chain_cur_loc[chain] != self.chain_nxt_loc[chain]) * self.chain_mig_cost[chain]
                
    def inc_array (self, ar, min_val, max_val):
        """
        input: an array, in which elements[i] is within [min_val[i], max_val[i]] for each i within the array's size
        output: the same array, where the value is incremented by 1 
        Used for finding a brute-force solution.
        """
        for idx in range (ar.size-1, -1, -1):
            if (ar[idx] < max_val[idx]):
                ar[idx] += 1
                return ar
            ar[idx] = min_val[idx]
        return ar 
     
if __name__ == "__main__":

    t = time.time()
    my_simulator = SFC_mig_simulator (ap_file_name = 'shorter.ap', verbose = VERBOSE_RES_AND_DETAILED_LOG, tree_height = 1, children_per_node=2)
    my_simulator.simulate ('alg_top')
    # my_simulator.calc_sol_cost_CLP ()
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

