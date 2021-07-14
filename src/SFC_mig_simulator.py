# Bugs: 
# plp has RCs > 30 for s0? 
# Plp's cost is higher than that of alg_top? probably due to added mig' cost
import networkx as nx
import numpy as np
import math
import itertools 
import time
import random
import heapq
import pulp as plp
from cmath import sqrt
# from scipy.optimize import linprog # currently unused

from usr_c    import usr_c    # class of the users of alg
from usr_lp_c import usr_lp_c # class of the users, when using LP
from decision_var_c import decision_var_c # class of the decision variables
from printf import printf
import loc2ap_c

# Levels of verbose (which output is generated)
VERBOSE_DEBUG    = 0
VERBOSE_RES      = 1 # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_LOG      = 2 # Write to a ".log" file
VERBOSE_ADD_LOG  = 3 # Write to a detailed ".log" file

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
    calc_sol_cost = lambda self: sum ([self.chain_cost_homo (usr, usr.lvl) for usr in self.usrs])
   
    # calculate the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    chain_cost_homo = lambda self, usr, lvl: self.link_cost_of_CLP_at_lvl[lvl] + self.CPU_cost_at_lvl[lvl] * usr.B[lvl] + self.calc_mig_cost (usr, lvl)     
    
    # # calculate the migration cost incurred for a usr if located on a given lvl
    calc_mig_cost = lambda self, usr, lvl : (usr.S_u[lvl] != usr.cur_s and usr.cur_s!=-1) * self.uniform_mig_cost * len (usr.theta_times_lambda)
          
    # Calculate the number of CPU units actually used in each server
    used_cpu_in_all_srvrs = lambda self: np.array ([self.G.nodes[s]['RCs'] - self.G.nodes[s]['a'] for s in self.G.nodes])      
          
    # calculate the proved upper bnd on the rsrc aug that bottomUp may need to find a feasible sol, given such a sol exists for the non-augmented prob'
    calc_upr_bnd_rsrc_aug = lambda self: np.max ([usr.C_u for usr in self.usrs]) / np.min ([np.min (usr.B) for usr in self.usrs])

    # # Returns the AP covering a given (x,y) location, assuming that the cells are identical fixed-size squares
    loc2ap_sq = lambda self, x, y: int (math.floor ((y / self.cell_Y_edge) ) * self.num_of_APs_in_row + math.floor ((x / self.cell_X_edge) )) 

    # Returns the server to which a given user is currently assigned
    cur_server_of = lambda self, usr: usr.S_u[usr.lvl] 

    # Returns the total amount of cpu used by users at a certain server
    used_cpu_in = lambda self, s: sum ([usr.B[usr.lvl] for usr in self.usrs if usr.nxt_s==s])

    lp_used_cpu_in = lambda self, s: sum ( np.array ( [d_var.usr.B[self.G.nodes[s]['lvl']] * d_var.plp_var.value() for d_var in list (filter (lambda d_var : d_var.s == s, self.d_vars))]))
    
    # Calculates the cost of locating the whole chain of a given user on a server at a given lvl in its Su.
    # This is when when the current state may be non co-located-placement. That is, distinct VMs (or fractions) of the same chain may be found in several distinnct server. 
    chain_cost_from_non_CLP_state = lambda self, usr, lvl: \
                    sum ([param.cur_st for param in list (filter (lambda param: param.usr == usr and param.s != usr.S_u[lvl], self.cur_st_params))]) * \
                    self.uniform_mig_cost + self.CPU_cost_at_lvl[lvl] * usr.B[lvl] + self.link_cost_of_CLP_at_lvl[lvl]

    def set_last_time (self):
        """
        If needed by the verbose level, set the variable 'self.last_rt' (last measured real time), to be read later for calculating the time taken to run code pieces
        """
        if (VERBOSE_ADD_LOG in self.verbose):
            self.last_rt = time.time()
     
    def print_sol_to_res_and_log (self):
        """
        Print to the res and/or log files the solution and/or additional info.
        """
        if (VERBOSE_RES in self.verbose):
            self.print_sol_to_res()
        if (VERBOSE_ADD_LOG in self.verbose):
            printf (self.log_output_file, '\nSolved in {:.3f} [sec]\n' .format (time.time() - self.last_rt))
     
    def calc_rsrc_aug (self):
        """
        Calculate the (maximal) rsrc aug' used by the current solution, using a single (scalar) R
        """
        used_cpu_in = self.used_cpu_in_all_srvrs ()
        return max (np.max ([(used_cpu_in[s] / self.G.nodes[s]['cpu cap']) for s in self.G.nodes()]), 1)    

    def rst_sol (self):
        """
        Reset the solution, namely, Dis-place all users. This is done by: 
        1. Resetting the placement of each user to a concrete level in the tree, and to a concrete server.
        2. Init the available cpu at each server to its (possibly augmented) cpu capacity. 
        """
        for usr in self.usrs:
            usr.lvl   = -1
            usr.nxt_s = -1
        for s in self.G.nodes():
            self.G.nodes[s]['a'] = self.G.nodes[s]['RCs']
        # print ('in rst sol: RCS={}, a={}' .format (self.G.nodes[0]['RCs'], self.G.nodes[0]['a']))


    # def calc_decision_var_cost (self):
    #     """
    #     Not complete yet.
    #     Caluclate the cost of setting a decision var, when using the linear prog'
    #     """
    #     # consider the BW and CPU cost
    #     cost = self.link_cost_of_CLP_at_lvl[decision_var.lvl] + self.CPU_cost_at_lvl[decision_var.lvl] * decision_var.usr.B[decision_var.lvl] 
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
                plp_var = plp.LpVariable (lowBound=0, upBound=1, name='x_{}' .format (id))
                decision_var = decision_var_c (id=id, usr=usr, lvl=lvl, s=usr.S_u[lvl], plp_var=plp_var) # generate a decision var, containing the lp var + details about its meaning 
                self.d_vars.append (decision_var)
                single_place_const += plp_var
                obj_func           += self.chain_cost_from_non_CLP_state (usr, lvl) * plp_var # add the cost of this decision var to the objective func
                id += 1
            model += (single_place_const == 1) # demand that each chain is placed in a single server
        model += obj_func

        # Generate CPU capacity constraints
        for s in self.G.nodes():
            cpu_cap_const = []
            for d_var in list (filter (lambda item : item.s == s, self.d_vars)): # for every decision variable meaning placing a chain on this server 
                cpu_cap_const += (d_var.usr.B[d_var.lvl] * d_var.plp_var) # Add the overall cpu of this chain, if located on s
            if (cpu_cap_const != []):
                model += (cpu_cap_const <= self.G.nodes[s]['RCs']) 

        # solve using another solver: solve(GLPK(msg = 0))
        model.solve(plp.PULP_CBC_CMD(msg=0)) # solve the model, without printing output
        
        if (VERBOSE_RES in self.verbose):
            printf (self.res_output_file, 't{}\n' .format(self.t)) 
            printf (self.res_output_file, 'plp.stts{} \n' .format(model.status)) 
            printf (self.res_output_file, 't{}.plp.stts{} cost={:.2f}\n' .format(self.t, model.status, model.objective.value())) 
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_output_file, 't{}.plp.stts{} cost={:.2f}\n' .format(self.t, model.status, model.objective.value())) 
                                                                            #plp.LpStatus[model.status]))
        if (VERBOSE_LOG in self.verbose): 
            if (model.status == 1): # successfully solved
                self.print_lp_sol_to_log ()
            else:
                printf (self.log_output_file, 'failed. status={}\n' .format(plp.LpStatus[model.status]))

        exit ()
        # Make the solution the "current state", for the next time slot  
        
    def print_cost_per_usr (self):
        """
        For debugging / analysis:
        print the cost of each chain. 
        """     
        for usr in self.usrs:
            chain_cost = self.chain_cost_homo (usr, usr.lvl)  
            print ('cost of usr {} = {}' .format (u, chain_cost))
        
    # def init_output_file (self, file_name, overwrite=False):
    #     """
    #     Open an output file for writing, overwriting previous content in that file. 
    #     Input: output file name.
    #     Output: file descriptor.  
    #     """
    #     if (overwrite):
    #         with open('../res/' + file_name, 'w') as FD:
    #             FD.write('')                
    #     FD  = open ('../res/' + file_name,  "w")
    #     return FD

    def init_res_file (self):
        """
        Open the res file for writing.
        """
        self.res_file_name = "../res/" + self.ap_file_name.split(".")[0] + ".res"  
        self.res_output_file =  open ('../res/' + self.res_file_name,  "a") 

    def init_log_file (self, overwrite = True):
        """
        Open the log file for writing and write initial comments lines on it
        """
        self.log_file_name = "../res/" + self.ap_file_name.split(".")[0] + '.' + self.alg.split("_")[1] + '.log'  
        self.log_output_file =  open ('../res/' + self.log_file_name,  "w") 
        printf (self.log_output_file, '//RCs = augmented capacity of server s. a=available capacity. C_s = non-augmented capacity of s.\n' )

    def print_sol_to_res (self):
        """
        print a solution for the problem to the output log file 
        """
        used_cpu_in = self.used_cpu_in_all_srvrs ()
        printf (self.res_output_file, 't{}.alg cost={:.2f} rsrc_aug={:.2f}\n' .format(
            self.t, 
            self.calc_sol_cost(),
            self.calc_rsrc_aug())) 

    def print_lp_sol_to_log (self):
        """
        print a lp fractional solution for the problem to the output log file 
        """
        # for usr in self.usrs:
        #     for decision_var in list (filter (lambda decision_var : decision_var.usr == usr and decision_var, self.decision_vars)): # list_of_relevant_decision_vars =
        for s in self.G.nodes():
            printf (self.log_output_file, 's{} RCs={} used_cpu={}\n' .format (s, self.G.nodes[s]['RCs'], self.lp_used_cpu_in (s) ))

        if (VERBOSE_ADD_LOG in self.verbose): 
            for d_var in self.d_vars: 
                if d_var.plp_var.value() > 0:
                    printf (self.log_output_file, '\nu {} lvl {:.0f} s {:.0f} val {:.2f}' .format(
                           d_var.usr.id, d_var.lvl, d_var.s, d_var.plp_var.value()))            

    def print_sol_to_log (self):
        """
        print a detailed solution for the mig' problem to the output log file 
        """
        printf (self.log_output_file, 't{}.alg cost={:.2f} rsrc_aug={:.2f}\n' .format(
            self.t, 
            self.calc_sol_cost(),
            self.calc_rsrc_aug())) 
        
        for s in self.G.nodes():
            used_cpu_in_s = self.used_cpu_in (s)
            printf (self.log_output_file, 's{} : Rcs={}, a={}, used cpu={:.0f}, Cs={}\t chains {}\n' .format (
                    s,
                    self.G.nodes[s]['RCs'],
                    self.G.nodes[s]['a'],
                    used_cpu_in_s,
                    self.G.nodes[s]['cpu cap'],
                    [usr.id for usr in self.usrs if usr.nxt_s==s]))
            self.check_cpu_usage_single_srvr(s)
            
    def check_cpu_usage_all_srvrs (self):
        """
        Used for debug. Checks for all cells whether the allocated cpu + the available cpu = the total cpu.
        """
        for s in self.G.nodes():
            self.check_cpu_usage_single_srvr (s)
            
    def check_cpu_usage_single_srvr (self, s):
        """
        Used for debug. Checks for all cells whether the allocated cpu + the available cpu = the total cpu.
        """
        if (self.used_cpu_in(s) + self.G.nodes[s]['a'] != self.G.nodes[s]['RCs']):
            printf (self.log_output_file, 'Error in calculating the cpu utilization of s{}: used_cpu = {}, a={}, Rcs={}' .format 
                    (s, self.used_cpu_in(s), self.G.nodes[s]['a'], self.G.nodes[s]['RCs']))
            print ('Error in using cpu utilization. Please see the log file: {}' .format (self.log_file_name))
            exit ()           
            
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
                if (self.G.nodes[usr.S_u[lvl]]['a'] >= usr.B[lvl] and self.chain_cost_homo(usr, lvl) < self.chain_cost_homo(usr, usr.lvl)): # if there's enough available space to move u to level lvl, and this would reduce cost
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

    def CPUAll_single_usr (self, usr): 
        """
        CPUAll algorithm, for a single usr:
        calculate the minimal CPU allocation required by the given usr, when the highest server on which u is located is s.
        The current implementation assumes that the network is a balanced tree, and the delays of all link of level $\ell$ are identical.
        This version of the alg' assumes a balanced homogeneous tree, 
        so that the netw' delay between every two servers is unequivocally defined by their levels.
        """
        slack = [usr.target_delay - self.link_delay_of_CLP_at_lvl[lvl] for lvl in range (self.tree_height+1)]
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
            
    def gen_parameterized_tree (self):
        """
        Generate a parameterized tree with specified height and children-per-non-leaf-node. 
        """
        self.G                 = nx.generators.classic.balanced_tree (r=self.children_per_node, h=self.tree_height) # Generate a tree of height h where each node has r children.
        self.cpu_cap_at_lvl    = np.array ([10 * (lvl+1) for lvl in range (self.tree_height+1)], dtype='uint16')
        if (self.ap_file_name != 'shorter.ap'):
            self.cpu_cap_at_lvl *= 100
        self.CPU_cost_at_lvl   = [1 * (self.tree_height + 1 - lvl) for lvl in range (self.tree_height+1)]
        self.link_cost_at_lvl  = self.uniform_link_cost * np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of using a link from level i to level i+1, or vice versa.
        self.link_delay_at_lvl = np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of using a link from level i to level i+1, or vice versa.
        
        # overall link cost and link capacity of a Single-Server Placement of a chain at each lvl
        self.link_cost_of_CLP_at_lvl  = [2 * sum([self.link_cost_at_lvl[i]  for i in range (lvl)]) for lvl in range (self.tree_height+1)]
        self.link_delay_of_CLP_at_lvl = [2 * sum([self.link_delay_at_lvl[i] for i in range (lvl)]) for lvl in range (self.tree_height+1)] 
        
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
                    # self.G.nodes[shortest_path[s][root][lvl]]['link cost'] = self.link_cost_of_CLP_at_lvl[lvl]
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

    def __init__ (self, ap_file_name = 'shorter.ap', verbose = [], tree_height = 3, children_per_node = 4):
        """
        """
        
        # verbose and debug      
        self.debug                      = False 
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
        self.usrs                       = []
        
        # Init output files
        if (VERBOSE_RES in self.verbose):
            self.init_res_file() 
        self.gen_parameterized_tree ()

    def simulate (self, alg):
        """
        Simulate the whole simulation using the chosen alg: LP, or ALG_TOP (our alg).
        """
        self.alg = alg
        print ('Simulating {}. num of leaves = {}. ap file = {}' .format (self.alg, self.num_of_leaves, self.ap_file_name))
        if (self.alg == 'alg_top'):
            self.simulate_alg_top()
        elif (self.alg == 'alg_lp'):
            self.simulate_lp ();
        else:
            print ('Sorry, alg {} is not implemented yet')
            exit ()
    
    def simulate_lp (self):
        """
        Simulate the whole simulation, using a LP fractional solution.
        At each time step:
        - Read and parse from an input ".ap" file the AP cells of each user who moved. 
        - solve the problem using a LP, using Python's Pulp LP solver. 
        - Write outputs results and/or logs to files.
        - update cur_st = nxt_st
        """
        self.cur_st_params = []
        # Init RCs       
        self.rsrc_aug = 1
        for s in self.G.nodes():
            self.G.nodes[s]['RCs'] = self.rsrc_aug * self.G.nodes[s]['cpu cap'] # for now, assume no resource aug' 

        print ('rsrc aug = {}' .format (self.rsrc_aug))
        
        if (VERBOSE_LOG in self.verbose):
            self.init_log_file()
        # Open input and output files
        self.ap_file  = open ("../res/" + self.ap_file_name, "r")  
        if (VERBOSE_RES in self.verbose):
            self.init_log_file()
                    
        for line in self.ap_file: 
        
            # Ignore comments and emtpy lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue
        
            line = line.split ('\n')[0]
            splitted_line = line.split (" ")
        
            if (splitted_line[0] == "t"):
                self.t = int(splitted_line[2])
                if (VERBOSE_LOG in self.verbose):
                    printf (self.log_output_file, '\ntime = {}\n**************************************\n' .format (self.t))
                continue
        
            elif (splitted_line[0] == "usrs_that_left:"):
        
                for usr in list (filter (lambda usr : usr.id in [int(usr_id) for usr_id in splitted_line[1:] if usr_id!=''], self.usrs)):
        
                    self.rmv_usr_rsrcs(usr) #Remove the rsrcs used by this usr
                    self.usrs.remove  (usr)                    
                continue
        
            elif (splitted_line[0] == "new_usrs:"):              
                self.rd_new_usrs_line_lp (splitted_line[1:])
            elif (splitted_line[0] == "old_usrs:"):              
                self.rd_old_usrs_line_lp (splitted_line[1:])
            if (VERBOSE_DEBUG in self.verbose and (len(self.usrs)==0)):
                print ('Error: there are no usrs')
                exit ()
            self.set_last_time()
            self.solve_by_plp () 
            self.print_sol_to_res_and_log ()
            exit () #$$$
            
    def simulate_alg_top (self):
        """
        Simulate the whole simulation, using our algorithm, alg_top.
        At each time step:
        - Read and parse from an input ".ap" file the AP cells of each user who moved. 
        - solve the problem using alg_top (our alg). 
        - Write outputs results and/or logs to files.
        - update cur_st = nxt_st
        """
        if (VERBOSE_LOG in self.verbose):
            self.init_log_file()
             
        # reset Hs and RCs       
        for s in self.G.nodes():
            self.G.nodes[s]['Hs']  = set() 
            self.G.nodes[s]['RCs'] = self.G.nodes[s]['cpu cap'] # Initially, no rsrc aug --> at each server, we've exactly his non-augmented capacity. 

        # Open input and output files
        self.ap_file  = open ("../res/" + self.ap_file_name, "r")  
        if (VERBOSE_RES in self.verbose):
            self.init_log_file()
            
        for line in self.ap_file: 

            # Ignore comments lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue

            line = line.split ('\n')[0]
            splitted_line = line.split (" ")

            if (splitted_line[0] == "t"):
                self.t = int(splitted_line[2])
                if (VERBOSE_LOG in self.verbose):
                    printf (self.log_output_file, '\ntime = {}\n**************************************\n' .format (self.t))
                continue
        
            elif (splitted_line[0] == "usrs_that_left:"):

                for usr in list (filter (lambda usr : usr.id in [int(usr_id) for usr_id in splitted_line[1:] if usr_id!=''], self.usrs)):

                    self.rmv_usr_rsrcs(usr) #Remove the rsrcs used by this usr
                    self.usrs.remove (usr)                    
                continue
        
            elif (splitted_line[0] == "new_usrs:"):              
                self.rd_new_usrs_line (splitted_line[1:])
            elif (splitted_line[0] == "old_usrs:"):              
                self.rd_old_usrs_line (splitted_line[1:])                
                self.set_last_time()
                self.alg_top()
                self.print_sol_to_res_and_log ()
                for usr in self.usrs: # The solution found at this time slot is the "cur_state" for next slot
                     usr.cur_s = usr.nxt_s        
               
    def binary_search (self):
        """
        Binary search for a feasible sol that minimizes the resource augmentation R.
        The search is done by calling bottom_up ().
        """
        self.rst_sol() # dis-allocate all users
        self.CPUAll(self.usrs) 
        max_R = self.calc_upr_bnd_rsrc_aug () 
        
        # init cur RCs and a(s) to the number of available CPU in each server, assuming maximal rsrc aug' 
        for s in self.G.nodes():
            self.G.nodes[s]['RCs'] = math.ceil (max_R * self.G.nodes[s]['cpu cap']) 
            self.G.nodes[s]['a']   = self.G.nodes[s]['RCs'] #currently-available rsrcs at server s  

        if (not (self.bottom_up ())):
            print ('did not find a feasible sol even with maximal rsrc aug')
            exit ()

        # Now we know that we found an initial feasible sol 
                   
        ub = np.array([self.G.nodes[s]['RCs']     for s in self.G.nodes()]) # upper-bnd on the (augmented) cpu cap' that may be required
        lb = np.array([self.G.nodes[s]['cpu cap'] for s in self.G.nodes()]) # lower-bnd on the (augmented) cpu cap' that may be required
        
        while True: 
             
            if ( np.array([ub[s] <= lb[s]+1 for s in self.G.nodes()], dtype='bool').all()): # Did the binary search converged?
                for s in self.G.nodes(): # Yep, so allocate this minimal found amount of rsrc aug to all servers
                    self.G.nodes[s]['RCs'] = math.floor (ub[s])  
                self.rst_sol()         # and re-solve the prob'
                res = self.bottom_up()
                return 

            # Now we know that the binary search haven't converged yet
            # Update the available capacity at each server according to the value of resource augmentation for this iteration            
            for s in self.G.nodes():
                self.G.nodes[s]['RCs'] = math.floor (0.5*(ub[s] + lb[s]))  
            self.rst_sol()

            # Solve using bottom-up
            if (self.bottom_up()):
                if (VERBOSE_ADD_LOG in self.verbose): 
                    printf (self.log_output_file, 'In bottom-up IF\n')
                    self.check_cpu_usage_all_srvrs()
                    self.print_sol_to_log()
                ub = np.array([self.G.nodes[s]['RCs'] for s in self.G.nodes()])        
            else:
                lb = np.array([self.G.nodes[s]['RCs'] for s in self.G.nodes()])
    
    def alg_top (self):
        """
        Our top-level alg'
        """
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_output_file, 'beginning alg top\n')
        # Try to solve the problem by changing the placement or CPU allocation only for the new / moved users
        if (not(self.bottom_up())):
            if (VERBOSE_LOG in self.verbose):
                printf (self.log_output_file, 'By binary search:\n')
            self.binary_search()

        # By hook or by crook, now we have a feasible solution        
        if (VERBOSE_ADD_LOG in self.verbose): 
            printf (self.log_output_file, 'b4 push-up\n')
            self.print_sol_to_log()
        
        self.push_up ()
        if (VERBOSE_LOG in self.verbose):
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
    #         self.G.nodes[s]['a'] = self.G.nodes[s]['RCs'] - sum ([usr.B[usr.lvl] for usr in self.usrs if usr.nxt_s == s])                
   
    def bottom_up (self):
        """
        Bottom-up alg'. 
        Assigns all self.usrs that weren't assigned yet (either new usrs, or old usrs that moved, and now they don't satisfy the target delay).
        Looks for a feasible sol'.
        Returns true iff a feasible sol was found
        """        
        for s in range (len (self.G.nodes())-1, -1, -1): # for each server s, in an increasing order of levels (DFS).v
            lvl = self.G.nodes[s]['lvl']
            Hs = [usr for usr in self.G.nodes[s]['Hs'] if (usr.lvl == -1)] # usr.lvl==-1 verifies that this usr wasn't placed yet
            for usr in sorted (Hs, key = lambda usr : len(usr.B)): # for each chain in Hs, in an increasing order of level ('L')
                if (self.G.nodes[s]['a'] > usr.B[lvl]):
                    usr.nxt_s = s
                    usr.lvl   = lvl
                    self.G.nodes[s]['a'] -= usr.B[lvl]
                elif (len (usr.B)-1 == lvl):
                    return False
        return True

    def rd_new_usrs_line_lp (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file, when looking for a LP solution for the problem
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
    
            usr = usr_lp_c (id = int(tuple[0])) # generate a new usr, which is assigned as "un-placed" yet (usr.lvl==-1) 
            self.CPUAll_single_usr (usr)
            self.usrs.append (usr)
            AP_id = int(tuple[1])
            if (AP_id >= self.num_of_leaves):
                if (self.warned_about_too_large_ap == False):
                    print ('********* WARNING: Encountered AP num {} in the input file, but in the tree there are only {} leaves. Changing the ap to {} *********' .format (AP_id, self.num_of_leaves, self.num_of_leaves-1))
                    exit ()
            s = self.ap2s[AP_id]
            usr.S_u.append (s)
            for lvl in (range (len(usr.B)-1)):
                s = self.parent_of(s)
                usr.S_u.append (s)

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
                    print ('********* WARNING: Encountered AP num {} in the input file, but in the tree there are only {} leaves. Changing the ap to {} *********' .format (AP_id, self.num_of_leaves, self.num_of_leaves-1))
                    exit ()
                    self.warned_about_too_large_ap = True
                AP_id = self.num_of_leaves-1
            s = self.ap2s[AP_id]
            usr.S_u.append (s)
            self.G.nodes[s]['Hs'].add(usr) # Hs is the list of chains that may be located on each server while satisfying the delay constraint
            for lvl in (range (len(usr.B)-1)):
                s = self.parent_of(s)
                usr.S_u.append (s)
                self.G.nodes[s]['Hs'].add(usr)                       
                    
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
            if (len(list_of_usr) == 0):
                print  ('Error at t={}: input file={}. Did not find old / recycled usr {}' .format (self.t, self.ap_file_name, tuple[0]))
                exit ()
            usr = list_of_usr[0]
            usr_cur_cpu = usr.B[usr.lvl]
            AP_id       = int(tuple[1])
            if (AP_id > self.num_of_leaves):
                AP_id = self.num_of_leaves-1
                if (self.warned_about_too_large_ap == False):
                    print ('Encountered AP num {} in the input file, but in the tree there are only {} leaves. Changing the ap to {}' .format (AP_id, self.num_of_leaves, self.num_of_leaves-1))
                    self.warned_about_too_large_ap = True
                    exit ()
            self.CPUAll_single_usr (usr)

            # Add this usr to the Hs of every server to which it belongs at its new location
            s       = self.ap2s[AP_id]
            usr.S_u = [s]
            self.G.nodes[s]['Hs'].add(usr)
            for lvl in (range (len(usr.B)-1)):
                s = self.parent_of(s)
                usr.S_u.append (s)
                self.G.nodes[s]['Hs'].add(usr)                               
    
            if (usr.cur_s in usr.S_u and usr_cur_cpu <= usr.B[usr.lvl]): # Can satisfy delay constraint while staying in the cur location and keeping the CPU budget 
                continue
            # Free the resources of this user in its old, current place
            self.rmv_usr_rsrcs (usr)            
            # Now we know that this is a critical usr, namely a user that needs more CPU and/or migration for satisfying its target delay constraint 
            # dis-place this user (mark it as having nor assigned level, neither assigned server), and free its assigned CPU 
            usr.lvl   = -1
            usr.nxt_s = -1

    def rd_old_usrs_line_lp (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file, when using a LP solver for the problem.
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
            if (len(list_of_usr) == 0):
                print  ('Error at t={}: input file={}. Did not find old / recycled usr {}' .format (self.t, self.ap_file_name, tuple[0]))
                exit ()
            usr    = list_of_usr[0]
            AP_id  = int(tuple[1])
            if (AP_id > self.num_of_leaves):
                AP_id = self.num_of_leaves-1
                if (self.warned_about_too_large_ap == False):
                    print ('Encountered AP num {} in the input file, but in the tree there are only {} leaves. Changing the ap to {}' .format (AP_id, self.num_of_leaves, self.num_of_leaves-1))
                    exit ()
            self.CPUAll_single_usr (usr)

            # Add this usr to the Hs of every server to which it belongs at its new location
            s       = self.ap2s[AP_id]
            usr.S_u = [s]
            for lvl in (range (len(usr.B)-1)):
                s = self.parent_of(s)
                usr.S_u.append (s)
    
            # Free the resources of this user in its old, current place
            self.rmv_usr_rsrcs (usr)            

    def rmv_usr_rsrcs (self, usr):
        """
        Remove a usr from the Hs (relevant usrs) of every server to which it belonged, at its previous location; 
        and increase the avilable rsrcs of the srvr that currently place this usr
        """
        for s in [s for s in self.G.nodes() if usr in self.G.nodes[s]['Hs']]:
            self.G.nodes[s]['Hs'].remove (usr) 
        self.G.nodes[usr.cur_s]['a'] += usr.B[usr.lvl] # free the CPU units used by the user in the old location
            
    
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
    my_simulator = SFC_mig_simulator (ap_file_name = 'shorter.ap', verbose = [VERBOSE_RES, VERBOSE_LOG, VERBOSE_ADD_LOG], tree_height = 2, children_per_node=2)
    my_simulator.simulate ('alg_lp')
