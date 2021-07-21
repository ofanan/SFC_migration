import networkx as nx
import numpy as np
import math
import itertools 
import time
import heapq
import pulp as plp
from cmath import sqrt
import matplotlib.pyplot as plt
# from scipy.optimize import linprog # currently unused

from usr_c    import usr_c    # class of the users of alg
from usr_lp_c import usr_lp_c # class of the users, when using LP
from decision_var_c import decision_var_c # class of the decision variables
from printf import printf
import loc2ap_c

# Levels of verbose (which output is generated)
VERBOSE_DEBUG     = 0
VERBOSE_RES       = 1 # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_LOG       = 2 # Write to a ".log" file
VERBOSE_ADD_LOG   = 3 # Write to a detailed ".log" file
VERBOSE_MOB       = 4 # Write data about the mobility of usrs, and about the num of migrated chains per cycle
VERBOSE_COST_COMP = 5 # Print the cost of each component in the cost function
# Status returned by algorithms solving the prob' 
sccs = 1
fail = 2

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
                    
    # Write an output line for the LP sol, to a given output file (given as a FD==file descriptor) 
    write_lp_res_line = lambda self, FD, model : printf (FD, 't{}.lp.stts{} | cost = {:.0f}\n' .format(self.t, model.status, model.objective.value()))

    # Print a solution for the problem to the output res file 
    print_sol_res_line = lambda self, output_file : printf (output_file, 't{}.{}.stts{} | cost = {:.0f} | rsrc_aug = {:.2f}\n' .format(self.t, self.alg, self.stts, self.calc_sol_cost(), self.rsrc_aug)) 

    # parse a line detailing the list of usrs who moved, in an input ".ap" format file
    parse_old_usrs_line = lambda self, line : list (filter (lambda item : item != '', line[0].split ("\n")[0].split (")")))

    # returns true iff server s has enough available capacity to host user u; assuming that usr.B (the required cpu allowing to place u on each lvl) is known  
    s_has_sufic_avail_cpu_for_usr = lambda self, s, usr : (self.G.nodes[s]['a'] >= usr.B[self.G.nodes[s]['lvl']])
    
    # returns a list of the critical usrs
    critical_usrs = lambda self : list (filter (lambda usr : usr.nxt_s==-1, self.usrs)) 
    
    def set_last_time (self):
        """
        If needed by the verbose level, set the variable 'self.last_rt' (last measured real time), to be read later for calculating the time taken to run code pieces
        """
        if (VERBOSE_LOG in self.verbose):
            self.last_rt = time.time()

    def print_sol_cost_components (self):
        """
        prints to a file statistics about the cost of each component in the cost function (cpu, link, and migration). 
        """
        
        total_cost = [self.total_cpu_cost_in_slot[t] + self.total_link_cost_in_slot[t] + self.total_mig_cost_in_slot[t] for t in range(len(self.total_cpu_cost_in_slot))]        
        printf (self.cost_comp_output_file, 'total_cost = {}\n' .format (total_cost))
        printf (self.cost_comp_output_file, 'cpu_cost={}\nlink_cost={}\nmig_cost={}\n' .format (
                self.total_cpu_cost_in_slot, self.total_link_cost_in_slot, self.total_mig_cost_in_slot))

        cpu_cost_ratio  = [self.total_cpu_cost_in_slot[t]/total_cost[t]  for t in range(len(total_cost))]
        link_cost_ratio = [self.total_link_cost_in_slot[t]/total_cost[t] for t in range(len(total_cost))]
        mig_cost_ratio  = [self.total_mig_cost_in_slot[t]/total_cost[t]  for t in range(len(total_cost))]
        printf (self.cost_comp_output_file, 'cpu_cost_ratio = {:.3f}\nlink_cost_ratio = {:.3f}\nmig_cost_ratio = {:.3f}\n'.format (
            cpu_cost_ratio, link_cost_ratio, link_cost_ratio))
            
        printf (self.cost_comp_output_file, 'avg ratio are: cpu={:.3f}, link={:.3f}, mig={:.3f}\n' .format (
            np.average(cpu_cost_ratio), np.average(link_cost_ratio), np.average(mig_cost_ratio) ) )
            
    def calc_sol_cost_components (self):
        """
        Calculates and keeps the cost of each component in the cost function (cpu, link, and migration). 
        """
        
        self.total_cpu_cost_in_slot.append  (sum ([self.CPU_cost_at_lvl[usr.lvl] * usr.B[usr.lvl] for usr in self.usrs]))
        self.total_link_cost_in_slot.append (sum ([self.link_cost_of_CLP_at_lvl[usr.lvl]          for usr in self.usrs]))
        self.total_mig_cost_in_slot.append  (sum ([self.calc_mig_cost(usr, usr.lvl)               for usr in self.usrs]))
     
    def print_sol_to_res_and_log (self):
        """
        Print to the res, log, and debug files the solution and/or additional info.
        """
        if (VERBOSE_RES in self.verbose):
            self.print_sol_res_line(self.res_output_file)
        elif (VERBOSE_COST_COMP in self.verbose):
            self.calc_sol_cost_components()

        if (VERBOSE_LOG in self.verbose):
            self.print_sol_res_line(self.log_output_file)
            printf (self.log_output_file, '\nSolved in {:.3f} [sec]\n' .format (time.time() - self.last_rt))
            self.print_sol_to_log()
            if (self.stts != sccs):
                printf (self.log_output_file, 'Note: the solution above is partial, as the alg did not find a feasible solution\n')
                return         
        if (VERBOSE_DEBUG in self.verbose and self.stts==sccs): 
            for usr in self.usrs:
                if (usr.lvl==-1):
                    error_msg = 'Error: t={}. stts={}, but usr {} is not placed\n' .format (self.t, self.stts, usr.id)
                    print  (error_msg)
                    printf (self.debug_file, error_msg)
                    exit ()
        if (VERBOSE_MOB in self.verbose):
            usrs_who_migrated_at_this_cycle = list (filter (lambda usr: usr.cur_s != -1 and usr.cur_s != usr.nxt_s, self.usrs))
            self.num_of_migs_in_cycle.append (len(usrs_who_migrated_at_this_cycle))
            for usr in usrs_who_migrated_at_this_cycle: 
                self.mig_from_to_lvl[self.G.nodes[usr.cur_s]['lvl']] [self.G.nodes[usr.nxt_s]['lvl']] += 1
            
    def update_rsrc_aug (self):
        """
        Calculate the (maximal) rsrc aug' used by the current solution, using a single (scalar) R
        """
        used_cpu_in = self.used_cpu_in_all_srvrs ()
        self.rsrc_aug  = max (np.max ([(used_cpu_in[s] / self.G.nodes[s]['cpu cap']) for s in self.G.nodes()]), self.rsrc_aug) # this is the minimal rsrc aug to be used from now and on    
        return self.rsrc_aug

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

    def solve_by_plp (self):
        """
        Find an optimal fractional solution using Python's pulp LP library.
        pulp library can use commercial tools (e.g., Gurobi, Cplex) to efficiently solve the prob'.
        """
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

        model.solve(plp.PULP_CBC_CMD(msg=0)) # solve the model, without printing output # to solve it using another solver: solve(GLPK(msg = 0))
        
        if (VERBOSE_RES in self.verbose):
            self.write_lp_res_line (self.res_output_file, model)
        sol_status = plp.LpStatus[model.status] 
        if (VERBOSE_LOG in self.verbose):            
            self.write_lp_res_line (self.log_output_file, model)
        if (model.status == 1): # successfully solved
            if (VERBOSE_LOG in self.verbose):            
                self.print_lp_sol_to_log ()
                printf (self.log_output_file, '\nSolved in {:.3f} [sec]\n' .format (time.time() - self.last_rt))
        else:
            printf (self.log_output_file, 'failed. status={}\n' .format(plp.LpStatus[model.status]))
            print  ('Running the LP failed. status={}' .format(plp.LpStatus[model.status]))
            exit ()

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
        self.log_file_name = "../res/" + self.ap_file_name.split(".")[0] + '.' + self.alg + ('.detailed' if VERBOSE_ADD_LOG in self.verbose else '') +'.log'  
        self.log_output_file =  open ('../res/' + self.log_file_name,  "w") 
        printf (self.log_output_file, '//RCs = augmented capacity of server s\n' )

    def print_lp_sol_to_log (self):
        """
        print a lp fractional solution to the output log file 
        """
        for s in self.G.nodes():
            printf (self.log_output_file, 's{} RCs={} used cpu={}\n' .format (s, self.G.nodes[s]['RCs'], self.lp_used_cpu_in (s) ))

        if (VERBOSE_ADD_LOG in self.verbose): 
            for d_var in self.d_vars: 
                if d_var.plp_var.value() > 0:
                    printf (self.log_output_file, '\nu {} lvl {:.0f} s {:.0f} val {:.2f}' .format(
                           d_var.usr.id, d_var.lvl, d_var.s, d_var.plp_var.value()))            

    def print_sol_to_log (self):
        """
        print the solution found by alg' for the mig' problem to the output log file 
        """
        for s in self.G.nodes():
            used_cpu_in_s = self.used_cpu_in (s)
            chains_in_s   = [usr.id for usr in self.usrs if usr.nxt_s==s]
            if (used_cpu_in_s > 0): 
                printf (self.log_output_file, 's{} : Rcs={}, a={}, used cpu={:.0f}, Cs={}, num_of_chains={}' .format (
                        s,
                        self.G.nodes[s]['RCs'],
                        self.G.nodes[s]['a'],
                        used_cpu_in_s,
                        self.G.nodes[s]['cpu cap'],
                        len (chains_in_s),                       
                        ))
                if (VERBOSE_ADD_LOG in self.verbose and used_cpu_in_s > 0): 
                    printf (self.log_output_file, ' chains {}\n' .format (chains_in_s))
                else: 
                    printf (self.log_output_file, '\n')
        if (VERBOSE_DEBUG in self.verbose): 
            self.check_cpu_usage_all_srvrs () #Checks for all servers whether the allocated cpu + the available cpu = the total cpu. 
            
    def check_cpu_usage_all_srvrs (self):
        """
        Used for debug. Checks for all servers whether the allocated cpu + the available cpu = the total cpu.
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
                    self.G.nodes [usr.nxt_s]    ['a'] += usr.B[usr.lvl] # inc the available CPU at the previosly-suggested place for this usr  
                    self.G.nodes [usr.S_u[lvl]] ['a'] -= usr.B[lvl]     # dec the available CPU at the new  loc of the moved usr
                    
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
        self.CPU_cost_at_lvl   = [1 * (self.tree_height + 1 - lvl) for lvl in range (self.tree_height+1)]
        self.link_cost_at_lvl  = self.uniform_link_cost * np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of using a link from level i to level i+1, or vice versa.
        self.link_delay_at_lvl = 2 * np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of using a link from level i to level i+1, or vice versa.
        self.cpu_cap_at_lvl    = np.array ([30  * (lvl+1) for lvl in range (self.tree_height+1)], dtype='uint16') if self.ap_file_name == 'shorter.ap' else\
                                 np.array ([620 * (lvl+1) for lvl in range (self.tree_height+1)], dtype='uint16') # Lux city center 64 APs require 360*1.95=702
        
        # overall link cost and link capacity of a Single-Server Placement of a chain at each lvl
        self.link_cost_of_CLP_at_lvl  = [2 * sum([self.link_cost_at_lvl[i]  for i in range (lvl)]) for lvl in range (self.tree_height+1)]
        self.link_delay_of_CLP_at_lvl = [2 * sum([self.link_delay_at_lvl[i] for i in range (lvl)]) for lvl in range (self.tree_height+1)] 
        
        self.G = self.G.to_directed()

        shortest_path = nx.shortest_path(self.G)

        # levelize the tree (assuming a balanced tree)
        self.ap2s             = [] # Will contain a least translating the AP number (==leaf #) to the ID of the co-located server.
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

    def __init__ (self, ap_file_name = 'shorter.ap', verbose = [], tree_height = 3, children_per_node = 4, run_to_calc_rsrc_aug=False):
        """
        """
        
        # verbose and debug      
        self.verbose                    = verbose
        
        # Network parameters
        self.tree_height                = tree_height
        self.children_per_node          = children_per_node # num of children of every non-leaf node
        self.run_to_calc_rsrc_aug       = run_to_calc_rsrc_aug
        self.uniform_mig_cost           = 1
        self.Lmax                       = 0
        self.uniform_Tpd                = 2
        self.uniform_link_cost          = 1
        self.uniform_theta_times_lambda = [2, 10, 2] # "1" here means 100MHz 
        self.uniform_Cu                 = 20 
        self.uniform_target_delay       = 20 #[ms]
        self.warned_about_too_large_ap  = False
        self.ap_file_name               = ap_file_name #input file containing the APs of all users along the simulation
        self.usrs                       = []
        self.max_R                      = 4 # maximal rsrc augmenation to consider
        
        # Init output files
        if (VERBOSE_RES in self.verbose):
            self.init_res_file()
        if (VERBOSE_COST_COMP in self.verbose):
            self.init_cost_comp () 
        if (VERBOSE_DEBUG in self.verbose):
            self.debug_file = open ('../res/debug.txt', 'w') 
        if (VERBOSE_MOB in self.verbose):
            self.num_of_moves_in_cycle = [] # self.num_of_moves_in_cycle[t] will hold the num of usrs who moved at cycle t.   
            self.num_of_migs_in_cycle  = [] # self.num_of_migs[t] will hold the num of chains that the alg' migrated in cycle t.
            self.mig_from_to_lvl      = np.zeros ([self.tree_height+1, self.tree_height+1], dtype='int') # self.mig_from_to_lvl[i][j] will hold the num of migrations from server in lvl i to server in lvl j, along the sim

        self.gen_parameterized_tree  ()
        self.delay_const_sanity_check()

    def init_cost_comp (self):
        """
        Open the output file to which we will write the cost of each component in the sim
        """
        self.cost_comp_file_name = "../res/" + self.ap_file_name.split(".")[0] + ".cost_comp.res"  
        self.cost_comp_output_file =  open ('../res/' + self.cost_comp_file_name,  "w") 
        
        self.total_cpu_cost_in_slot  = []
        self.total_link_cost_in_slot = []
        self.total_mig_cost_in_slot  = []
        self.total_cost_in_slot      = []

    def delay_const_sanity_check (self):
        """
        Sanity check for the usr parameters' feasibility.
        """
        usr = usr_c (id=0, theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.uniform_target_delay, C_u=self.uniform_Cu)
        self.CPUAll_single_usr (usr) 
        if (len(usr.B)==0):
            print ('Error: cannot satisfy delay constraints of usr {}, even on a leaf. theta_times_lambda={}, target_delay ={}' .format (
                    usr.id, usr.theta_times_lambda, usr.target_delay))
            exit ()
        if (self.link_delay_of_CLP_at_lvl[-1] > self.uniform_target_delay):
            print ('**** Warning: the network delay at the root > target delay. Hence, no user can use the root server. ****')
            if (self.ap_file_name not in ['shorter.ap', 'short_0.ap', 'short_1.ap']):
                exit ()     

    def simulate (self, alg, final_cycle=99999):
        """
        Simulate the whole simulation using the chosen alg: LP, or ALG_TOP (our alg).
        """
        self.alg         = alg
        self.final_cycle = final_cycle
        self.is_first_t = True # Will indicate that this is the first simulated time slot
        if (VERBOSE_LOG in self.verbose):
            self.init_log_file()
        if (VERBOSE_MOB in self.verbose):
            self.mob_file_name   = "../res/" + self.ap_file_name.split(".")[0] + '.' + self.alg.split("_")[1] + '.mob.log'  
            self.mob_output_file =  open ('../res/' + self.mob_file_name,  "w") 
            printf (self.mob_output_file, '// results for running alg alg_top on input file {}\n' .format (self.ap_file_name))
            printf (self.mob_output_file, '// results for running alg alg_top on input file shorter.ap with {} leaves\n' .format (self.num_of_leaves))
            printf (self.mob_output_file, '// index i,j in the matrices below represent the total num of migs from lvl i to lvl j\n')
             
        print ('Simulating {}. num of leaves = {}. ap file = {} ' .format (self.alg, self.num_of_leaves, self.ap_file_name))
        self.stts     = sccs
        self.rsrc_aug = 1
        self.set_augmented_cpu_in_all_srvrs ()
        print ('rsrc aug = {}' .format (self.rsrc_aug))
        if (self.alg in ['our_alg', 'wfit', 'ffit']):
            self.simulate_algs()
        elif (self.alg == 'alg_lp'):
            self.simulate_lp ();
        else:
            print ('Sorry, alg {} that you selected is not supported' .format (self.alg))
            exit ()

    def set_augmented_cpu_in_all_srvrs (self):
        """
        Set the capacity in each server to its cpu capacity, time the resource augmentation. 
        """
        for s in self.G.nodes():
            self.G.nodes[s]['RCs'] = self.rsrc_aug * self.G.nodes[s]['cpu cap'] # for now, assume no resource aug' 

    
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
                self.rd_line_t (splitted_line[2])
                continue
        
            elif (splitted_line[0] == "usrs_that_left:"):
        
                for usr in list (filter (lambda usr : usr.id in [int(usr_id) for usr_id in splitted_line[1:] if usr_id!=''], self.usrs)):
                    self.usrs.remove  (usr)                    
                continue
        
            elif (splitted_line[0] == "new_usrs:"):              
                self.rd_new_usrs_line_lp (splitted_line[1:])
            elif (splitted_line[0] == "old_usrs:"):              
                self.rd_old_usrs_line_lp (splitted_line[1:])
                self.set_last_time()
                self.solve_by_plp () 
                self.cur_st_params = self.d_vars
                    
    def rd_line_t (self, time_str):
        """
        read the line describing a new time slot in the input file. Init some variables accordingly.
        """ 
        self.t = int(time_str)
        if (self.is_first_t):
            self.init_t     = self.t
            self.is_first_t = False
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_output_file, '\ntime = {}\n**************************************\n' .format (self.t))
        if (self.alg in ['our_alg', 'wfit', 'ffit'] and (self.t % 100 == 1)): # once in a while, reshuffle the random ids of usrs, to mitigate unfairness due to tie-breaking by the ID, when sorting usrs 
            for usr in self.usrs:
                usr.calc_rand_id ()
                    
    def simulate_algs (self):
        """
        Simulate the whole simulation, using an algorithm (rather than a LP solver).
        At each time step:
        - Read and parse from an input ".ap" file the AP cells of each user who moved. 
        - solve the problem using alg_top (our alg). 
        - Write outputs results and/or logs to files.
        - update cur_st = nxt_st
        """
        
        # reset Hs and RCs       
        for s in self.G.nodes():
            self.G.nodes[s]['RCs'] = self.G.nodes[s]['cpu cap'] # Initially, no rsrc aug --> at each server, we've exactly his non-augmented capacity. 
            if (self.alg in ['our_alg']):
                self.G.nodes[s]['Hs']  = set() 
        
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
                self.rd_line_t (splitted_line[2])
                if (self.t > self.final_cycle):
                    self.post_processing ()
                    exit ()
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
                if (VERBOSE_LOG in self.verbose):
                    self.set_last_time()
                    printf (self.log_output_file, 't={}. beginning alg top\n' .format (self.t))
                    
                # solve the prob' using the requested alg'    
                if   (self.alg == 'our_alg'):
                    self.stts = self.alg_top(self.bottom_up)
                elif (self.alg == 'ffit'):
                    self.stts = self.alg_top(self.first_fit)
                elif (self.alg == 'wfit'):
                    self.stts = self.alg_top(self.worst_fit)
                else:
                    print ('Sorry, alg {} that you selected is not supported' .format (self.alg))
                    exit ()
        
                # By hook or by crook, now we have a feasible solution     
                if (self.stts == sccs and self.alg == 'our_alg'):  
                    self.push_up ()
                    if (VERBOSE_LOG in self.verbose):
                        printf (self.log_output_file, 'after push-up\n')
                
                self.print_sol_to_res_and_log ()
                if (self.stts==fail):
                    self.rst_sol()
                
                for usr in self.usrs: # The solution found at this time slot is the "cur_state" for next slot
                     usr.cur_s = usr.nxt_s
        
        self.post_processing()
    
    def post_processing (self):
        """
        Organize, writes and plots the simulation results, after the simulation is done
        """
        if (VERBOSE_MOB in self.verbose):
            self.print_mob ()        
        if (VERBOSE_COST_COMP in self.verbose):
            self.print_sol_cost_components ()
    
    def print_mob (self):
        """
        print statistics about the number of usrs who moved, and the num of migrations between every two levels in the tree.
        """

        sim_len = float(self.t - self.init_t)
        del (self.num_of_migs_in_cycle[0]) # remove the mig' recorded in the first cycle, which is irrelevant (corner case)
        printf (self.mob_output_file, '// avg num of usrs that moved per slot = {:.0f}\n'   .format (float(sum(self.num_of_moves_in_cycle)) / sim_len))
        printf (self.mob_output_file, '// avg num of usrs who migrated per slot = {:.0f}\n' .format (float(sum(self.num_of_migs_in_cycle)) / sim_len))
        avg_num_of_migs_to_from_per_slot = np.divide (self.mig_from_to_lvl, sim_len)
        for lvl_src in range (self.tree_height+1):
            for lvl_dst in range (self.tree_height+1):
                printf (self.mob_output_file, '{:.0f}\t' .format (avg_num_of_migs_to_from_per_slot[lvl_src][lvl_dst]))
            printf (self.mob_output_file, '\n')
        printf (self.mob_output_file, 'moves_in_slot = {}\n' .format (self.num_of_moves_in_cycle))
        printf (self.mob_output_file, 'migs_in_slot = {}\n'  .format (self.num_of_migs_in_cycle))

        # plot the mobility
        plt.figure()
        plt.title ('Migrations and mobility at each cycle')
        plt.plot (range(int(sim_len)), self.num_of_moves_in_cycle, label='Total vehicles moved to another cell [number/sec]', linestyle='None',  marker='o', markersize = 4)
        plt.plot (range(int(sim_len)), self.num_of_migs_in_cycle, label='Total chains migrated to another server [number/sec]', linestyle='None',  marker='.', markersize = 4)
        plt.xlabel ('time [seconds, starting at 07:30]')
        plt.legend()
        plt.savefig ('../res/{}.mob.jpg' .format(self.ap_file_name.split('.')[0]))
        plt.clf()
    
    def first_fit_reshuffle (self):
        """
        Run the first-fit alg' when considering all existing usrs in the system (not only critical usrs).
        Returns sccs if found a feasible placement, fail otherwise
        """
        for usr in sorted (self.usrs, key = lambda usr : usr.rand_id):
            if ( not (self.first_fit_place_usr (usr))):
                return fail
        return sccs
    
    def first_fit (self):
        """
        Run the worst-fit alg'.
        Returns sccs if found a feasible placement, fail otherwise
        """

        for usr in sorted (self.critical_usrs(), key = lambda usr : usr.rand_id):
            if ( not (self.first_fit_place_usr (usr))): # failed to find a feasible sol' when considering only the critical usrs
                self.rst_sol()
                return self.first_fit_reshuffle() # try again, by reshuffling the whole usrs' placements
        return sccs
    
    def first_fit_place_usr (self, usr):
        """
        places a usr on the first server that fits it (that is, enough capacity for it), on the downward path of its delay-feasible servers.
        If failed to place the usr, returns fail. Else, returns sccs.
        """
        
        for s in reversed(usr.S_u):
            if (self.s_has_sufic_avail_cpu_for_usr (s, usr)): # if the available cpu at this server > the required cpu for this usr at this lvl...
                usr.nxt_s = s
                usr.lvl   = self.G.nodes[s]['lvl'] 
                return sccs
        return fail
    
    def worst_fit_reshuffle (self):
        """
        Run the worst-fit alg' when considering all existing usrs in the system (not only critical usrs).
        Returns sccs if found a feasible placement, fail otherwise
        """
        for usr in sorted (self.usrs, key = lambda usr : (usr.B[-1], usr.rand_id)):
            if ( not (self.worst_fit_place_usr (usr))): 
                return fail
        return sccs
    
    def worst_fit (self):
        """
        Run the worst-fit alg'.
        Returns sccs if found a feasible placement, fail otherwise
        """
        critical_usrs = self.critical_usrs () 
        
        # first, handle the old, existing usrs, in an increasing order of the available cpu on the currently-hosting server
        for usr in sorted (list (filter (lambda usr : usr.cur_s!=-1 and usr.nxt_s==-1, critical_usrs)), 
                           key = lambda usr : (self.G.nodes[usr.cur_s]['a'], usr.rand_id)): 
            if (not(self.worst_fit_place_usr (usr))) : # Failed to migrate this usr)):
                self.rst_sol()
                return self.worst_fit_reshuffle()
                        
        # next, handle the new usrs, namely, that are not currently hosted on any server
        for usr in list (filter (lambda usr : usr.cur_s==-1 and usr.nxt_s==-1, critical_usrs)): 
            if (not(self.worst_fit_place_usr (usr))) : # Failed to migrate this usr)):
                self.rst_sol()
                return self.worst_fit_reshuffle()

        return sccs

    def write_fail_to_log_n_res (self):
        """
        Write to the log and to the res file that the alg' currently running did not succeed to place all the usrs
        """
        if (VERBOSE_RES in self.verbose):
            printf (self.res_output_file, 't{:.0f}.{}.stts2' .format (self.t, self.alg))
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_output_file, 't{:.0f}.{}.stts2' .format (self.t, self.alg))

    def worst_fit_place_usr (self, usr):
        """
        Try to place the given usr on a server, chosen in a worst-fit manner.
        Returns True if successfully placed the usr.
        """
        delay_feasible_servers = sorted (usr.S_u, key = lambda s : self.G.nodes[s]['a'], reverse=True) # sort the delay-feasible servers in a dec' order of available resources (worst-fit approach)
        for s in delay_feasible_servers: # for every delay-feasible server 
            if (self.s_has_sufic_avail_cpu_for_usr (s, usr)): # if the available cpu at this server > the required cpu for this usr at this lvl...
                mig_dst   = self.G.nodes[s]['id'] # id of the migration's destination
                usr.nxt_s = mig_dst # mark this server as this usr's place in the next slot
                usr.lvl   = self.G.nodes[s]['lvl']
                self.G.nodes[s]['a'] -= usr.B[self.G.nodes[mig_dst]['lvl']] # dec' the available cpu at the dest' accordingly. If this is an old user, the resources it used in the current location were already released by rd_old_usrs_line ()
                return True
        return False  
    
    def alg_top (self, placement_alg):
        """
        Binary search for a feasible sol that minimizes the resource augmentation R.
        The search is done by calling bottom_up ().
        """
        
        # Try to solve the problem by changing the placement or CPU allocation only for the new / moved users
        if (placement_alg() == sccs):
            return sccs

        # Couldn't solve the problem without additional rsrc aug --> begin a binary search for the amount of rsrc aug' needed.
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_output_file, 'By binary search:\n')

        self.rst_sol() # dis-allocate all users
        self.CPUAll(self.usrs) 
        max_R = self.max_R if self.run_to_calc_rsrc_aug else self.calc_upr_bnd_rsrc_aug ()   
        
        # init cur RCs and a(s) to the number of available CPU in each server, assuming maximal rsrc aug' 
        for s in self.G.nodes(): 
            self.G.nodes[s]['RCs'] = math.ceil (max_R * self.G.nodes[s]['cpu cap']) 
            self.G.nodes[s]['a']   = self.G.nodes[s]['RCs'] #currently-available rsrcs at server s  

        if (placement_alg() != sccs):
            print ('did not find a feasible sol even with maximal rsrc aug')
            exit ()

        # Now we know that we found an initial feasible sol 
                   
        ub = np.array([                self.G.nodes[s]['RCs']     for s in self.G.nodes()]) # upper-bnd on the (augmented) cpu cap' that may be required
        lb = np.array([self.rsrc_aug * self.G.nodes[s]['cpu cap'] for s in self.G.nodes()]) # lower-bnd on the (augmented) cpu cap' that may be required
        
        while True: 
             
            if ( np.array([ub[s] <= lb[s]+1 for s in self.G.nodes()], dtype='bool').all()): # Did the binary search converged?
                for s in self.G.nodes(): # Yep, so allocate this minimal found amount of rsrc aug to all servers
                    self.G.nodes[s]['RCs'] = math.ceil (ub[s])  
                self.rst_sol()         # and re-solve the prob'
                if (placement_alg() == sccs): 
                    self.update_rsrc_aug () # update the rsrc augmnetation to the lvl used in practice by the fesible sol found by this binary search
                    return sccs
                
                # We've got a prob', Houston
                print ('Error in the binary search: though I found a feasible sol, but actually this sol is not feasible')
                exit ()
                
            # Now we know that the binary search haven't converged yet
            # Update the available capacity at each server according to the value of resource augmentation for this iteration            
            for s in self.G.nodes():
                self.G.nodes[s]['RCs'] = math.floor (0.5*(ub[s] + lb[s]))  
            self.rst_sol()

            # Solve using the given placement alg'
            if (placement_alg() == sccs):
                if (VERBOSE_ADD_LOG in self.verbose): 
                    printf (self.log_output_file, 'In bottom-up IF\n')
                    self.print_sol_to_log()
                    if (VERBOSE_DEBUG in self.verbose):
                        self.check_cpu_usage_all_srvrs()
                ub = np.array([self.G.nodes[s]['RCs'] for s in self.G.nodes()])        
            else:
                lb = np.array([self.G.nodes[s]['RCs'] for s in self.G.nodes()])
    
    def bottom_up (self):
        """
        Our bottom-up alg'. 
        Assigns all self.usrs that weren't assigned yet (either new usrs, or old usrs that moved, and now they don't satisfy the target delay).
        Looks for a feasible sol'.
        Returns sccs if a feasible sol was found, fail else.
        """        
        for s in range (len (self.G.nodes())-1, -1, -1): # for each server s, in an increasing order of levels (DFS).v
            lvl = self.G.nodes[s]['lvl']
            Hs = [usr for usr in self.G.nodes[s]['Hs'] if (usr.lvl == -1)] # usr.lvl==-1 verifies that this usr wasn't placed yet
            for usr in sorted (Hs, key = lambda usr : (len(usr.B), usr.rand_id)): # for each chain in Hs, in an increasing order of level ('L')
                if (self.G.nodes[s]['a'] > usr.B[lvl]):
                    usr.nxt_s = s
                    usr.lvl   = lvl
                    self.G.nodes[s]['a'] -= usr.B[lvl]
                elif (len (usr.B)-1 == lvl):
                    return fail
        return sccs

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
    
            usr = usr_lp_c (id = int(tuple[0]), theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.uniform_target_delay, C_u=self.uniform_Cu) # generate a new usr, which is assigned as "un-placed" yet (usr.lvl==-1)
            self.CPUAll_single_usr (usr)
            self.usrs.append (usr)
            AP_id = int(tuple[1])
            # self.check_AP_id (AP_id)
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
            tuple = tuple[1].split (',')
            
            usr = usr_c (id                 = int(tuple[0]), # generate a new usr, which is assigned as "un-placed" yet (usr.lvl==-1) 
                         theta_times_lambda = self.uniform_theta_times_lambda,
                         target_delay       = self.uniform_target_delay,
                         C_u                = self.uniform_Cu)
            self.CPUAll_single_usr (usr)
            self.usrs.append (usr)
            AP_id = int(tuple[1])
            # self.check_AP_id (AP_id)
            s = self.ap2s[AP_id]
            usr.S_u.append (s)
            for lvl in (range (len(usr.B)-1)):
                s = self.parent_of(s)
                usr.S_u.append (s)
            
            # Hs is the list of chains that may be located on each server while satisfying the delay constraint. Only some of the algs' use it
            if (self.alg in ['our_alg']):
                for s in usr.S_u:
                    self.G.nodes[s]['Hs'].add(usr)                       
                    
    def rd_old_usrs_line (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file.
        The input includes a list of tuples of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the tuples, the function assigns each chain to its relevant list of chains, Hs.  
        """
        if (line == []): # if the list of old users that moved is empty
            return
        
        splitted_line = self.parse_old_usrs_line(line)
        if (VERBOSE_MOB in self.verbose and self.t > self.init_t):
            self.num_of_moves_in_cycle.append (len (splitted_line)) # record the num of usrs who moved at this cycle  

        for tuple in splitted_line:
            if (len(tuple) <= 1):
                break
            tuple = tuple.split("(")
            tuple = tuple[1].split (',')
            
            list_of_usr = list(filter (lambda usr : usr.id == int(tuple[0]), self.usrs))
            if (len(list_of_usr) == 0):
                print  ('Error at t={}: input file={}. Did not find old / recycled usr {}' .format (self.t, self.ap_file_name, tuple[0]))
                exit ()
            usr = list_of_usr[0]
            usr.cur_cpu = usr.B[usr.lvl]
            AP_id       = int(tuple[1])
            # self.check_AP_id (AP_id)
            self.CPUAll_single_usr (usr) # update usr.B by the new requirements of this usr.

            # Add this usr to the Hs of every server to which it belongs at its new location
            s       = self.ap2s[AP_id]
            usr.S_u = [s]
            for lvl in (range (len(usr.B)-1)):
                s = self.parent_of(s)
                usr.S_u.append (s)
    
            # If this alg ues 'Hs', we have to update it, so that for each server, 
            # Hs of a server includes a usr only if s is delay-feasible for this usr also after the usr moved. 
            if (self.alg in ['our_alg'] and usr.cur_s in usr.S_u and usr.cur_cpu <= usr.B[usr.lvl]): # Can satisfy delay constraint while staying in the cur location and keeping the CPU budget
                
                for s in [s for s in self.G.nodes() if usr in self.G.nodes[s]['Hs']]:
                    self.G.nodes[s]['Hs'].remove (usr) 
                for s in usr.S_u:
                    self.G.nodes[s]['Hs'].add(usr)                               
                continue
            # Now we know that this is a critical usr, namely a user that needs more CPU and/or migration for satisfying its target delay constraint 
            # dis-place this user (mark it as having nor assigned level, neither assigned server), and free its assigned CPU

            self.rmv_usr_rsrcs (usr) # Free the resources of this user in its current place            
            usr.lvl   = -1
            usr.nxt_s = -1

            # if the currently-run alg' uses 'Hs', Add the usr to the relevant 'Hs'.
            # Hs is the set of relevant usrs) at each of its delay-feasible server
            if (self.alg in ['our_alg']):
                for s in usr.S_u:
                    self.G.nodes[s]['Hs'].add(usr)                               

    def rd_old_usrs_line_lp (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file, when using a LP solver for the problem.
        The input includes a list of tuples of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the tuples, the function assigns each chain to its relevant list of chains, Hs.  
        """
        if (line == []): # if the list of old users that moved is empty
            return
        
        splitted_line = self.parse_old_usrs_line(line)

        for tuple in splitted_line:
            if (len(tuple) <= 1):
                break
            tuple = tuple.split("(")
            tuple = tuple[1].split (',')
            
            list_of_usr = list(filter (lambda usr : usr.id == int(tuple[0]), self.usrs))
            if (len(list_of_usr) == 0):
                print  ('Error at t={}: input file={}. Did not find old / recycled usr {}' .format (self.t, self.ap_file_name, tuple[0]))
                exit ()
            usr    = list_of_usr[0]
            AP_id  = int(tuple[1])
            # self.check_AP_id (AP_id)
            self.CPUAll_single_usr (usr)

            # Add this usr to the Hs of every server to which it belongs at its new location
            s       = self.ap2s[AP_id]
            usr.S_u = [s]
            for lvl in (range (len(usr.B)-1)):
                s = self.parent_of(s)
                usr.S_u.append (s)
    
    def check_AP_id (self, AP_id):
        if (AP_id >= self.num_of_leaves):
            AP_id = self.num_of_leaves-1
        if (self.warned_about_too_large_ap == False):
            print ('Encountered AP num {} in the input file, but in the tree there are only {} leaves. Changing the ap to {}' .format (AP_id, self.num_of_leaves, self.num_of_leaves-1))
            exit ()
    
    def rmv_usr_rsrcs (self, usr):
        """
        Remove a usr from the Hs (relevant usrs) of every server to which it belonged, at its previous location; 
        and increase the avilable rsrcs of the srvr that currently place this usr
        """
        self.G.nodes[usr.cur_s]['a'] += usr.B[usr.lvl] # free the CPU units used by the user in the old location
        if (self.alg not in ['our_alg']):
            return 
        
        # Now we know that the alg' that currently runs uses 'Hs'. Hence, we have to clean them.
        for s in [s for s in self.G.nodes() if usr in self.G.nodes[s]['Hs']]:
            self.G.nodes[s]['Hs'].remove (usr) 
            
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

    ap_file_name = 'shorter.ap' #'vehicles_n_speed_0730.ap'
    my_simulator = SFC_mig_simulator (ap_file_name          = ap_file_name, 
                                      verbose               = [VERBOSE_RES], # defines which sanity checks are done during the simulation, and which outputs will be written   
                                      tree_height           = 2 if ap_file_name=='shorter.ap' else 3, 
                                      children_per_node     = 2 if ap_file_name=='shorter.ap' else 4,
                                      run_to_calc_rsrc_aug  = True # When true, this run will binary-search the minimal resource aug. needed to find a feasible sol. for the prob'  
                                      )
    my_simulator.simulate (alg='ffit', final_cycle = 27060)# 
                           
                           # (alg          - 'ffit', #algorithm to simulate 
                           # final_cycle  = 27060,     # last time slot to run. Currently the first slot is 27000 (07:30).
                           # )

    # moves_in_slot = [74, 59, 55, 42, 62, 60, 57, 54, 53, 63, 56, 59, 58, 52, 54, 59, 51, 52, 54, 55, 49, 49, 53, 68, 43, 57, 37, 71, 48, 61, 47, 52, 52, 61, 55, 57, 52, 56, 44, 49, 46, 55, 47, 56, 48, 50, 64, 50, 75, 55, 53, 55, 51, 60, 41, 62, 61, 46, 51, 44, 55, 45, 57, 60, 56, 55, 73, 62, 58, 65, 62, 52, 49, 54, 53, 51, 45, 68, 43, 61, 58, 67, 52, 56, 51, 54, 62, 68, 52, 56, 57, 60, 59, 67, 48, 68, 44, 62, 54, 55, 60, 53, 51, 58, 60, 59, 58, 52, 67, 47, 50, 49, 52, 54, 66, 50, 51, 53, 63, 49, 51, 48, 51, 47, 57, 49, 62, 43, 63, 56, 55, 54, 56, 65, 51, 53, 69, 74, 48, 58, 66, 44, 52, 51, 64, 52, 48, 76, 53, 71, 55, 59, 49, 52, 63, 52, 66, 55, 54, 54, 55, 64, 56, 79, 43, 62, 79, 54, 74, 68, 48, 68, 57, 55, 68, 47, 50, 50, 67, 59, 65, 60, 62, 56, 57, 59, 51, 47, 50, 66, 55, 71, 64, 44, 62, 63, 60, 64, 57, 66, 51, 64, 47, 57, 51, 69, 60, 53, 54, 56, 71, 49, 51, 46, 45, 53, 53, 49, 45, 63, 58, 62, 52, 56, 64, 64, 64, 71, 59, 87, 74, 60, 62, 58, 73, 60, 60, 63, 56, 62, 65, 62, 67, 54, 65, 51, 55, 69, 50, 64, 69, 57, 67, 50, 60, 61, 67, 54, 56, 52, 51, 62, 59, 44, 60, 43, 48, 57, 63, 45, 67, 45, 52, 55, 65, 54, 64, 59, 68, 72, 59, 77, 55, 60, 64, 66, 59, 54, 61, 68, 58, 50, 48, 63, 57, 57, 51, 74, 62, 52, 59, 58, 55, 71, 48, 63, 75, 65, 52, 74, 71, 66, 59, 67, 56, 64, 61, 70, 72, 56, 60, 59, 47, 64, 58, 46, 48, 57, 54, 68, 61, 58, 68, 71, 48, 69, 57, 64, 64, 59, 56, 58, 58, 61, 56, 63, 60, 53, 54, 56, 42, 53, 59, 47, 55, 59, 57, 52, 48, 60, 50, 74, 53, 63, 55, 54, 70, 58, 61, 51, 59, 56, 67, 65, 58, 57, 50, 58, 53, 62, 49, 48, 43, 55, 43, 45, 54, 56, 65, 67, 57, 70, 55, 67, 65, 58, 62, 74, 64, 66, 56, 69, 71, 68, 57, 54, 55, 59, 49, 56, 70, 67, 66, 64, 63, 59, 65, 62, 68, 60, 68, 52, 65, 42, 64, 62, 66, 77, 61, 62, 59, 58, 58, 73, 48, 50, 52, 48, 54, 42, 42, 66, 57, 58, 65, 52, 49, 51, 49, 43, 60, 43, 52, 45, 47, 50, 53, 50, 64, 59, 51, 63, 63, 58, 72, 65, 66, 53, 55, 46, 58, 56, 47, 52, 58, 63, 54, 59, 71, 60, 65, 54, 59, 57, 62, 60, 62, 50, 68, 55, 59, 66, 57, 56, 51, 52, 57, 74, 61, 54, 42, 52, 68, 51, 52, 55, 50, 49, 70, 58, 70, 59, 54, 49, 58, 61, 50, 64, 59, 68, 62, 58, 53, 56, 55, 63, 49, 55, 58, 57, 60, 52, 66, 54, 52, 41, 55, 59, 48, 52, 50, 40, 53, 53, 48, 57, 51, 59, 48, 58, 45, 53, 57, 51, 54, 56, 82, 52, 65, 67, 62, 69, 57, 63, 63, 63, 68, 52, 53, 62, 52, 50, 58, 56, 45, 58, 78, 63, 45, 50, 58, 60, 63, 55, 55, 56, 60, 57, 60, 67, 60, 62, 54, 65, 54, 62, 50, 47, 65, 64, 45, 61, 56, 48, 58, 68, 59, 52, 71, 55, 62, 57, 53, 60, 66, 52, 50, 52, 47, 53, 49, 60, 61, 63, 53, 62, 46, 60, 54, 66, 54, 59, 66, 64, 56, 51, 68, 56, 58, 71, 56, 61, 57, 56, 47, 62, 61, 49, 61, 61, 52, 57, 51, 55, 57, 51, 61, 55, 63, 65, 46, 61, 58, 51, 56, 44, 58, 48, 64, 66, 57, 59, 58, 48, 54, 61, 44, 54, 58, 64, 50, 46, 61, 61, 45, 51, 50, 57, 63, 53, 41, 62, 58, 50, 56, 43, 49, 53, 42, 54, 64, 65, 47, 70, 56, 51, 58, 60, 57, 54, 51, 55, 53, 54, 58, 62, 53, 59, 53, 57, 56, 72, 59, 65, 61, 62, 63, 60, 46, 70, 44, 59, 57, 53, 49, 57, 61, 51, 61, 55, 53, 68, 55, 53, 52, 54, 58, 57, 62, 41, 81, 51, 70, 45, 59, 59, 52, 58, 54, 51, 48, 56, 52, 51, 54, 50, 61, 60, 57, 59, 69, 64, 50, 61, 55, 72, 57, 58, 54, 48, 59, 51, 59, 60, 46, 51, 52, 53, 38, 61, 55, 49, 55, 53, 39, 59, 51, 61, 54, 57, 60, 66, 63, 58, 51, 46, 61, 47, 58, 67, 72, 67, 62, 59, 61, 55, 68, 52, 63, 61, 66, 55, 46, 60, 67, 51, 56, 40, 50, 64, 44, 48, 57, 56, 43, 50, 58, 66, 46, 58, 51, 57, 44, 60, 54, 59, 43, 61, 54, 55, 61, 53, 68, 55, 61, 62, 47, 59, 66, 49, 51, 48, 57, 65, 55, 53, 62, 57, 69, 55, 52, 60, 45, 55, 66, 54, 60, 65, 58, 50, 56, 54, 55, 62, 69, 66, 57, 68, 55, 61, 51, 68, 60, 67, 53, 64, 62, 75, 66, 57, 62, 61, 78, 57, 63, 61, 64, 62, 54, 54, 64, 60, 72, 50, 52, 64, 47, 54, 47, 53, 44, 60, 56, 69, 59, 67, 46, 56, 57, 67, 57, 55, 54, 52, 68, 61, 64, 55, 49, 53, 65, 59, 56, 60, 59, 60, 55, 69, 61, 65, 74, 59, 61, 51, 56, 66, 67, 53, 60, 51, 58, 54, 64, 57, 70, 61, 66, 74, 53, 80, 55, 53, 58, 67, 55, 72, 59, 61, 44, 59, 46, 49, 74, 61, 58, 58, 63, 65, 65, 76, 59, 59, 69, 49, 67, 56, 69, 62, 55, 64, 54, 55, 57, 60, 59, 56, 52, 61, 66, 49, 50, 51, 50, 48, 58, 52, 57, 61, 56, 63, 62, 49, 53, 49, 52, 47, 53, 51, 63, 58, 66, 52, 58, 51, 66, 56, 58, 63, 57, 59, 48, 64, 54, 70, 65, 51, 53, 58, 46, 56, 52, 54, 62, 53, 58, 57, 55, 52, 64, 59, 61, 62, 64, 59, 61, 73, 62, 60, 57, 64, 48, 63, 55, 56, 64, 52, 58, 55, 42, 44, 57, 51, 45, 61, 53, 58, 69, 54, 53, 46, 67, 51, 65, 45, 55, 59, 44, 62, 52, 76, 64, 59, 59, 69, 70, 71, 54, 45, 60, 57, 59, 53, 51, 60, 56, 61, 51, 64, 57, 56, 56, 60, 61, 46, 53, 57, 61, 58, 65, 72, 63, 60, 74, 61, 68, 71, 66, 65, 67, 59, 53, 63, 62, 66, 68, 52, 51, 65, 46, 67, 59, 70, 57, 66, 52, 66, 50, 65, 56, 67, 55, 59, 56, 74, 54, 49, 53, 51, 62, 61, 57, 53, 63, 55, 66, 61, 44, 44, 63, 57, 61, 61, 67, 49, 58, 55, 58, 50, 62, 60, 54, 60, 54, 56, 72, 60, 59, 61, 69, 50, 57, 67, 53, 57, 49, 59, 67, 45, 65, 53, 62, 50, 59, 58, 55, 68, 66, 60, 63, 68, 57, 64, 70, 60, 62, 70, 53, 60, 69, 49, 67, 59, 56, 60, 57, 60, 65, 66, 53, 56, 55, 69, 53, 63, 58, 56, 74, 61, 48, 63, 47, 51, 55, 57, 60, 51, 51, 50, 67, 48, 59, 55, 57, 63, 55, 57, 63, 57, 54, 45, 56, 61, 64, 53, 56, 57, 63, 48, 65, 62, 42, 72, 54, 52, 65, 51, 62, 59, 75, 58, 69, 66, 64, 63, 69, 70, 62, 49, 54, 63, 63, 59, 58, 69, 50, 68, 59, 69, 62, 48, 65, 70, 50, 66, 54, 64, 54, 51, 55, 49, 64, 61, 58, 58, 50, 53, 61, 46, 66, 58, 59, 55, 55, 69, 52, 61, 60, 58, 49, 58, 72, 62, 52, 67, 59, 59, 55, 74, 46, 59, 69, 47, 49, 64, 55, 59, 50, 44, 62, 59, 54, 43, 58, 51, 53, 61, 46, 43, 70, 61, 62, 61, 59, 54, 64, 62, 64, 65, 69, 68, 61, 58, 55, 62, 58, 68, 60, 55, 59, 50, 59, 60, 54, 66, 65, 64, 61, 51, 62, 61, 71, 61, 63, 58, 60, 58, 66, 46, 58, 63, 63, 60, 49, 52, 58, 44, 53, 53, 60, 69, 67, 53, 63, 62, 57, 65, 52, 65, 54, 53, 64, 63, 65, 56, 55, 49, 52, 60, 61, 65, 60, 67, 54, 72, 53, 59, 61, 54, 62, 53, 65, 46, 48, 63, 55, 49, 53, 57, 58, 55, 59, 74, 60, 64, 69, 65, 78, 61, 60, 76, 55, 67, 62, 62, 65, 64, 64, 53, 72, 58, 47, 56, 61, 61, 60, 65, 70, 52, 71, 61, 63, 64, 64, 64, 69, 55, 67, 51, 69, 55, 63, 48, 53, 58, 47, 55, 55, 62, 61, 64, 58, 72, 66, 55, 66, 61, 62, 67, 54, 62, 61, 66, 54, 65, 52, 61, 61, 56, 63, 67, 61, 58, 62, 44, 64, 56, 61, 78, 66, 60, 70, 62, 58, 69, 80, 64, 64, 60, 67, 64, 71, 61, 65, 61, 67, 56, 65, 56, 62, 67, 57, 56, 65, 62, 56, 57, 59, 63, 63, 49, 58, 69, 62, 69, 58, 61, 57, 54, 65, 57, 64, 58, 62, 61, 50, 55, 55, 52, 57, 60, 59, 76, 70, 65, 67, 61, 50, 46, 53, 64, 51, 64, 62, 56, 61, 46, 54, 63, 60, 56, 50, 61, 58, 57, 65, 71, 58, 68, 60, 66, 54, 63, 74, 77, 79, 65, 71, 65, 73, 58, 62, 47, 59, 66, 66, 54, 57, 59, 74, 50, 66, 56, 66, 79, 55, 62, 67, 60, 54, 61, 71, 53, 52, 61, 55, 70, 44, 67, 67, 68, 53, 61, 72, 67, 67, 70, 66, 54, 63, 50, 70, 60, 63, 66, 78, 62, 60, 68, 57, 49, 67, 60, 60, 61, 67, 56, 64, 55, 73, 64, 60, 65, 63, 69, 56, 69, 54, 53, 52, 63, 56, 66, 62, 62, 59, 74, 73, 73, 73, 63, 66, 60, 73, 71, 63, 64, 60, 69, 65, 58, 66, 75, 66, 57, 62, 60, 67, 57, 53, 69, 68, 57, 66, 48, 66, 58, 59, 60, 53, 60, 60, 50, 65, 59, 61, 53, 35, 59, 59, 62, 57, 55, 64, 63, 61, 73, 57, 65, 48, 57, 73, 55, 68, 54, 56, 59, 54, 62, 45, 62, 58, 58, 62, 65, 46, 68, 64, 63, 52, 62, 71, 67, 72, 74, 60, 58, 65, 61, 51, 76, 51, 59, 56, 52, 56, 55, 69, 53, 60, 67, 72, 54, 52, 73, 64, 72, 58, 75, 58, 74, 65, 68, 57, 65, 55, 55, 59, 56, 47, 70, 64, 72, 50, 73, 57, 61, 56, 64, 56, 61, 62, 58, 50, 62, 59, 60, 67, 47, 67, 61, 61, 58, 69, 56, 54, 54, 50, 63, 55, 59, 52, 69, 56, 57, 61, 54, 56, 71, 63, 66, 72, 67, 61, 60, 53, 53, 67, 64, 71, 62, 66, 66, 48, 56, 61, 64, 64, 59, 53, 53, 75, 68, 51, 63, 75, 62, 52, 70, 64, 49, 68, 59, 61, 64, 50, 57, 48, 62, 56, 59, 49, 50, 62, 58, 48, 62, 54, 56, 58, 67, 61, 58, 60, 51, 41, 54, 58, 47, 52, 60, 52, 47, 52, 64, 55, 55, 43, 52, 57, 62, 60, 40, 62, 55, 68, 66, 65, 43, 59, 68, 51, 52, 58, 49, 53, 56, 55, 47, 65, 55, 55, 70, 53, 70, 66, 54, 57, 59, 67, 54, 65, 60, 52, 64, 55, 73, 58, 61, 67, 70, 56, 56, 61, 62, 58, 66, 56, 50, 62, 65, 55, 60, 67, 52, 45, 52, 47, 59, 51, 46, 50, 62, 53, 47, 51, 57, 33, 55, 50, 54, 62, 50, 61, 58, 66, 61, 74, 73, 60, 65, 73, 59, 61, 64, 67, 56, 53, 59, 56, 68, 49, 66, 57, 46, 66, 60, 65, 48, 65, 57, 53, 63, 61, 66, 59, 68, 57, 70, 49, 56, 53, 47, 57, 56, 62, 52, 69, 66, 69, 65, 62, 62, 63, 66, 68, 76, 78, 47, 70, 53, 65, 50, 67, 57, 52, 44, 58, 57, 62, 55, 51, 34, 59, 59, 50, 59, 55, 48, 58, 50, 57, 58, 51, 66, 70, 52, 50, 52, 68, 51, 58, 59, 67, 60, 63, 50, 52, 64, 63, 74, 54, 67, 43, 52, 65, 53, 52, 56, 56, 55, 59, 65, 59, 53, 61, 71, 63, 57, 65, 69, 64, 66, 75, 52, 66, 71, 50, 65, 62, 56, 68, 53, 55, 58, 71, 60, 58, 73, 64, 50, 65, 66, 59, 59, 62, 71, 58, 68, 59, 63, 61, 67, 47, 55, 56, 54, 67, 53, 62, 61, 63, 39, 50, 57, 49, 64, 62, 62, 51, 50, 53, 49, 62, 62, 68, 54, 52, 43, 47, 57, 49, 53, 61, 69, 66, 49, 63, 60, 59, 75, 55, 76, 64, 77, 55, 60, 58, 79, 57, 66, 75, 60, 62, 65, 53, 58, 70, 69, 77, 64, 60, 58, 70, 62, 58, 57, 58, 64, 59, 56, 60, 59, 61, 62, 52, 57, 62, 59, 59, 60, 54, 66, 66, 64, 64, 67, 50, 59, 60, 65, 46, 56, 67, 53, 61, 62, 70, 56, 72, 59, 66, 57, 56, 49, 50, 63, 68, 74, 63, 53, 60, 71, 72, 57, 70, 62, 54, 60, 66, 47, 60, 62, 53, 45, 63, 57, 61, 69, 46, 48, 62, 69, 62, 72, 67, 63, 63, 64, 57, 72, 56, 53, 79, 68, 66, 70, 63, 74, 63, 66, 65, 64, 47, 55, 64, 65, 59, 71, 53, 63, 69, 52, 69, 68, 59, 48, 62, 62, 65, 69, 52, 60, 61, 57, 62, 56, 64, 58, 61, 65, 60, 59, 56, 67, 44, 59, 48, 70, 58, 50, 52, 67, 68, 60, 64, 70, 59, 66, 55, 66, 56, 59, 57, 50, 69, 52, 61, 61, 62, 54, 51, 64, 67, 61, 67, 67, 55, 66, 66, 66, 48, 58, 59, 58, 76, 60, 70, 60, 65, 70, 53, 69, 67, 61, 67, 66, 62, 62, 71, 60, 56, 56, 44, 57, 57, 67, 56, 58, 64, 63, 57, 59, 63, 57, 67, 54, 63, 50, 70, 65, 54, 73, 57, 56, 60, 57, 55, 53, 61, 54, 56, 55, 54, 63, 49, 51, 65, 46, 63, 75, 49, 48, 71, 47, 72, 61, 63, 74, 69, 66, 67, 57, 57, 51, 75, 58, 69, 63, 74, 69, 63, 59, 68, 73, 64, 63, 69, 59, 58, 51, 62, 81, 56, 70, 60, 66, 52, 79, 54, 54, 67, 69, 56, 63, 72, 55, 48, 55, 53, 63, 50, 49, 63, 83, 57, 70, 60, 59, 71, 53, 62, 65, 67, 56, 60, 62, 75, 59, 65, 66, 69, 59, 49, 68, 49, 65, 58, 51, 60, 59, 54, 59, 51, 52, 56, 55, 59, 53, 59, 52, 53, 51, 69, 68, 66, 66, 51, 73, 68, 63, 73, 62, 66, 56, 66, 70, 61, 63, 60, 65, 71, 81, 52, 87, 66, 70, 64, 55, 56, 62, 65, 66, 71, 63, 63, 61, 61, 58, 65, 45, 56, 51, 50, 62, 53, 49, 51, 54, 52, 50, 61, 55, 62, 54, 63, 58, 66, 67, 63, 68, 57, 68, 56, 46, 59, 53, 52, 60, 72, 63, 56, 61, 48, 54, 52, 53, 48, 73, 45, 58, 59, 42, 59, 47, 60, 56, 53, 67, 49, 64, 70, 53, 63, 75, 65, 55, 65, 52, 74, 67, 53, 74, 56, 65, 67, 70, 58, 54, 52, 59, 55, 63, 58, 54, 69, 63, 68, 73, 51, 63, 65, 69, 54, 71, 53, 67, 65, 62, 53, 47, 65, 55, 53, 65, 56, 55, 63, 53, 59, 50, 59, 66, 77, 60, 61, 50, 75, 57, 59, 54, 65, 58, 68, 69, 59, 54, 49, 54, 55, 64, 41, 62, 65, 52, 51, 75, 63, 62, 58, 82, 60, 54, 57, 61, 60, 63, 60, 72, 56, 59, 65, 63, 73, 68, 63, 59, 76, 62, 64, 68, 66, 61, 58, 77, 64, 70, 54, 57, 61, 59, 59, 55, 56, 58, 52, 62, 70, 47, 68, 72, 64, 68, 48, 51, 53, 44, 61, 64, 45, 67, 54, 66, 47, 58, 62, 60, 61, 48, 57, 54, 62, 58, 68, 54, 49, 52, 56, 58, 57, 60, 47, 51, 43, 51, 55, 60, 54, 54, 73, 57, 76, 53, 64, 73, 49, 71, 69, 71, 68, 68, 63, 61, 48, 58, 55, 66, 60, 66, 63, 65, 51, 56, 70, 51, 61, 60, 57, 54, 62, 56, 61, 48, 58, 51, 68, 56, 51, 49, 60, 52, 56, 55, 65, 57, 63, 54, 49, 60, 57, 52, 70, 67, 50, 66, 53, 66, 66, 63, 69, 54, 69, 60, 64, 63, 56, 54, 57, 56, 52, 55, 54, 62, 44, 55, 61, 61, 51, 57, 48, 63, 60, 56, 61, 62, 68, 59, 71, 67, 62, 85, 51, 63, 47, 56, 46, 75, 53, 66, 68, 66, 67, 56, 67, 57, 51, 71, 60, 47, 59, 57, 50, 70, 58, 61, 54, 72, 58, 58, 65, 52, 67, 50, 52, 51, 44, 55, 37, 57, 50, 57, 61, 57, 63, 60, 46, 54, 54, 66, 44, 78, 67, 62, 52, 65, 59, 61, 57, 66, 57, 61, 42, 64, 50, 61, 68, 57, 54, 51, 56, 51, 49, 60, 52, 56, 50, 74, 63, 62, 48, 59, 60, 51, 60, 59, 58, 61, 67, 59, 61, 54, 60, 58, 51, 62, 52, 55, 60, 68, 50, 60, 55, 52, 63, 60, 47, 53, 38, 60, 56, 65, 56, 58, 65, 50, 60, 62, 54, 61, 60, 52, 60, 55, 62, 55, 47, 66, 58, 58, 62, 72, 66, 57, 47, 53, 47, 49, 58, 69, 53, 66, 60, 54, 66, 58, 64, 50, 58, 61, 50, 70, 58, 49, 65, 52, 65, 60, 52, 56, 63, 51, 60, 68, 63, 59, 50, 65, 58, 75, 53, 72, 55, 48, 64, 56, 59, 59, 62, 69, 72, 60, 59, 53, 57, 66, 59, 54, 66, 67, 60, 51, 37, 55, 55, 58, 56, 60, 51, 54, 52, 49, 51, 42, 59, 66, 44, 61, 64, 64, 74, 63, 66, 64, 57, 54, 70, 51, 52, 53, 59, 45, 59, 50, 43, 54, 49, 68, 56, 60, 59, 62, 46, 66, 54, 52, 76, 71, 51, 58, 62, 58, 59, 68, 63, 77, 61, 58, 58, 69, 67, 62, 59, 54, 64, 63, 57, 57, 61, 67, 67, 66, 61, 63, 61, 58, 59, 58, 55, 54, 58, 57, 56, 46, 48, 59, 41, 52, 51, 59, 66, 60, 60, 48, 54, 55, 55, 62, 53, 53, 59, 55, 76, 64, 51, 56, 48, 63, 43, 54, 53, 56, 72, 65, 46, 62, 60, 51, 61, 78, 53, 59, 54, 49, 66, 56, 52, 54, 41, 58, 59, 62, 60, 71, 62, 59, 63, 72, 56, 58, 61, 60, 66, 76, 60, 67, 62, 61, 60, 69, 57, 68, 71, 71, 69, 69, 62, 56, 60, 55, 69, 65, 57, 57, 51, 43, 45, 44, 48, 55, 60, 61, 47, 52, 54, 56, 67, 56, 57, 47, 61, 59, 49, 66, 60, 58, 45, 67, 46, 59, 53, 54, 57, 68, 51, 66, 67, 52, 53, 52, 60, 67, 62, 55, 58, 55, 53, 58, 53, 57, 62, 53, 49, 55, 61, 65, 51, 54, 74, 59, 67, 59, 67, 65, 57, 71, 53, 64, 56, 70, 59, 69, 56, 58, 63, 49, 55, 56, 50, 58, 48, 54, 62, 57, 51, 65, 54, 61, 59, 61, 57, 54, 68, 61, 54, 64, 46, 48, 69, 64, 59, 61, 60, 50, 57, 64, 63, 62, 53, 59, 62, 61, 69, 49, 55, 67, 49, 65, 45, 45, 58, 69, 46, 58, 64, 54, 59, 55, 62, 54, 69, 39, 43, 57, 50, 49, 51, 60, 59, 57, 65, 63, 64, 47, 59, 68, 60, 60, 60, 62, 52, 68, 72, 60, 75, 59, 69, 70, 53, 52, 66, 64, 44, 59, 42, 52, 63, 55, 44, 51, 60, 50, 52, 62, 51, 53, 55, 64, 58, 55, 56, 58, 66, 51, 56, 62, 58, 59, 67, 51, 61, 57, 54, 53, 45, 63, 61, 65, 54, 53, 58, 65, 56, 54, 64, 51, 56, 45, 56, 43, 46, 56, 55, 54, 56, 60, 54, 43, 47, 55, 59, 75, 64, 52, 55, 57, 50, 57, 62, 58, 60, 62, 64, 60, 76, 62, 72, 64, 61, 63, 62, 76, 57, 64, 71, 49, 43, 58, 52, 64, 52, 52, 56, 50, 51, 47, 51, 49, 69, 57, 61, 58, 58, 53, 57, 54, 60, 63, 56, 68, 68, 56, 73, 52, 62, 66, 59, 60, 62, 60, 63, 66, 63, 68, 69, 47, 61, 52, 44, 63, 60, 58, 48, 64, 59, 44, 54, 51, 43, 60, 59, 68, 56, 84, 71, 75, 59, 72, 60, 57, 60, 65, 57, 71, 51, 57, 62, 55, 59, 71, 67, 69, 62, 57, 66, 62, 48, 44, 69, 48, 57, 52, 44, 58, 62, 48, 48, 50, 47, 50, 55, 51, 47, 61, 65, 60, 60, 63, 51, 47, 50, 66, 55, 65, 58, 67, 71, 55, 64, 54, 54, 50, 52, 68, 40, 49, 54, 44, 52, 63, 53, 63, 47, 72, 56, 54, 60, 56, 49, 61, 54, 74, 59, 65, 66, 65, 55, 59, 65, 77, 61]
    # migs_in_slot = [61, 1236, 46, 28, 50, 53, 46, 35, 57, 1301, 37, 51, 49, 37, 34, 47, 49, 39, 39, 1257, 38, 50, 37, 70, 33, 47, 29, 48, 1180, 57, 35, 53, 53, 1153, 47, 49, 41, 47, 32, 1175, 35, 48, 39, 40, 30, 39, 1117, 45, 53, 47, 42, 53, 40, 35, 30, 63, 42, 30, 46, 1200, 39, 34, 53, 48, 48, 41, 54, 38, 1197, 53, 46, 39, 42, 49, 47, 51, 35, 68, 41, 57, 52, 1191, 37, 43, 56, 37, 52, 1188, 37, 55, 1107, 46, 50, 56, 52, 48, 28, 39, 1107, 47, 34, 50, 36, 42, 1123, 39, 48, 39, 48, 1066, 37, 43, 1100, 42, 60, 1155, 47, 49, 48, 42, 43, 1106, 39, 40, 59, 1098, 47, 44, 1016, 45, 43, 48, 50, 50, 35, 45, 62, 67, 1170, 48, 1052, 49, 43, 39, 39, 53, 1076, 67, 52, 51, 40, 56, 45, 45, 1066, 42, 62, 43, 32, 42, 46, 1049, 44, 47, 36, 919, 74, 52, 50, 1005, 36, 49, 47, 46, 53, 31, 45, 914, 51, 47, 1058, 962, 47, 37, 41, 47, 35, 991, 49, 56, 983, 57, 49, 878, 48, 55, 49, 996, 58, 52, 36, 41, 997, 48, 857, 56, 56, 37, 975, 49, 45, 36, 826, 37, 42, 39, 811, 32, 749, 57, 44, 916, 42, 51, 825, 63, 48, 863, 62, 568, 66, 858, 53, 574, 54, 48, 884, 54, 46, 918, 67, 967, 45, 925, 56, 904, 46, 61, 915, 55, 56, 41, 987, 49, 47, 744, 43, 50, 48, 52, 45, 47, 45, 932, 47, 30, 41, 1028, 52, 35, 1081, 38, 49, 45, 55, 51, 47, 1026, 50, 986, 863, 71, 42, 990, 46, 57, 40, 39, 46, 62, 1041, 40, 41, 65, 1013, 60, 44, 62, 797, 39, 45, 43, 38, 60, 46, 44, 1065, 48, 45, 61, 52, 51, 51, 56, 1112, 61, 44, 49, 71, 63, 54, 1184, 45, 49, 46, 44, 41, 59, 46, 54, 56, 49, 1175, 52, 33, 56, 49, 54, 50, 58, 1146, 54, 60, 44, 45, 57, 1079, 55, 46, 46, 42, 59, 42, 44, 1156, 38, 39, 35, 33, 53, 40, 48, 44, 45, 47, 1187, 52, 50, 48, 41, 975, 47, 1024, 53, 52, 51, 1067, 45, 40, 40, 939, 36, 44, 50, 39, 36, 47, 55, 1047, 64, 56, 1002, 59, 48, 57, 51, 49, 56, 1096, 47, 52, 48, 50, 46, 46, 41, 1089, 982, 42, 44, 51, 65, 38, 1061, 47, 42, 55, 50, 55, 44, 56, 48, 51, 31, 41, 50, 1149, 82, 60, 65, 37, 51, 1074, 49, 39, 40, 43, 42, 1003, 33, 39, 58, 48, 48, 53, 999, 42, 38, 44, 28, 56, 34, 42, 1037, 34, 40, 48, 47, 43, 60, 34, 57, 52, 1160, 63, 48, 65, 47, 40, 35, 54, 45, 44, 50, 46, 39, 45, 47, 1267, 54, 69, 44, 49, 51, 42, 43, 43, 1181, 56, 48, 56, 65, 49, 46, 37, 49, 49, 1278, 49, 40, 36, 61, 57, 34, 51, 46, 42, 42, 57, 52, 53, 50, 53, 43, 44, 50, 1270, 45, 47, 56, 38, 41, 42, 53, 41, 53, 46, 53, 53, 36, 1279, 44, 54, 52, 43, 41, 52, 48, 1243, 46, 40, 37, 43, 43, 1221, 48, 46, 56, 32, 54, 43, 1024, 62, 38, 48, 51, 59, 47, 38, 52, 64, 52, 1260, 61, 47, 50, 53, 1118, 48, 50, 37, 27, 53, 1171, 54, 41, 66, 50, 36, 42, 50, 51, 57, 41, 1275, 46, 50, 38, 53, 48, 54, 36, 36, 59, 39, 44, 37, 35, 47, 51, 41, 53, 41, 46, 1338, 61, 57, 49, 51, 41, 50, 1216, 42, 53, 59, 41, 34, 1184, 37, 33, 45, 43, 48, 1159, 45, 52, 44, 47, 47, 1090, 45, 53, 57, 51, 42, 44, 65, 41, 45, 61, 39, 1255, 50, 36, 39, 54, 54, 44, 51, 51, 53, 44, 1220, 34, 60, 35, 60, 55, 53, 44, 1222, 41, 45, 45, 45, 43, 1146, 45, 56, 63, 43, 43, 37, 42, 42, 51, 41, 51, 56, 41, 1215, 41, 50, 48, 44, 45, 1232, 39, 39, 1314, 36, 52, 52, 59, 41, 1233, 45, 48, 38, 46, 53, 53, 35, 1249, 41, 37, 53, 52, 1197, 41, 53, 49, 42, 42, 1195, 52, 55, 53, 50, 53, 40, 64, 57, 48, 1236, 47, 49, 57, 44, 45, 35, 1165, 59, 49, 42, 1169, 41, 41, 59, 30, 48, 1198, 46, 47, 38, 41, 46, 49, 49, 30, 1169, 47, 54, 19, 40, 56, 47, 53, 49, 43, 45, 1255, 42, 44, 41, 46, 51, 41, 49, 46, 49, 1136, 33, 47, 49, 50, 40, 1105, 40, 35, 46, 46, 55, 44, 36, 1171, 43, 45, 36, 50, 41, 41, 1145, 48, 38, 48, 52, 38, 39, 1089, 47, 54, 43, 48, 46, 1147, 61, 44, 58, 56, 71, 53, 1175, 53, 55, 55, 53, 38, 1114, 39, 57, 51, 1098, 50, 50, 43, 34, 42, 40, 42, 36, 37, 1170, 48, 41, 40, 53, 1044, 48, 43, 1016, 44, 32, 51, 1128, 47, 38, 53, 64, 1123, 53, 35, 58, 47, 49, 65, 1209, 48, 62, 44, 39, 38, 50, 1228, 42, 35, 53, 54, 1045, 56, 48, 49, 35, 38, 61, 41, 45, 62, 1229, 35, 50, 44, 38, 59, 59, 1142, 57, 52, 40, 49, 35, 60, 1216, 59, 39, 46, 51, 70, 51, 45, 58, 1211, 71, 48, 52, 40, 46, 56, 40, 49, 54, 1204, 64, 42, 40, 56, 43, 45, 31, 47, 1160, 42, 1127, 63, 44, 54, 45, 46, 1049, 47, 1216, 42, 39, 45, 52, 44, 1010, 51, 44, 1042, 51, 43, 48, 63, 44, 49, 39, 48, 40, 63, 1150, 64, 42, 55, 53, 68, 61, 43, 44, 1157, 43, 55, 51, 60, 71, 56, 53, 1141, 52, 75, 58, 56, 59, 1183, 53, 63, 58, 45, 40, 54, 1016, 43, 54, 51, 51, 43, 1051, 54, 49, 60, 1037, 40, 46, 42, 49, 36, 46, 1029, 48, 58, 49, 46, 40, 1105, 38, 49, 996, 55, 49, 37, 44, 30, 25, 29, 1045, 60, 45, 44, 61, 58, 45, 48, 44, 46, 42, 48, 1149, 43, 57, 46, 54, 50, 44, 34, 1104, 45, 53, 51, 46, 948, 25, 961, 42, 56, 50, 41, 61, 1143, 42, 56, 55, 49, 44, 40, 44, 48, 1117, 42, 53, 48, 43, 52, 52, 1100, 52, 63, 56, 45, 43, 38, 37, 61, 1089, 41, 55, 35, 937, 36, 37, 38, 932, 37, 30, 48, 44, 46, 56, 1002, 37, 38, 47, 35, 1025, 28, 36, 51, 35, 1040, 44, 64, 52, 48, 57, 1130, 65, 57, 46, 40, 1118, 47, 46, 1022, 49, 52, 46, 1083, 44, 51, 43, 41, 1008, 56, 56, 38, 42, 1067, 71, 58, 64, 1010, 65, 46, 71, 62, 60, 1151, 63, 60, 67, 1034, 38, 61, 49, 61, 59, 42, 1117, 58, 35, 955, 46, 64, 1089, 63, 47, 51, 34, 60, 49, 69, 34, 37, 52, 44, 1174, 41, 41, 38, 53, 983, 41, 39, 57, 49, 67, 1086, 32, 43, 50, 66, 55, 46, 58, 36, 1158, 43, 39, 42, 50, 47, 1113, 47, 50, 49, 1052, 54, 54, 1066, 60, 47, 49, 56, 52, 56, 33, 1158, 59, 40, 58, 56, 49, 52, 51, 1078, 49, 65, 50, 1122, 54, 59, 56, 1192, 52, 48, 48, 70, 1128, 49, 65, 41, 61, 56, 1118, 53, 46, 59, 1013, 61, 63, 65, 980, 69, 48, 54, 50, 53, 62, 52, 34, 56, 38, 36, 1086, 52, 47, 43, 55, 45, 1068, 41, 53, 36, 1112, 48, 640, 44, 47, 44, 42, 34, 48, 66, 1080, 50, 49, 51, 1094, 47, 52, 68, 43, 70, 45, 1123, 61, 39, 45, 36, 60, 1173, 60, 57, 67, 1051, 52, 64, 49, 32, 51, 52, 56, 1173, 50, 51, 1105, 56, 62, 51, 60, 42, 56, 1139, 42, 1145, 45, 50, 54, 1037, 36, 41, 53, 48, 1125, 57, 35, 45, 61, 1107, 53, 60, 48, 50, 38, 55, 1080, 57, 943, 54, 48, 55, 60, 60, 59, 46, 48, 1131, 52, 67, 39, 57, 55, 42, 41, 1095, 41, 47, 46, 30, 951, 43, 51, 49, 56, 977, 52, 48, 25, 27, 49, 941, 57, 902, 54, 913, 58, 63, 52, 1079, 68, 57, 47, 46, 1050, 900, 51, 65, 64, 934, 56, 44, 50, 914, 45, 51, 42, 55, 59, 1010, 37, 56, 61, 54, 1033, 41, 56, 40, 55, 997, 57, 53, 43, 949, 51, 41, 52, 39, 928, 40, 1050, 55, 64, 959, 52, 1090, 50, 63, 45, 56, 58, 44, 1086, 50, 48, 55, 53, 49, 939, 60, 51, 1115, 59, 72, 50, 61, 42, 70, 1030, 58, 48, 42, 58, 992, 41, 47, 34, 41, 54, 56, 57, 52, 52, 74, 52, 1118, 59, 67, 68, 60, 1076, 949, 43, 992, 62, 47, 45, 41, 61, 989, 64, 51, 37, 51, 61, 994, 48, 55, 56, 38, 59, 947, 61, 51, 56, 884, 54, 47, 896, 50, 54, 48, 41, 962, 41, 47, 45, 43, 914, 56, 60, 51, 890, 74, 54, 43, 59, 51, 48, 67, 987, 63, 54, 56, 944, 887, 53, 48, 54, 53, 993, 56, 986, 46, 47, 885, 51, 39, 1017, 60, 67, 56, 60, 58, 1037, 51, 77, 51, 69, 69, 1113, 56, 58, 1051, 48, 50, 55, 44, 55, 42, 61, 1084, 59, 50, 49, 53, 944, 46, 49, 56, 43, 1040, 43, 57, 948, 65, 51, 52, 54, 869, 49, 40, 52, 45, 53, 50, 44, 1018, 42, 55, 52, 49, 57, 946, 63, 60, 64, 52, 958, 42, 42, 52, 998, 56, 53, 40, 50, 934, 40, 55, 979, 48, 37, 972, 44, 53, 58, 57, 54, 50, 1051, 66, 54, 52, 59, 963, 86, 1032, 63, 63, 1112, 44, 55, 56, 52, 1027, 50, 56, 46, 48, 60, 46, 1045, 885, 54, 61, 52, 57, 1056, 916, 40, 936, 884, 50, 48, 972, 41, 963, 44, 56, 49, 55, 40, 982, 55, 65, 60, 912, 838, 53, 42, 47, 929, 53, 48, 852, 75, 62, 884, 56, 52, 40, 898, 49, 47, 43, 48, 809, 63, 801, 785, 50, 44, 55, 814, 66, 774, 52, 866, 36, 47, 849, 53, 54, 56, 861, 49, 66, 57, 807, 64, 50, 850, 45, 64, 842, 53, 748, 48, 821, 52, 525, 57, 53, 52, 50, 50, 930, 50, 990, 44, 881, 51, 50, 61, 836, 61, 53, 39, 44, 872, 790, 54, 46, 780, 41, 791, 47, 876, 53, 49, 52, 31, 62, 901, 60, 52, 56, 965, 60, 39, 56, 65, 45, 57, 56, 980, 69, 753, 60, 41, 913, 55, 53, 998, 56, 35, 1011, 63, 51, 32, 964, 72, 819, 59, 1032, 58, 48, 59, 52, 52, 984, 53, 61, 52, 40, 951, 51, 56, 56, 47, 49, 59, 48, 1024, 66, 820, 53, 44, 945, 48, 687, 55, 56, 53, 53, 41, 39, 1017, 44, 46, 76, 45, 64, 40, 971, 47, 636, 45, 45, 798, 45, 955, 39, 59, 857, 52, 49, 48, 43, 978, 49, 41, 62, 61, 45, 44, 1030, 48, 46, 49, 42, 950, 60, 49, 62, 53, 46, 887, 66, 44, 898, 48, 61, 844, 53, 34, 844, 70, 71, 62, 55, 990, 56, 37, 50, 976, 56, 64, 44, 766, 51, 58, 967, 41, 47, 880, 61, 39, 66, 50, 42, 56, 967, 49, 59, 843, 57, 527, 60, 46, 844, 38, 42, 51, 52, 32, 56, 34, 52, 49, 1015, 52, 53, 47, 888, 36, 59, 48, 37, 892, 48, 58, 44, 36, 943, 865, 51, 37, 42, 52, 60, 52, 1002, 53, 42, 55, 55, 57, 35, 972, 61, 48, 930, 55, 47, 1005, 52, 57, 1000, 65, 59, 53, 53, 57, 57, 53, 1084, 57, 51, 58, 40, 41, 40, 49, 48, 55, 59, 1047, 46, 71, 50, 56, 42, 53, 62, 1133, 65, 45, 51, 54, 47, 1115, 45, 54, 38, 1037, 44, 39, 50, 919, 30, 41, 48, 37, 49, 41, 45, 27, 40, 29, 1125, 46, 44, 42, 48, 60, 1112, 66, 74, 66, 48, 1062, 52, 36, 49, 1066, 43, 49, 46, 43, 1057, 45, 46, 43, 39, 1091, 50, 57, 36, 58, 60, 42, 1041, 56, 53, 60, 52, 57, 1011, 45, 42, 878, 57, 51, 54, 38, 993, 61, 64, 56, 61, 45, 988, 45, 682, 60, 63, 66, 40, 64, 46, 45, 1047, 58, 53, 54, 31, 52, 44, 54, 47, 31, 31, 63, 42, 42, 1144, 45, 33, 50, 55, 1117, 48, 45, 60, 72, 63, 51, 44, 1168, 36, 46, 48, 59, 47, 1164, 47, 38, 48, 56, 59, 1139, 49, 41, 49, 55, 42, 39, 1129, 44, 41, 56, 57, 58, 54, 51, 1155, 60, 51, 56, 1119, 60, 60, 60, 1186, 50, 53, 1133, 61, 45, 59, 67, 1170, 56, 40, 58, 57, 52, 1147, 50, 1125, 1039, 42, 36, 42, 50, 46, 54, 54, 39, 1184, 39, 512, 47, 52, 31, 40, 1134, 44, 41, 53, 48, 24, 41, 40, 38, 51, 53, 44, 1206, 41, 48, 1073, 61, 57, 62, 48, 1189, 45, 40, 49, 38, 1186, 46, 54, 53, 1044, 46, 43, 55, 49, 43, 1198, 44, 66, 58, 47, 45, 1177, 55, 56, 1135, 56, 55, 54, 48, 1104, 64, 55, 67, 1015, 49, 49, 61, 56, 1102, 60, 47, 53, 48, 36, 38, 1142, 46, 51, 53, 37, 38, 55, 1098, 47, 46, 44, 50, 1078, 56, 59, 33, 59, 1149, 57, 39, 51, 56, 53, 1136, 47, 54, 38, 62, 53, 1068, 49, 41, 911, 45, 44, 61, 989, 63, 49, 51, 64, 52, 1035, 952, 39, 48, 41, 1093, 37, 56, 57, 53, 33, 951, 50, 56, 1078, 43, 36, 48, 59, 58, 51, 61, 54, 59, 64, 50, 51, 60, 43, 1235, 64, 67, 1052, 55, 63, 59, 64, 67, 61, 45, 56, 1178, 59, 42, 66, 37, 43, 1071, 44, 55, 58, 46, 1110, 50, 39, 59, 66, 51, 1022, 48, 50, 1003, 43, 981, 55, 55, 44, 45, 1164, 47, 56, 38, 1106, 32, 61, 51, 1076, 44, 67, 60, 66, 48, 60, 40, 57, 50, 1229, 49, 1171, 52, 44, 60, 46, 42, 55, 57, 46, 54, 44, 46, 1268, 61, 49, 39, 54, 64, 55, 43, 60, 55, 1200, 62, 45, 59, 50, 48, 994, 48, 62, 46, 54, 56, 63, 45, 66, 1198, 64, 50, 60, 40, 48, 44, 1126, 47, 45, 58, 51, 38, 49, 59, 52, 56, 46, 60, 47, 50, 59, 1254, 66, 51, 42, 1145, 43, 55, 1134, 55, 42, 43, 43, 44, 41, 1133, 49, 64, 46, 1096, 71, 40, 48, 61, 47, 63, 50, 1171, 62, 55, 65, 1079, 60, 47, 37, 60, 57, 77, 51, 1225, 65, 58, 48, 66, 56, 63, 1152, 67, 41, 1090, 44, 52, 52, 48, 55, 1117, 52, 46, 69, 39, 45, 1107, 54, 43, 46, 54, 41, 40, 42, 45, 50, 37, 1175, 54, 70, 49, 50, 49, 56, 54, 38, 1170, 46, 60, 55, 44, 51, 73, 41, 41, 55, 1233, 56, 42, 63, 42, 54, 45, 1155, 60, 1084, 41, 54, 39, 46, 53, 1096, 57, 43, 49, 52, 49, 37, 57, 60, 46, 64, 39, 52, 1192, 52, 55, 64, 70, 44, 1115, 66, 53, 53, 51, 1078, 57, 59, 44, 66, 51, 59, 1096, 47, 46, 57, 955, 63, 59, 56, 57, 1034, 51, 44, 51, 32, 40, 927, 34, 40, 40, 32, 37, 41, 938, 53, 48, 918, 55, 40, 44, 53, 990, 61, 882, 58, 941, 54, 51, 41, 998, 44, 39, 56, 57, 57, 1032, 43, 45, 47, 37, 1008, 45, 63, 37, 45, 935, 26, 58, 38, 56, 47, 49, 59, 43, 1048, 62, 63, 932, 57, 63, 47, 64, 1017, 70, 918, 44, 60, 48, 49, 57, 60, 981, 46, 45, 48, 41, 55, 51, 52, 1056, 48, 49, 57, 35, 57, 907, 52, 45, 53, 885, 51, 66, 55, 885, 40, 51, 39, 767, 48, 966, 46, 835, 48, 53, 42, 985, 858, 62, 38, 902, 49, 68, 50, 46, 1024, 52, 1060, 51, 42, 48, 50, 999, 49, 54, 46, 975, 65, 64, 961, 888, 62, 696, 58, 55, 70, 65, 48, 47, 48, 55, 1141, 952, 59, 57, 44, 55, 1031, 53, 65, 53, 48, 1019, 47, 956, 52, 49, 51, 926, 51, 57, 62, 46, 983, 65, 42, 48, 45, 52, 43, 50, 966, 63, 37, 914, 59, 56, 67, 42, 41, 1019, 41, 48, 805, 42, 59, 943, 53, 45, 51, 961, 39, 52, 923, 51, 46, 943, 52, 64, 36, 926, 35, 976, 49, 43, 941, 37, 46, 31, 38, 49, 54, 1110, 53, 79, 943, 74, 48, 46, 899, 37, 823, 59, 55, 71, 976, 48, 863, 42, 40, 40, 45, 968, 50, 50, 57, 54, 45, 1011, 824, 54, 777, 46, 853, 60, 52, 51, 36, 48, 42, 50, 947, 47, 43, 919, 29, 59, 42, 50, 44, 907, 28, 649, 59, 852, 41, 68, 61, 40, 946, 40, 62, 47, 62, 63, 60, 50, 999, 53, 53, 886, 46, 45, 46, 730, 54, 43, 56, 32, 43, 972, 58, 42, 57, 790, 41, 49, 58, 891, 49, 57, 940, 68, 50, 48, 78, 984, 54, 34, 58, 30, 63, 1074, 55, 1126, 64, 789, 50, 794, 892, 41, 52, 46, 44, 986, 49, 55, 61, 52, 56, 1034, 54, 55, 36, 1066, 41, 53, 1027, 41, 49, 1087, 44, 710, 54, 41, 976, 60, 1011, 56, 625, 40, 53, 43, 50, 44, 1046, 50, 47, 35, 52, 41, 934, 43, 52, 42, 40, 37, 936, 43, 51, 858, 54, 830, 48, 44, 41, 771, 49, 58, 52, 41, 66, 913, 39, 55, 52, 60, 883, 53, 61, 55, 889, 63, 57, 888, 43, 51, 42, 36, 53, 56, 46, 1000, 58, 35, 896, 40, 48, 54, 47, 947, 45, 42, 37, 924, 40, 38, 51, 41, 962, 51, 52, 43, 996, 51, 34, 902, 45, 56, 38, 36, 954, 48, 61, 906, 65, 50, 50, 823, 41, 39, 911, 57, 903, 48, 60, 921, 59, 58, 56, 1015, 882, 53, 58, 37, 63, 888, 31, 47, 46, 65, 50, 49, 52, 53, 994, 48, 57, 1012, 45, 44, 58, 56, 65, 42, 55, 44, 41, 56, 46, 53, 46, 55, 46, 56, 47, 51, 1160, 56, 59, 60, 44, 56, 57, 987, 39, 32, 44, 52, 51, 38, 43, 959, 42, 904, 45, 42, 39, 50, 56, 27, 39, 1059, 55, 56, 63, 955, 49, 40, 43, 50, 40, 1042, 40, 49, 936, 43, 44, 968, 46, 715, 57, 531, 43, 682, 57, 819, 56, 840, 48, 1158, 928, 40, 48, 43, 66, 52, 57, 60, 73, 38, 1072, 54, 61, 59, 37, 50, 1027, 52, 892, 37, 41, 793, 53, 65, 59, 955, 52, 58, 56, 41, 50, 38, 35, 967, 876, 45, 36, 35, 850, 33, 42, 973, 41, 52, 874, 50, 39, 53, 46, 967, 53, 43, 33, 57, 1043, 59, 63, 39, 941, 38, 763, 30, 49, 43, 895, 70, 57, 35, 882, 43, 41, 41, 57, 1030, 41, 40, 822, 48, 47, 942, 38, 913, 46, 63, 48, 47, 55, 51, 55, 50, 68, 1159, 58, 791, 50, 63, 68, 45, 1036, 55, 59, 53, 52, 37, 66, 44, 1083, 58, 45, 44, 49, 1025, 47, 73, 1032, 44, 861, 46, 25, 37, 38, 37, 40, 45, 1007, 43, 47, 49, 49, 53, 46, 56, 46, 1097, 51, 46, 48, 62, 53, 37, 1152, 44, 47, 50, 42, 54, 59, 43, 48, 65, 1146, 54, 41, 50, 49, 1008, 39, 50, 54, 39, 57, 40, 47, 44, 38, 56, 51, 51, 43, 51, 50, 56, 47, 49, 50, 1262, 65, 43, 55, 41, 53, 39, 54, 51, 1314, 54, 38, 47, 42, 1190, 42, 45, 46, 46, 39, 54, 1097, 42, 65, 42, 48, 1119, 44, 42, 40, 59, 51, 32, 44, 1065, 47, 63, 56, 42, 49, 63, 40, 49, 55, 1235, 56, 52, 1082, 44, 51, 58, 43, 50, 60, 45, 1145, 36, 37, 48, 59, 37, 44, 1048, 39, 44, 40, 55, 53, 61, 26, 42, 49, 52, 41, 43, 51, 1128, 44, 52, 59, 48, 41, 58, 50, 1103, 51, 55, 55, 46, 1092, 68, 48, 63, 56, 58, 1034, 46, 38, 53, 1000, 37, 64, 905, 35, 798, 43, 996, 42, 55, 41, 985, 51, 50, 34, 54, 45, 1017, 36, 62, 39, 958, 52, 997, 52, 45, 955, 60, 48, 51, 50, 47, 45, 44, 1085, 66, 61, 27, 936, 40, 46, 36, 41, 62, 39, 43, 1019, 33, 35, 39, 42, 42, 43, 35, 992, 51, 35, 30, 874, 58, 50, 956, 48, 948, 58, 30, 34, 48, 45, 51, 1036, 58, 46, 57, 47, 56, 53, 46, 1045, 52, 55, 51, 46, 48, 34, 32, 55, 40, 1079, 48, 53, 36, 48, 41, 30, 49, 37, 46, 40, 1106, 47, 49, 36, 36, 1130, 52, 63, 44, 60, 57, 51, 60, 51, 45, 1150, 1004, 49, 44, 47, 47, 52, 46, 62, 1094, 41, 49, 45, 1060, 54, 47, 48, 918, 59, 55, 987, 36, 44, 43, 43, 40, 44, 44, 68, 1060, 67, 1099, 68, 57, 45, 55, 61, 49, 53, 44, 1088, 53, 51, 56, 903, 65, 48, 913, 51, 50, 53, 44, 943, 59, 46, 907, 43, 46, 47, 60, 49, 969, 49, 42, 768, 835, 858, 38, 57, 906, 41, 64, 50, 43, 898, 43, 48, 927, 70, 861, 63, 56, 913, 42, 939, 48, 36, 47, 42, 37, 33, 62, 1018, 41, 52, 38, 46, 31, 1054, 52, 35, 32, 46, 35, 55, 53, 59, 46, 1106, 58, 53, 60, 53, 46, 1023, 51]
    # plt.figure()
    # plt.title ('Migrations and mobility at each cycle')
    # plt.plot (range(len(moves_in_slot)), moves_in_slot, label='num of usrs moved to another cell', linestyle='None',  marker='o', markersize = 4)
    # plt.plot (range(len(migs_in_slot)), migs_in_slot, label='num of chains migrated by the algorithm', linestyle='None',  marker='.', markersize = 4)
    # plt.xlabel ('time [seconds, starting at 07:30]')
    # plt.legend()
    # plt.savefig ('../res/vehicles_n_speed_0730.mob.jpg')
    # plt.clf()
