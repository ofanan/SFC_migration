import networkx as nx
import numpy as np
import math
import itertools 
import time
import heapq
import pulp as plp
from cmath import sqrt
import matplotlib.pyplot as plt
import random
from pathlib import Path

from usr_c    import usr_c    # class of the users of alg
from usr_lp_c import usr_lp_c # class of the users, when using LP
from decision_var_c import decision_var_c # class of the decision variables
from printf import printf
import loc2ap_c

# Levels of verbose / operation modes (which output is generated)
VERBOSE_DEBUG         = 0
VERBOSE_RES           = 1 # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_LOG           = 2 # Write to a ".log" file
VERBOSE_ADD_LOG       = 3 # Write to a detailed ".log" file
VERBOSE_ADD2_LOG      = 4
VERBOSE_MOB           = 5 # Write data about the mobility of usrs, and about the num of migrated chains per slot
VERBOSE_COST_COMP     = 6 # Print the cost of each component in the cost function
VERBOSE_CALC_RSRC_AUG = 7 # Use binary-search to calculate the minimal reseource augmentation required to find a sol 
VERBOSE_MOVED_RES     = 8 # calculate the cost incurred by the usrs who moved only 
VERBOSE_CRITICAL_RES  = 9 # calculate the cost incurred by the usrs who moved only 
VERBOSE_FLAVORS       = 10 # Use 2 users' flavors

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

    # calculate the total cost of a solution by an algorithm (not by LP)
    calc_alg_sol_cost = lambda self, usrs: sum ([self.chain_cost_homo (usr, usr.lvl) for usr in usrs])
   
    # calculate the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    chain_cost_homo = lambda self, usr, lvl: self.link_cost_of_CLP_at_lvl[lvl] + self.CPU_cost_at_lvl[lvl] * usr.B[lvl] + self.calc_mig_cost_CLP (usr, lvl)     
    
    # # calculate the migration cost incurred for a usr if placed on a given lvl, assuming a CLP (co-located placement), namely, the whole chain is placed on a single server
    calc_mig_cost_CLP = lambda self, usr, lvl : (usr.S_u[lvl] != usr.cur_s and usr.cur_s!=-1) * self.uniform_mig_cost * len (usr.theta_times_lambda)
          
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
                    len (usr.theta_times_lambda) * self.uniform_mig_cost + self.CPU_cost_at_lvl[lvl] * usr.B[lvl] + self.link_cost_of_CLP_at_lvl[lvl]
                    
    # Print a solution for the problem to the output res file  
    print_sol_res_line = lambda self, output_file, sol_cost : printf (output_file, 't{}_{}_cpu{}_p{}_stts{} | cost = {:.0f}\n' .format(self.t, self.alg, self.G.nodes[len (self.G.nodes)-1]['RCs'], self.prob_of_target_delay[0], self.stts, sol_cost)) 

    # parse a line detailing the list of usrs who moved, in an input ".ap" format file
    parse_old_usrs_line = lambda self, line : list (filter (lambda item : item != '', line[0].split ("\n")[0].split (")")))

    # returns true iff server s has enough available capacity to host user u; assuming that usr.B (the required cpu allowing to place u on each lvl) is known  
    s_has_sufic_avail_cpu_for_usr = lambda self, s, usr : (self.G.nodes[s]['a'] >= usr.B[self.G.nodes[s]['lvl']])
    
    # returns a list of the unplaced usrs. As old usrs that are not critical are considered placed, the returned list is actually also the list of critical usrs. 
    unplaced_usrs  = lambda self : list (filter (lambda usr : usr.nxt_s==-1, self.usrs)) 

    # sort the given list of usrs in a first-fit manner
    first_fit_sort = lambda self, list_of_usrs: sorted (list_of_usrs, key = lambda usr : usr.rand_id)

    # sort the given list of usrs in a cpvnf manner
    cpvnf_sort     = lambda self, list_of_usrs: sorted (list_of_usrs, key = lambda usr : (usr.B[0], usr.rand_id), reverse=True)

    # Randomize a target delay for a usr, using the distribution and values defined in self.
    randomize_target_delay = lambda self : self.target_delay[0] if (random.random() < self.prob_of_target_delay[0]) else self.target_delay[1] 
    
    
    gen_res_file_name  = lambda self, mid_str : '../res/{}{}_p{}.res' .format (self.ap_file_name.split(".")[0], mid_str, self.prob_of_target_delay[0])

    calc_cpu_capacities = lambda self, cpu_cap_at_leaf : np.array ([cpu_cap_at_leaf * (lvl+1) for lvl in range (self.tree_height+1)], dtype='uint16')
    
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
        del (self.total_cpu_cost_in_slot[0])
        del (self.total_link_cost_in_slot[0])
        del (self.total_mig_cost_in_slot[0])
        total_cost = [self.total_cpu_cost_in_slot[t] + self.total_link_cost_in_slot[t] + self.total_mig_cost_in_slot[t] for t in range(len(self.total_cpu_cost_in_slot))] #Ignore the cost in the first slot, in which there're no mig        
        printf (self.cost_comp_output_file, 'total_cost = {}\n' .format (total_cost))
        printf (self.cost_comp_output_file, 'cpu_cost={}\nlink_cost={}\nmig_cost={}\n' .format (
                self.total_cpu_cost_in_slot, self.total_link_cost_in_slot, self.total_mig_cost_in_slot))

        cpu_cost_ratio  = [self.total_cpu_cost_in_slot[t]/total_cost[t]  for t in range(len(total_cost))]
        link_cost_ratio = [self.total_link_cost_in_slot[t]/total_cost[t] for t in range(len(total_cost))]
        mig_cost_ratio  = [self.total_mig_cost_in_slot[t]/total_cost[t]  for t in range(len(total_cost))]
        printf (self.cost_comp_output_file, 'cpu_cost_ratio = {}\nlink_cost_ratio = {}\nmig_cost_ratio = {}\n'.format (
            cpu_cost_ratio, link_cost_ratio, mig_cost_ratio))
            
        printf (self.cost_comp_output_file, 'avg ratio are: cpu={:.3f}, link={:.3f}, mig={:.3f}\n' .format (
            np.average(cpu_cost_ratio), np.average(link_cost_ratio), np.average(mig_cost_ratio) ) )
            
    def calc_sol_cost_components (self):
        """
        Calculates and keeps the cost of each component in the cost function (cpu, link, and migration). 
        """
        
        self.total_cpu_cost_in_slot.append  (sum ([self.CPU_cost_at_lvl[usr.lvl] * usr.B[usr.lvl] for usr in self.usrs]))
        self.total_link_cost_in_slot.append (sum ([self.link_cost_of_CLP_at_lvl[usr.lvl]          for usr in self.usrs]))
        self.total_mig_cost_in_slot.append  (sum ([self.calc_mig_cost_CLP(usr, usr.lvl)               for usr in self.usrs]))
     
    def print_sol_to_res_and_log (self):
        """
        Print to the res, log, and debug files the solution and/or additional info.
        """
        if (VERBOSE_RES in self.verbose and (not(self.is_first_t))): # in the first slot there're no migrations (only placement), and hence the cost is wrong, and we ignore it.
            self.print_sol_res_line (output_file=self.res_output_file, sol_cost=self.calc_alg_sol_cost(self.usrs))
        if (VERBOSE_MOVED_RES in self.verbose and (not(self.is_first_t))): # in the first slot there're no migrations (only placement), and hence the cost is wrong, and we ignore it.
            self.print_sol_res_line (output_file=self.moved_res_output_file,    sol_cost=self.calc_alg_sol_cost(self.moved_usrs))
        if (VERBOSE_CRITICAL_RES in self.verbose and (not(self.is_first_t))): # in the first slot there're no migrations (only placement), and hence the cost is wrong, and we ignore it.
            self.print_sol_res_line (output_file=self.critical_res_output_file, sol_cost=self.calc_alg_sol_cost(self.critical_usrs))
        elif (VERBOSE_COST_COMP in self.verbose):
            self.calc_sol_cost_components()

        if (VERBOSE_LOG in self.verbose): # Commented-out, because this is already printed during alg_top()
            printf (self.log_output_file, 'after push-up:\n')
            self.print_sol_res_line (output_file=self.log_output_file, sol_cost=self.calc_alg_sol_cost(self.usrs))
        if (VERBOSE_ADD_LOG in self.verbose):
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
            usrs_who_migrated_at_this_slot = list (filter (lambda usr: usr.cur_s != -1 and usr.cur_s != usr.nxt_s, self.usrs))
            self.num_of_migs_in_slot.append (len(usrs_who_migrated_at_this_slot))
            for usr in usrs_who_migrated_at_this_slot: 
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

    def calc_cost_of_moved_usrs_plp (self):
        """
        Returns the overall cost of users that moved in a LP solution 
        """
        
        cost = 0
        for d_var in self.d_vars: 
            if d_var.plp_var.value() > 0: # the LP solution set non-zero value for this decision variable
                if d_var.usr in self.moved_usrs:
                    cost += self.chain_cost_from_non_CLP_state (d_var.usr, d_var.lvl)
        return cost

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
                id += 1
                if (VERBOSE_MOVED_RES in self.verbose and not (self.is_first_t) and usr not in self.moved_usrs): 
                    continue # In this mode, starting from the 2nd slot, the obj' func' should consider only the users who moved 
                # if (VERBOSE_CRITICAL_RES in self.verbose and not (self.is_first_t) and usr not in self.critical_usrs): 
                #     continue # In this mode, starting from the 2nd slot, the obj' func' should consider only the users who are critical 
                obj_func += self.chain_cost_from_non_CLP_state (usr, lvl) * plp_var # add the cost of this decision var to the objective func
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
        
        self.stts = model.status
        
        if (VERBOSE_RES in self.verbose):
            self.print_sol_res_line (output_file=self.res_output_file, sol_cost=model.objective.value())
        if (VERBOSE_MOVED_RES in self.verbose and not(self.is_first_t)):
            self.print_sol_res_line (output_file=self.moved_res_output_file, sol_cost=self.calc_cost_of_moved_usrs_plp())
        sol_status = plp.LpStatus[model.status] 
        if (VERBOSE_LOG in self.verbose):            
            self.print_sol_res_line (output_file=self.res_output_file, sol_cost=model.objective.value())
        if (model.status == 1): # successfully solved
            if (VERBOSE_LOG in self.verbose):            
                self.print_lp_sol_to_log ()
                printf (self.log_output_file, '\nSuccessfully solved in {:.3f} [sec]\n' .format (time.time() - self.last_rt))
        else:
            print  ('Running the LP failed. status={}' .format(plp.LpStatus[model.status]))
            if (VERBOSE_LOG in self.verbose):
                printf (self.log_output_file, 'failed. status={}\n' .format(plp.LpStatus[model.status]))
        return self.stts

    def init_res_file (self):
        """
        Open the res file for writing, as follows:
        If a res file with the relevant name already exists - open it for appending.
        Else, open a new res file, and write to it comment header lines, explaining the file's format  
        """
        
        self.res_file_name = self.gen_res_file_name (mid_str = ('_opt' if self.alg=='opt' else '') )
        
        if Path('../res/' + self.res_file_name).is_file(): # does this res file already exist?
            self.res_output_file =  open ('../res/' + self.res_file_name,  "a")
        else:
            self.res_output_file =  open ('../res/' + self.res_file_name,  "w")
            self.print_res_file_prefix (self.res_output_file)

    def init_moved_res_file (self):
        """
        Open the res file for writing, as follows:
        If a res file with the relevant name already exists - open it for appending.
        Else, open a new res file, and write to it comment header lines, explaining the file's format  
        """
        self.moved_res_file_name = self.gen_res_file_name (mid_str = ('_moved_opt' if self.alg=='opt' else '_moved')) 
        
        if Path('../res/' + self.moved_res_file_name).is_file(): # does this res file already exist?
            self.moved_res_output_file =  open ('../res/' + self.moved_res_file_name,  "a")
        else:
            self.moved_res_output_file =  open ('../res/' + self.moved_res_file_name,  "w")
            self.print_res_file_prefix (self.moved_res_output_file)

    def init_critical_res_file (self):
        """
        Open the res file for writing, as follows:
        If a res file with the relevant name already exists - open it for appending.
        Else, open a new res file, and write to it comment header lines, explaining the file's format  
        """
        self.critical_res_file_name = self.gen_res_file_name (mid_str = ('_critical_opt' if self.alg=='opt' else '_critical')) 
        
        if Path('../res/' + self.critical_res_file_name).is_file(): # does this res file already exist?
            self.critical_res_output_file =  open ('../res/' + self.critical_res_file_name,  "a")
        else:
            self.critical_res_output_file =  open ('../res/' + self.critical_res_file_name,  "w")
            self.print_res_file_prefix (self.critical_res_output_file)

    def print_res_file_prefix (self, res_file):
        printf (res_file, '// format: t{T}.{Alg}.cpu{C}.stts{s} | cost = c, where\n// T is the slot cnt (read from the input file)\n')
        printf (res_file, '// Alg is the algorithm / solver used.\n// C is the num of CPU units used in the leaf\n')
        printf (res_file, '// c is the total cost of the solution\n\n') 

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
                if (VERBOSE_ADD2_LOG in self.verbose and used_cpu_in_s > 0): 
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
        
    def push_up (self, usrs):
        """
        Push-up chains: take a feasible solution, and greedily try pushing each chain as high as possible in the tree. 
        Do that when chains are sorted in a decreasing order of the # of CPU units they're currently using.
        """
        # Assume here that the available cap' at each server 'a' is already calculated by the alg' that was run 
        # before calling push-up ()
        heapq._heapify_max(usrs)
 
        n = 0  # num of failing push-up tries in sequence; when this number reaches the number of users, return

        while n < len (usrs):
            usr = usrs[n]
            for lvl in range (len(usr.B)-1, usr.lvl, -1): #
                if (self.G.nodes[usr.S_u[lvl]]['a'] >= usr.B[lvl] and self.chain_cost_homo(usr, lvl) < self.chain_cost_homo(usr, usr.lvl)): # if there's enough available space to move u to level lvl, and this would reduce cost
                    self.G.nodes [usr.nxt_s]    ['a'] += usr.B[usr.lvl] # inc the available CPU at the previosly-suggested place for this usr  
                    self.G.nodes [usr.S_u[lvl]] ['a'] -= usr.B[lvl]     # dec the available CPU at the new  loc of the moved usr
                    
                    # update usr.lvl and usr.nxt_s accordingly 
                    usr.lvl      = lvl               
                    usr.nxt_s    = usr.S_u[usr.lvl]    
                    
                    # update the moved usr's location in the heap
                    usrs[n] = usrs[-1] # replace the usr to push-up with the last usr in the heap
                    usrs.pop() # pop the last user from the heap
                    heapq.heappush(usrs, usr) # push back to the heap the user we have just pushed-up
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
        self.cpu_cap_at_lvl    = self.calc_cpu_capacities (self.cpu_cap_at_leaf)                                
        
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

    def __init__ (self, ap_file_name = 'shorter.ap', verbose = [], tree_height = 3, children_per_node = 4, cpu_cap_at_leaf=561):
        """
        """
        
        # verbose and debug      
        self.verbose                    = verbose

        self.ap_file_name               = ap_file_name #input file containing the APs of all users along the simulation
        
        # Network parameters
        self.tree_height                = tree_height
        self.children_per_node          = children_per_node # num of children of every non-leaf node
        self.cpu_cap_at_leaf            = 28 if self.ap_file_name == 'shorter.ap' else cpu_cap_at_leaf 
        self.uniform_mig_cost           = 200
        self.Lmax                       = 0
        self.uniform_Tpd                = 2
        self.uniform_link_cost          = 3
        self.uniform_theta_times_lambda = [2, 10, 2] # "1" here means 100MHz 
        self.long_chain_theta_times_lambda = [2, 10, 10, 10, 10, 10, 10, 2] # "1" here means 100MHz 
        self.uniform_Cu                 = 20 
        self.target_delay               = [20, 100] # in [ms], lowest to highest
        self.prob_of_target_delay       = [0.3] 
        self.warned_about_too_large_ap  = False
        self.usrs                       = []
        self.max_R                      = 2 # maximal rsrc augmenation to consider
        random.seed                     (42) # Use a fixed pseudo-number seed 
        
        # Init output files
        if (VERBOSE_COST_COMP in self.verbose):
            self.init_cost_comp () 
        if (VERBOSE_DEBUG in self.verbose):
            self.debug_file = open ('../res/debug.txt', 'w') 
        if (VERBOSE_MOB in self.verbose):
            self.num_of_moves_in_slot = [] # self.num_of_moves_in_slot[t] will hold the num of usrs who moved at slot t.   
            self.num_of_migs_in_slot  = [] # self.num_of_migs[t] will hold the num of chains that the alg' migrated in slot t.
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
        usr = usr_c (id=0, theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.randomize_target_delay(), C_u=self.uniform_Cu)
        self.CPUAll_single_usr (usr) 
        if (len(usr.B)==0):
            print ('Error: cannot satisfy delay constraints of usr {}, even on a leaf. theta_times_lambda={}, target_delay ={}' .format (
                    usr.id, usr.theta_times_lambda, usr.target_delay))
            exit ()       

    def simulate (self, alg, sim_len_in_slots=99999, initial_rsrc_aug=1):
        """
        Simulate the whole simulation using the chosen alg: LP, or ALG_TOP (our alg).
        """
        self.alg              = alg
        self.sim_len_in_slots = sim_len_in_slots
        self.is_first_t = True # Will indicate that this is the first simulated time slot

        self.init_input_and_output_files()        
             
        print ('Simulating {}. ap file = {} cpu cap at leaf={}' .format (self.alg, self.ap_file_name, self.cpu_cap_at_leaf))
        self.stts     = sccs
        self.rsrc_aug = initial_rsrc_aug
        self.set_augmented_cpu_in_all_srvrs ()
        if (self.alg in ['ourAlg', 'wfit', 'ffit', 'cpvnf']):
            self.simulate_algs()
        elif (self.alg == 'opt'):
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

    def init_input_and_output_files (self):

        # open input file
        self.ap_file  = open ("../res/" + self.ap_file_name, "r")  

        # open output files, and print there initial comments
        if (VERBOSE_RES in self.verbose):
            self.init_res_file()
        if (VERBOSE_MOVED_RES in self.verbose):
            self.init_moved_res_file()
        if (VERBOSE_CRITICAL_RES in self.verbose):
            self.init_critical_res_file()
        if (VERBOSE_LOG in self.verbose):
            self.init_log_file()
        if (VERBOSE_MOB in self.verbose):
            self.mob_file_name   = "../res/" + self.ap_file_name.split(".")[0] + '.' + self.alg.split("_")[1] + '.mob.log'  
            self.mob_output_file =  open ('../res/' + self.mob_file_name,  "w") 
            printf (self.mob_output_file, '// results for running alg alg_top on input file {}\n' .format (self.ap_file_name))
            printf (self.mob_output_file, '// results for running alg alg_top on input file shorter.ap with {} leaves\n' .format (self.num_of_leaves))
            printf (self.mob_output_file, '// index i,j in the matrices below represent the total num of migs from lvl i to lvl j\n')
    
    def rd_t_line (self, time_str):
        """
        read the line describing a new time slot in the input file. Init some variables accordingly.
        """ 
        self.t = int(time_str)
        if (self.is_first_t):
            self.first_slot     = self.t
            self.final_slot_to_simulate = self.t + self.sim_len_in_slots 
        if (VERBOSE_ADD_LOG in self.verbose):
            printf (self.log_output_file, '\ntime = {}\n**************************************\n' .format (self.t))
        if (self.alg in ['ourAlg', 'wfit', 'ffit']): # once in a while, reshuffle the random ids of usrs, to mitigate unfairness due to tie-breaking by the ID, when sorting usrs 
            for usr in self.usrs:
                usr.calc_rand_id ()
        self.moved_usrs    = [] # rst the list of usrs who moved in this time slot 
        self.critical_usrs = [] # rst the list of usrs who are critical in this time slot

                    
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
        
        for line in self.ap_file: 
        
            # Ignore comments and emtpy lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue
        
            line = line.split ('\n')[0]
            splitted_line = line.split (" ")
        
            if (splitted_line[0] == "t"):
                self.rd_t_line (splitted_line[2])
                if (self.t >= self.final_slot_to_simulate):
                    self.post_processing ()
                    return 
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
                self.stts = self.alg_top(self.solve_by_plp)
                self.cur_st_params = self.d_vars
                self.is_first_t = False
                    
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
            if (self.alg in ['ourAlg']):
                self.G.nodes[s]['Hs']  = set() 
        
        for line in self.ap_file: 

            # Ignore comments lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue

            line = line.split ('\n')[0]
            splitted_line = line.split (" ")

            if (splitted_line[0] == "t"):
                self.rd_t_line (splitted_line[2])
                if (self.t >= self.final_slot_to_simulate):
                    self.post_processing ()
                    return 
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
                if (VERBOSE_ADD_LOG in self.verbose):
                    self.set_last_time()
                    printf (self.log_output_file, 't={}. beginning alg top\n' .format (self.t))
                    
                # solve the prob' using the requested alg'    
                if   (self.alg in ['ourAlg']):
                    self.stts = self.alg_top(self.bottom_up)
                elif (self.alg == 'ffit'):
                    self.stts = self.alg_top(self.first_fit)
                elif (self.alg == 'wfit'):
                    self.stts = self.alg_top(self.worst_fit)
                elif (self.alg == 'cpvnf'):
                    self.stts = self.alg_top(self.cpvnf)
                else:
                    print ('Sorry, alg {} that you selected is not supported' .format (self.alg))
                    exit ()
        
                if (self.stts == sccs and self.alg in ['ourAlg']):
                    self.push_up (self.critical_usrs) #If  it's after a reshuffling, critical usrs will be identical to self.usrs
                    # if (self.reshuffled):  
                    #     self.push_up (self.usrs)
                    # else:
                    #     self.push_up (self.critical_usrs) # if not reshuffled, push-up critical usrs only 
                
                self.print_sol_to_res_and_log ()
                if (self.stts!=sccs):
                    return # Currently, we don't try to further simulate, once alg fails to find a sol even for a single slot
                
                for usr in self.usrs: # The solution found at this time slot is the "cur_state" for next slot
                     usr.cur_s = usr.nxt_s
                self.is_first_t = False
        
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

        sim_len = float(self.t - self.first_slot)
        del (self.num_of_migs_in_slot[0]) # remove the mig' recorded in the first slot, which is irrelevant (corner case)
        printf (self.mob_output_file, '// avg num of usrs that moved per slot = {:.0f}\n'   .format (float(sum(self.num_of_moves_in_slot)) / sim_len))
        printf (self.mob_output_file, '// avg num of usrs who migrated per slot = {:.0f}\n' .format (float(sum(self.num_of_migs_in_slot)) / sim_len))
        avg_num_of_migs_to_from_per_slot = np.divide (self.mig_from_to_lvl, sim_len)
        for lvl_src in range (self.tree_height+1):
            for lvl_dst in range (self.tree_height+1):
                printf (self.mob_output_file, '{:.0f}\t' .format (avg_num_of_migs_to_from_per_slot[lvl_src][lvl_dst]))
            printf (self.mob_output_file, '\n')
        printf (self.mob_output_file, 'moves_in_slot = {}\n' .format (self.num_of_moves_in_slot))
        printf (self.mob_output_file, 'migs_in_slot = {}\n'  .format (self.num_of_migs_in_slot))

        # plot the mobility
        plt.figure()
        plt.title ('Migrations and mobility at each slot')
        plt.plot (range(int(sim_len)), self.num_of_moves_in_slot, label='Total vehicles moved to another cell [number/sec]', linestyle='None',  marker='o', markersize = 4)
        plt.plot (range(int(sim_len)), self.num_of_migs_in_slot, label='Total chains migrated to another server [number/sec]', linestyle='None',  marker='.', markersize = 4)
        plt.xlabel ('time [seconds, starting at 07:30]')
        plt.legend()
        plt.savefig ('../res/{}.mob.jpg' .format(self.ap_file_name.split('.')[0]))
        plt.clf()
        
    def cpvnf_reshuffle (self):
        """
        Run the cpvnf alg' when considering all existing usrs in the system (not only critical usrs).
        Returns sccs if found a feasible placement, fail otherwise
        """
        
        for usr in self.cpvnf_sort (self.usrs):
            if (self.cpvnf_place_usr (usr)!= sccs): 
                return fail
        return sccs
    
    def cpvnf (self):
        """
        Run the cpvnf alg'.
        Returns sccs if found a feasible placement, fail otherwise
        """
        for usr in self.cpvnf_sort (self.unplaced_usrs()):
            if (self.cpvnf_place_usr (usr)!= sccs): 
                self.rst_sol()
                return self.cpvnf_reshuffle()
        return sccs
    
    def cpvnf_place_usr (self, usr):
        """
        places a usr on the server minimizing the cost, among the delay-feasible available servers. 
        If no available delay-feasible server exists, returns fail. Else, returns sccs.
        """
        avail_delay_feasible_srvrs = [s for s in usr.S_u if self.s_has_sufic_avail_cpu_for_usr (s, usr)]
        
        if (avail_delay_feasible_srvrs == []): 
            return fail
        
        optional_costs = [self.chain_cost_homo (usr, self.G.nodes[s]['lvl']) for s in avail_delay_feasible_srvrs]
        self.place_usr_u_on_srvr_s (usr, avail_delay_feasible_srvrs[optional_costs.index (min (optional_costs))])
        return sccs
    
    def first_fit_reshuffle (self):
        """
        Run the first-fit alg' when considering all existing usrs in the system (not only critical usrs).
        Returns sccs if found a feasible placement, fail otherwise
        """
        for usr in self.first_fit_sort (self.usrs): 
            if (self.first_fit_place_usr(usr) != sccs):
                return fail
        return sccs    
    
    def first_fit (self):
        """
        Run the worst-fit alg'.
        Returns sccs if found a feasible placement, fail otherwise
        """

        for usr in self.first_fit_sort (self.unplaced_usrs()): 
            if (self.first_fit_place_usr (usr)!= sccs): # failed to find a feasible sol' when considering only the critical usrs
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
                self.place_usr_u_on_srvr_s (usr, s)
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
        unplaced_usrs = self.unplaced_usrs () 
        
        # first, handle the old, existing usrs, in an increasing order of the available cpu on the currently-hosting server
        for usr in sorted (list (filter (lambda usr : usr.cur_s!=-1 and usr.nxt_s==-1, unplaced_usrs)), 
                           key = lambda usr : (self.G.nodes[usr.cur_s]['a'], usr.rand_id)): 
            if (not(self.worst_fit_place_usr (usr))) : # Failed to migrate this usr)):
                self.rst_sol()
                return self.worst_fit_reshuffle()
                        
        # next, handle the new usrs, namely, that are not currently hosted on any server
        for usr in sorted (list (filter (lambda usr : usr.cur_s==-1 and usr.nxt_s==-1, unplaced_usrs),
                           key = lambda usr : usr.rand_id)): 
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
                self.place_usr_u_on_srvr_s(usr, self.G.nodes[s]['id'] )
                return True
        return False  
    
    
    # def update_RCs (self, R):
    #     if (R < 1):
    #         print ("Error: required rsrc aug < 1")
    #         exit ()
    #     cpu_cap_at_lvl0 = self.cpu_cap_at_lvl[0] * R
    #     for s in self.G.nodes:
    #         self.G.nodes[s]['RCs'] = self.
    
    def alg_top (self, placement_alg):
        """
        Binary search for a feasible sol that minimizes the resource augmentation R.
        The search is done by calling the placement_alg given as input.
        """
               
        # Try to solve the problem by changing the placement or CPU allocation only for the new / moved users
        self.stts = placement_alg()
        
        if (VERBOSE_LOG in self.verbose):
            self.print_sol_res_line (self.log_output_file, self.calc_alg_sol_cost(self.usrs))
        
        if (self.stts == sccs):
            return sccs
        
        # Now we know that the first run fail. For all the benchmarks, it means that they also made a reshuffle. 
        # However, bottom-up haven't tried a reshuffle yet. So, we give it a try now.
        if (self.alg in ['ourAlg']):
            self.rst_sol()
            self.stts = self.bottom_up()
            if (VERBOSE_LOG in self.verbose):
                printf (self.log_output_file, 'after reshuffle:\n')
                self.print_sol_to_log()
                self.print_sol_res_line (self.log_output_file, self.calc_alg_sol_cost(self.usrs))
            if (self.stts == sccs):
                return sccs

        # Now we know that the run failed. We will progress to a binary search for the required rsrc aug' only if we're requested by the self.verbose attribute. 
        if (VERBOSE_CALC_RSRC_AUG not in self.verbose):
            return self.stts

        print ('Sorry, VERBOSE_CALC_RSRC_AUG is currently unsupported')
        exit ()
        # # Couldn't solve the problem without additional rsrc aug --> begin a binary search for the amount of rsrc aug' needed.
        # if (VERBOSE_LOG in self.verbose):
        #     printf (self.log_output_file, 'Starting binary search:\n')
        #
        # self.rst_sol() # dis-allocate all users
        # self.critical_usrs = self.usrs # We will now consider reshuffling all usrs, so all usrs are considered critical.
        # self.CPUAll(self.usrs) 
        # max_R = self.max_R if VERBOSE_CALC_RSRC_AUG in self.verbose else self.calc_upr_bnd_rsrc_aug ()   
        #
        # # init cur RCs and a(s) to the number of available CPU in each server, assuming maximal rsrc aug'
        # cpu_cap_at_lvl = self.calc_cpu_capacities (math.ceil (max_R * self.cpu_cap_at_leaf)) # init the leaves' capacities. 
        # for s in self.G.nodes(): 
        #     self.G.nodes[s]['RCs'] = math.ceil (max_R * self.G.nodes[s]['cpu cap']) 
        #     self.G.nodes[s]['a']   = self.G.nodes[s]['RCs'] #currently-available rsrcs at server s  
        #
        # self.stts = placement_alg() 
        # if (self.stts != sccs):
        #     print ('did not find a feasible sol even with maximal rsrc aug')
        #     exit ()
        #
        # # Now we know that we found an initial feasible sol 
        #
        # ub = np.array([                self.G.nodes[s]['RCs']     for s in self.G.nodes()]) # upper-bnd on the (augmented) cpu cap' that may be required
        # lb = np.array([self.rsrc_aug * self.G.nodes[s]['cpu cap'] for s in self.G.nodes()]) # lower-bnd on the (augmented) cpu cap' that may be required
        #
        # while True: 
        #
        #     if ( np.array([ub[-1] <= lb[-1]+1 for s in self.G.nodes()], dtype='bool').all()): # Did the binary search converged?
        #         for s in self.G.nodes(): # Yep, so allocate this minimal found amount of rsrc aug to all servers
        #             self.G.nodes[s]['RCs'] = math.ceil (ub[s])  
        #         self.rst_sol()         # and re-solve the prob'
        #         if (placement_alg() == sccs): 
        #             self.update_rsrc_aug () # update the rsrc augmnetation to the lvl used in practice by the fesible sol found by this binary search
        #             return sccs
        #
        #         # We've got a prob', Houston
        #         print ('Error in the binary search: though I found a feasible sol, but actually this sol is not feasible')
        #         exit ()
        #
        #     # Now we know that the binary search haven't converged yet
        #     # Update the available capacity at each server according to the value of resource augmentation for this iteration            
        #     for s in self.G.nodes():
        #         self.G.nodes[s]['RCs'] = math.floor (0.5*(ub[s] + lb[s]))  
        #     self.rst_sol()
        #
        #     # Solve using the given placement alg'
        #
        #     self.stts = placement_alg()
        #     if (VERBOSE_LOG in self.verbose):
        #             self.print_sol_res_line (self.log_output_file, self.calc_alg_sol_cost(self.usrs))
        #
        #     if (self.stts == sccs):
        #         if (VERBOSE_ADD_LOG in self.verbose): 
        #             printf (self.log_output_file, 'In binary search IF\n')
        #             self.print_sol_to_log()
        #             if (VERBOSE_DEBUG in self.verbose):
        #                 self.check_cpu_usage_all_srvrs()
        #         ub = np.array([self.G.nodes[s]['RCs'] for s in self.G.nodes()])        
        #     else:
        #         lb = np.array([self.G.nodes[s]['RCs'] for s in self.G.nodes()])
    
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
                    self.place_usr_u_on_srvr_s (usr, s)
                elif (len (usr.B)-1 == lvl):
                    return fail
        return sccs

    def place_usr_u_on_srvr_s (self, u, s):
        """
        Place the given usr on the given srvr, and reduce s's available cpu accordingly.
        """
        u.nxt_s               = s
        u.lvl                 = self.G.nodes[s]['lvl']
        self.G.nodes[s]['a'] -= u.B[self.G.nodes[s]['lvl']]

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
    
            usr = usr_lp_c (id = int(tuple[0]), theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.randomize_target_delay(), C_u=self.uniform_Cu) # generate a new usr, which is assigned as "un-placed" yet (usr.lvl==-1)
            self.moved_usrs.append(usr)
            self.usrs.append (usr)
            self.CPUAll_single_usr (usr)
            self.update_S_u(usr, AP_id=int(tuple[1])) # update the list of delay-feasible servers for this usr

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
            
            # usr = usr_c (id                 = int(tuple[0]), # generate a new usr, which is assigned as "un-placed" yet (usr.lvl==-1) 
            #              theta_times_lambda = self.uniform_theta_times_lambda,
            #              target_delay       = self.randomize_target_delay(),
            #              C_u                = self.uniform_Cu)
            #

            if VERBOSE_FLAVORS in self.verbose:
                if (random.random() < self.prob_of_target_delay[0]):
                    usr = usr_c (id=int(tuple[0]), theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.target_delay[0], C_u=self.uniform_Cu)
                else:    
                    usr = usr_c (id=int(tuple[0]), theta_times_lambda=self.long_chain_theta_times_lambda, target_delay=self.target_delay[1], C_u=10*self.uniform_Cu)
            else:
                usr = usr_c (id=int(tuple[0]), theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.randomize_target_delay(), C_u=self.uniform_Cu)
                       
            self.moved_usrs.append (usr)
            self.critical_usrs.append(usr)

            self.usrs.append (usr)
            self.CPUAll_single_usr (usr)
            self.update_S_u(usr, AP_id=int(tuple[1])) # Update the list of delay-feasible servers for this usr 
            
            # Hs is the list of chains that may be located on each server while satisfying the delay constraint. Only some of the algs' use it
            if (self.alg in ['ourAlg']):
                for s in usr.S_u:
                    self.G.nodes[s]['Hs'].add(usr)                       
                    
    def update_S_u (self, usr, AP_id):
        """
        Update the Su (list of delay-feasible servers) of a given usr, given its current AP
        """
                    
        # self.check_AP_id (AP_id)
        usr.S_u = []
        s = self.ap2s[AP_id]
        usr.S_u.append (s)
        for lvl in (range (len(usr.B)-1)):
            s = self.parent_of(s)
            usr.S_u.append (s)
    
    def rd_old_usrs_line (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file.
        The input includes a list of tuples of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the tuples, the function assigns each chain to its relevant list of chains, Hs.  
        """
        if (line == []): # if the list of old users that moved is empty
            return
        
        splitted_line = self.parse_old_usrs_line(line)
        if (VERBOSE_MOB in self.verbose and self.t > self.first_slot):
            self.num_of_moves_in_slot.append (len (splitted_line)) # record the num of usrs who moved at this slot  

        for tuple in splitted_line:
            if (len(tuple) <= 1):
                break
            tuple = tuple.split("(")
            tuple = tuple[1].split (',')
            
            list_of_usr = list(filter (lambda usr : usr.id == int(tuple[0]), self.usrs))
            usr = list_of_usr[0]
            self.moved_usrs.append (usr)
            usr.cur_cpu = usr.B[usr.lvl]
            
            self.CPUAll_single_usr (usr) # update usr.B by the new requirements of this usr.

            self.update_S_u(usr, AP_id=int(tuple[1])) # Add this usr to the Hs of every server to which it belongs in its new location
                        
            # Check whether it's possible to comply with the delay constraint of this usr while staying in its cur location and keeping the CPU budget 
            if (usr.cur_s in usr.S_u and usr.cur_cpu <= usr.B[usr.lvl]): 

                # Yep: the delay constraint are satisfied also in the current placement. 
                # However, we have to update the 'Hs' (list of usrs in the respective subtree) of the servers in its current and next locations. 
                
                if (self.alg in ['ourAlg'] and usr.cur_s in usr.S_u and usr.cur_cpu <= usr.B[usr.lvl]): 
                    for s in [s for s in self.G.nodes() if usr in self.G.nodes[s]['Hs']]:
                        self.G.nodes[s]['Hs'].remove (usr) # Remove the usr from  'Hs' in all locations 
                    for s in usr.S_u:
                        self.G.nodes[s]['Hs'].add(usr)     # Add the usr only to the 'Hs' of its delay-feasible servers                          
                continue
            
            # Now we know that this is a critical usr, namely a user that needs more CPU and/or migration for satisfying its target delay constraint 
            # dis-place this user (mark it as having nor assigned level, neither assigned server), and free its assigned CPU
            self.critical_usrs.append(usr)
            self.rmv_usr_rsrcs (usr) # Free the resources of this user in its current place            
            usr.lvl   = -1
            usr.nxt_s = -1

            # if the currently-run alg' uses 'Hs', Add the usr to the relevant 'Hs'.
            # Hs is the set of relevant usrs) at each of its delay-feasible server
            if (self.alg in ['ourAlg']):
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
                print  ('Error at t={}: input file={}. Did not find old / reslotd usr {}' .format (self.t, self.ap_file_name, tuple[0]))
                exit ()
            usr    = list_of_usr[0]
            self.moved_usrs.append(usr)
            self.CPUAll_single_usr (usr)
            self.update_S_u(usr, AP_id=int(tuple[1])) # Add this usr to the Hs of every server to which it belongs at its new location
    
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
        #if (usr.cur_s != -1 and usr.lvl != -1): # If the 
        self.G.nodes[usr.cur_s]['a'] += usr.B[usr.lvl] # free the CPU units used by the user in the old location
        if (self.alg not in ['ourAlg']):
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
     
    def calc_total_cpu_rsrcs (self):
        return sum ([self.G.nodes[s]['cpu cap'] for s in self.G.nodes])
     

if __name__ == "__main__":
    
    # ap_file_name = 'shorter.ap' #'0830_0831_256aps.ap' 
    #
    # for alg in ['ourAlg', 'ffit', 'cpvnf']: #['cpvnf', 'ffit', 'ourAlg']: #, 'ffit', 'opt']: 
    #     my_simulator = SFC_mig_simulator (ap_file_name          = ap_file_name, 
    #                                       verbose               = [VERBOSE_LOG, VERBOSE_ADD_LOG, VERBOSE_ADD2_LOG], # defines which sanity checks are done during the simulation, and which outputs will be written   
    #                                       tree_height           = 2 if ap_file_name=='shorter.ap' else 4, 
    #                                       children_per_node     = 2 if ap_file_name=='shorter.ap' else 4,
    #                                       cpu_cap_at_leaf       = 330
    #                                       )
    #
    #     my_simulator.simulate (alg              = alg, # pick an algorithm from the list: ['opt', 'ourAlg', 'wfit', 'ffit'] 
    #                            sim_len_in_slots = 9999, 
    #                            initial_rsrc_aug = 1
    #                            ) 
    # exit ()

    ap_file_name = '0829_0830_1secs_256aps.ap' #'shorter.ap' #
    min_req_cap = 195 # for 0830:-0831 prob=0.3 it is: 195
    step        = min_req_cap*0.1
    
    for alg in ['ourAlg', 'ffit', 'cpvnf']: #['cpvnf', 'ffit', 'ourAlg']: #, 'ffit', 'opt']: 
        for cpu_cap in [int(round((min_req_cap + step*i))) for i in range (10, 20)]: 
            my_simulator = SFC_mig_simulator (ap_file_name          = ap_file_name, 
                                              verbose               = [VERBOSE_CRITICAL_RES],# defines which sanity checks are done during the simulation, and which outputs will be written   
                                              tree_height           = 2 if ap_file_name=='shorter.ap' else 4, 
                                              children_per_node     = 2 if ap_file_name=='shorter.ap' else 4,
                                              cpu_cap_at_leaf       = cpu_cap
                                              )
    
            my_simulator.simulate (alg              = alg,  
                                   sim_len_in_slots = 61, 
                                   initial_rsrc_aug = 1
                                   )     
