import numpy as np
import math
import time
import heapq
import pulp as plp
import matplotlib.pyplot as plt
import networkx as nx
import random
import pickle
from pathlib import Path

from usr_c    import usr_c    # class of the users of alg
from usr_lp_c import usr_lp_c # class of the users, when using LP
from decision_var_c import decision_var_c # class of the decision variables
from printf import printf, printmat
# from pandas.tests.arrays.sparse.test_array import test_first_fill_value_loc

# Levels of verbose / operation modes (which output is generated)
VERBOSE_DEBUG         = 0
VERBOSE_RES           = 1 # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_LOG           = 2 # Write to a ".log" file
VERBOSE_ADD_LOG       = 3 # Write to a detailed ".log" file
VERBOSE_ADD2_LOG      = 4
VERBOSE_MOB           = 5 # Write data about the mobility of usrs, and about the num of migrated chains per slot
VERBOSE_COST_COMP     = 6 # Print the cost of each component in the cost function
VERBOSE_CALC_RSRC_AUG = 7 # Use binary-search to calculate the minimal reseource augmentation required to find a sol. The calculation is done only during a single time slot, and doesn't guarantee that the whole trace would succeed with this rsrc aug 
VERBOSE_MOVED_RES     = 8 # calculate the cost incurred by the usrs who moved only 
VERBOSE_CRITICAL_RES  = 9 # calculate the cost incurred by the usrs who moved only 

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
    # Returns the parent of a given node (server)
    parent_of = lambda self, s : self.G.nodes[s]['prnt']

    # calculate the total cost of a solution by a placement algorithm (not by a solution by LP)
    calc_alg_sol_cost = lambda self, usrs: sum ([self.chain_cost_alg (usr, usr.lvl, slot_len=self.slot_len) for usr in usrs])
   
    # calculate the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    chain_cost_alg = lambda self, usr, lvl, slot_len=1 : slot_len * (self.link_cost_of_CLP_at_lvl[lvl] + self.cpu_cost_of_usr_at_lvl(usr, lvl)) + self.calc_mig_cost_CLP (usr, lvl)
    
    # Returns the CPU cost of locating a given user on a server at a given level 
    cpu_cost_of_usr_at_lvl = lambda self, usr, lvl : self.CPU_cost_at_lvl[lvl] * usr.B[lvl]
    
    # # calculate the migration cost incurred for a usr if placed on a given lvl, assuming a CLP (co-located placement), namely, the whole chain is placed on a single server
    calc_mig_cost_CLP = lambda self, usr, lvl : (usr.S_u[lvl] != usr.cur_s and usr.cur_s!=-1) * self.uniform_chain_mig_cost
          
    # Calculate the number of CPU units actually used in each server
    used_cpu_in_all_srvrs = lambda self: np.array ([self.G.nodes[s]['RCs'] - self.G.nodes[s]['a'] for s in self.G.nodes])      
          
    # calculate the proved upper bnd on the rsrc aug that bottomUp may need to find a feasible sol, given such a sol exists for the non-augmented prob'
    calc_upr_bnd_rsrc_aug = lambda self: np.max ([usr.C_u for usr in self.usrs]) / np.min ([np.min (usr.B) for usr in self.usrs])

    # Returns the server currently assigned for a given user 
    cur_server_of = lambda self, usr: usr.S_u[usr.lvl] 

    # Returns the total amount of cpu used by users at a certain server
    alg_used_cpu_in = lambda self, s: sum ([usr.B[usr.lvl] for usr in self.usrs if usr.nxt_s==s])

    # Given a server s, returns the total CPU currently allocated to usrs assigned to server s.  
    opt_used_cpu_in = lambda self, s: sum ( np.array ( [d_var.usr.B[self.G.nodes[s]['lvl']] * d_var.plp_var.value() for d_var in list (filter (lambda d_var : d_var.s == s, self.d_vars))]))
    
    # # Calculates the cost of locating a given user on a server at a given lvl.
    # # This is when when the current state may be non co-located-placement. That is, distinct VMs (or fractions) of the same chain may be found in several distinct server. 
    # chain_cost_of_usr_at_lvl_opt = lambda self, usr, lvl, slot_len=1: \
    #                 sum ([param.cur_st for param in list (filter (lambda param: param.usr == usr and param.s != usr.S_u[lvl], self.cur_st_params))]) * \
    #                 self.uniform_chain_mig_cost + self.cpu_cost_of_usr_at_lvl (usr, lvl) + self.link_cost_of_CLP_at_lvl[lvl]  
                    
    # Calculates the cost of a given decision variable, if the decision var is assigned 1.
    # This is when when the current state may be non co-located-placement. That is, distinct VMs (or fractions) of the same chain may be found in several distinct server. 
    d_var_cost = lambda self, d_var: self.d_var_cpu_cost (d_var) + self.d_var_link_cost (d_var) + self.d_var_mig_cost (d_var)

    # Calculates the cpu cost of a given decision variable, if the decision var is assigned 1.
    d_var_cpu_cost  = lambda self, d_var : self.cpu_cost_of_usr_at_lvl (d_var.usr, d_var.lvl) 

    # Calculates the link cost of a given decision variable, if the decision var is assigned 1.
    d_var_link_cost = lambda self, d_var : self.link_cost_of_CLP_at_lvl[d_var.lvl]
    
    # Print a solution for the problem to the output res file  
    print_sol_res_line = lambda self, output_file : self.print_sol_res_line_opt (output_file) if (self.mode == 'opt') else self.print_sol_res_line_alg (output_file)
    
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
    
    # Generate a new usr
    gen_new_usr = lambda self, usr_id : usr_c (id=usr_id, theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.pseudo_random_target_delay(usr_id), C_u=self.uniform_Cu)

    # Pseudo-randomize a target delay for a usr, based on its given usr_id
    pseudo_random_target_delay = lambda self, usr_id : self.target_delay[0] if ( ((usr_id % 10)/10) < self.prob_of_target_delay[0]) else self.target_delay[1]
    
    gen_res_file_name  = lambda self, mid_str : '../res/{}{}_p{}.res' .format (self.ap_file_name.split(".")[0], mid_str, self.prob_of_target_delay[0])

    # Returns a vector with the cpu capacities in each lvl of the tree, given the cpu cap at the leaf lvl
    calc_cpu_capacities = lambda self, cpu_cap_at_leaf : [2**(lvl)*cpu_cap_at_leaf for lvl in range (self.tree_height+1)] if self.use_exp_cpu_cap else np.array ([cpu_cap_at_leaf * (lvl+1) for lvl in range (self.tree_height+1)], dtype='uint16')

    # average the upper-bound and the lower-bound of the binary search, while rounding and converting to int
    avg_up_and_lb = lambda self, ub, lb : int (math.floor(ub+lb)/2)
    
    # Returns the total CPU cost in the current time slot ASSUMING that all usrs are already assigned (that is, for each usr, usr.lvl indicates a valid feasible level in the tree). 
    calc_cpu_cost_in_slot_alg = lambda self : sum ([self.CPU_cost_at_lvl[usr.lvl] * usr.B[usr.lvl] for usr in self.usrs])
    
    # Returns the total link cost in the current time slot ASSUMING that all usrs are already assigned (that is, for each usr, usr.lvl indicates a valid feasible level in the tree). 
    calc_link_cost_in_slot_alg = lambda self : sum ([self.link_cost_of_CLP_at_lvl[usr.lvl]          for usr in self.usrs])
    
    # Returns the total migration cost in the current time slot, when running an algo'
    calc_mig_cost_in_slot_alg = lambda self : self.uniform_chain_mig_cost * len(list (filter (lambda usr: usr.cur_s != -1 and usr.cur_s != usr.nxt_s, self.usrs)))


    # Print a solution for the problem to the output res file when the solver is an algorithm (not an LP solver)  

    # Returns the total cpu cost in the current time slot, according to the LP solution
    calc_cpu_cost_in_slot_opt = lambda self : sum ([self.d_var_cpu_cost(d_var)  * d_var.plp_var.value() for d_var in self.d_vars])
    
    # Returns the total link cost in the current time slot, according to the LP solution
    calc_link_cost_in_slot_opt = lambda self : sum ([self.d_var_link_cost(d_var) * d_var.plp_var.value() for d_var in self.d_vars])
    
    # Returns the total mig' cost in the current time slot, according to the LP solution
    calc_mig_cost_in_slot_opt = lambda self : sum ([self.d_var_mig_cost(d_var)  * d_var.plp_var.value() for d_var in self.d_vars])

    # Returns a string, detailing the sim' parameters (time, amount of CPU at leaves, probability of RT app' at leaf, status of the solution)
    settings_str = lambda self : 't{}_{}_cpu{}_p{:.1f}_stts{}' .format(
                              self.t, self.mode, self.G.nodes[len (self.G.nodes)-1]['RCs'], self.prob_of_target_delay[0], self.stts)

    # Print a solution for the problem to the output res file when the solver is an LP solver  
    print_sol_res_line_opt = lambda self, output_file: printf (output_file, '{} | {}\n' .format(
            self.settings_str(), 
            self.sol_cost_str (cpu_cost  = self.calc_cpu_cost_in_slot_opt(),
                               link_cost = self.calc_link_cost_in_slot_opt()
                               ,mig_cost  = self.calc_mig_cost_in_slot_opt())))

    # Print a solution for the problem to the output res file when the solver is an algorithm (not an LP solver)  
    print_sol_res_line_alg = lambda self, output_file: printf (output_file, '{} | {}\n' .format(
            self.settings_str(), 
            self.sol_cost_str (cpu_cost  = self.calc_cpu_cost_in_slot_alg(),
                               link_cost = self.calc_link_cost_in_slot_alg()
                               ,mig_cost  = self.calc_mig_cost_in_slot_alg())))

    augmented_cpu_cap_at_leaf = lambda self: self.G.nodes[len (self.G.nodes)-1]['RCs']

    # Returns a string, detailing the sim' costs' components
    def sol_cost_str (self, cpu_cost, link_cost, mig_cost):
        tot_cost = cpu_cost + link_cost + mig_cost 
        return 'cpu_cost={:.2f} | link_cost={:.2f} | mig_cost={:.2f} | tot_cost={:.2f} | ratio=[{:.2f},{:.2f},{:.2f}]' .format(
                cpu_cost, link_cost, mig_cost, tot_cost, cpu_cost/tot_cost, link_cost/tot_cost, mig_cost/tot_cost)  

    def d_var_mig_cost (self, d_var): 
        if (d_var.usr.is_new): # No mig' cost for a new usr
            return 0
        my_param_list = list (filter (lambda param : param.usr==d_var.usr and param.s==d_var.s, self.cur_st_params)) # my_param_list is the list of cur-st param' assigning this usr to this server
        if (len(my_param_list)==0): #In the cur st, no part of this usr was assigned to this server.
            return self.uniform_chain_mig_cost # Assigning 1 to this d_var implies migrating the whole usr to this server.                                                                                                                             
        if (len(my_param_list)>1):
            print ('Error. Inaal Abuck')
            exit ()
        return (1 - my_param_list[0].plp_var.value()) * self.uniform_chain_mig_cost  
    
    def set_RCs_and_a (self, aug_cpu_capacity_at_lvl):
        """"
        given the (augmented) cpu cap' at each lvl, assign each server its 'RCs' (augmented CPU cap vals); and initialise 'a' (the amount of available CPU) to 'RCs' (the augmented CPU cap).
        """
        # self.print_netw()
        for s in self.G.nodes:
            self.G.nodes[s]['RCs'] = aug_cpu_capacity_at_lvl[self.G.nodes[s]['lvl']]
            self.G.nodes[s]['a'  ] = aug_cpu_capacity_at_lvl[self.G.nodes[s]['lvl']]
        
    def print_netw (self):
        """
        Used for debugging
        """
        debug_file = open ('../res/debug.txt', 'w')
        for s in self.G.nodes:
            printf (debug_file, 's={} ' .format(s))
            printf (debug_file, 'lvl={}\n' .format(self.G.nodes[s]['lvl']))


    
    def print_sol_cost_components (self):
        """
        prints to a file statistics about the cost of each component in the cost function (cpu, link, and migration). 
        """
        del (self.cpu_cost_in_slot            [0])
        del (self.link_cost_in_slot           [0])
        del (self.num_of_migs_in_slot         [0])
        del (self.num_of_critical_usrs_in_slot[0])
        del (self.num_of_moved_usrs_in_slot   [0])
        self.cost_of_migs_in_slot = self.uniform_chain_mig_cost * np.array (self.num_of_migs_in_slot)  
        
        printf (self.detailed_cost_comp_output_file, '//t = {}\n//*******************************\n' .format (self.t))
        printf (self.detailed_cost_comp_output_file, 'cpu_cost_in_slot={}\nlink_cost_in_slot={}\nnum_of_migs_in_slot={}\ncost_of_migs_in_slot={}\n' .format
               (self.cpu_cost_in_slot, self.link_cost_in_slot, self.num_of_migs_in_slot, self.cost_of_migs_in_slot))
        printf (self.detailed_cost_comp_output_file, 'num_of_moved_usrs_in_slot={}\nnum_of_critical_usrs_in_slot={}\n' .format (
                self.num_of_moved_usrs_in_slot, self.num_of_critical_usrs_in_slot))
        
        total_cpu_cost    = sum(self.cpu_cost_in_slot)
        total_link_cost   = sum(self.link_cost_in_slot)
        total_num_of_migs = sum(self.num_of_migs_in_slot)
        total_mig_cost    = total_num_of_migs * self.uniform_chain_mig_cost
        total_cost        = sum ([total_cpu_cost, total_link_cost, total_mig_cost])
        # total_cost        = np.sum (np.array ([total_cpu_cost, total_link_cost, total_mig_cost]))
        
        printf (self.cost_comp_output_file, '\nap_file={}\t| t = {}\t|total cpu cost={}\t| total_link_cost_={}\t| num_of_migs={}\t| total_mig_cost={}\n' .format
               (self.ap_file_name, self.t, total_cpu_cost, total_link_cost, total_num_of_migs, total_mig_cost))

        if (total_cost == 0):
            return
        
        total_cpu_cost_ratio  = total_cpu_cost  / total_cost 
        total_link_cost_ratio = total_link_cost / total_cost 
        total_mig_cost_ratio  = total_mig_cost  / total_cost 
        
        # cpu_cost_ratio  = [self.cpu_cost_in_slot[t]/total_cost[t]  for t in range(len(total_cost))]
        # link_cost_ratio = [self.link_cost_in_slot[t]/total_cost[t] for t in range(len(total_cost))]
        # mig_cost_ratio  = [self.total_mig_cost_in_slot[t]/total_cost[t]  for t in range(len(total_cost))]
        # printf (self.cost_comp_output_file, 'cpu_cost_ratio = {}\nlink_cost_ratio = {}\nmig_cost_ratio = {}\n'.format (
        #     cpu_cost_ratio, link_cost_ratio, mig_cost_ratio))
            
        printf (self.cost_comp_output_file, 'total_cpu_cost_ratio={:.4f} | total_link_cost_ratio={:.3f} | total_mig_cost_ratio={:.3f}\n' .format (
                total_cpu_cost_ratio, total_link_cost_ratio, total_mig_cost_ratio))
            
    def calc_sol_cost_components (self):
        """
        Calculates and keeps the cost of each component in the cost function (cpu, link, and migration). 
        """
        self.cpu_cost_in_slot.            append (sum ([self.CPU_cost_at_lvl[usr.lvl] * usr.B[usr.lvl] for usr in self.usrs]))
        self.link_cost_in_slot.           append (sum ([self.link_cost_of_CLP_at_lvl[usr.lvl]          for usr in self.usrs]))
        self.num_of_migs_in_slot.         append (len(list (filter (lambda usr: usr.cur_s != -1 and usr.cur_s != usr.nxt_s, self.usrs))))
        self.num_of_critical_usrs_in_slot.append (len(self.critical_usrs))
        self.num_of_moved_usrs_in_slot   .append (len(self.moved_usrs))
     
    def print_sol_to_res_and_log (self):
        """
        Print to the res, log, and debug files the solution and/or additional info, according to the verobse level, indicated by the variable self.verbose.
        """
        if (VERBOSE_RES in self.verbose): # and (not(self.is_first_t))): # in the first slot there're no migrations (only placement), and hence the cost is wrong, and we ignore it.
            self.print_sol_res_line (output_file=self.res_file)
        if (VERBOSE_MOVED_RES in self.verbose and (not(self.is_first_t))): # in the first slot there're no migrations (only placement), and hence the cost is wrong, and we ignore it.
            self.print_sol_res_line (output_file=self.moved_res_file)
        if (VERBOSE_CRITICAL_RES in self.verbose and (not(self.is_first_t))): # in the first slot there're no migrations (only placement), and hence the cost is wrong, and we ignore it.
            self.print_sol_res_line (output_file=self.critical_res_file)
        elif (VERBOSE_COST_COMP in self.verbose):
            self.calc_sol_cost_components()

        if (VERBOSE_LOG in self.verbose): # Commented-out, because this is already printed during alg_top()
            self.print_sol_res_line (output_file=self.log_output_file)
        if (VERBOSE_ADD_LOG in self.verbose):
            printf (self.log_output_file, '\nSolved in {:.3f} [sec]\n' .format (time.time() - self.last_rt)) 
            self.print_sol_to_log_alg()
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
                    cost += self.d_var_cost (d_var)
        return cost

    def solve_by_plp (self):
        """
        Find an optimal fractional solution using Python's pulp LP library.
        pulp library can use commercial tools (e.g., Gurobi, Cplex) to efficiently solve the prob'.
        """
        model = plp.LpProblem(name="SFC_mig", sense=plp.LpMinimize) # init a "model" that will hold the problem, objective, and constraints 
        
        # init the decision vars
        self.d_vars  = [] # decision variables  
        obj_func     = [] # objective function
        d_var_id           = 0  # cntr for the id of the decision variables 
        for usr in self.usrs:
            single_place_const = [] # will hold constraint assuring that each chain is placed in a single server
            
            for lvl in range(len(usr.B)): # will check all delay-feasible servers for this user
                plp_var = plp.LpVariable (lowBound=0, upBound=1, name='x_{}' .format (d_var_id))
                d_var = decision_var_c (d_var_id=d_var_id, usr=usr, lvl=lvl, s=usr.S_u[lvl], plp_var=plp_var) # generate a decision var, containing the lp var + details about its meaning 
                self.d_vars.append (d_var)
                single_place_const += plp_var
                d_var_id += 1
                # if (VERBOSE_MOVED_RES in self.verbose and not (self.is_first_t) and usr not in self.moved_usrs): 
                #     continue # In this mode, starting from the 2nd slot, the obj' func' should consider only the users who moved 
                # if (VERBOSE_CRITICAL_RES in self.verbose and not (self.is_first_t) and usr not in self.critical_usrs): 
                #     continue # In this mode, starting from the 2nd slot, the obj' func' should consider only the users who are critical
                obj_func += self.d_var_cost (d_var) * plp_var # add the cost of this decision var to the objective func
            model += (single_place_const == 1) # demand that each chain is placed in a single server
        model += obj_func

        # Generate CPU capacity constraints
        for s in self.G.nodes():
            cpu_cap_const = []
            for d_var in list (filter (lambda item : item.s == s, self.d_vars)): # for every decision variable meaning placing a chain on this server 
                cpu_cap_const += (d_var.usr.B[d_var.lvl] * d_var.plp_var) # Add the overall cpu of this chain, if located on s
            if (cpu_cap_const != []):
                model += (cpu_cap_const <= self.G.nodes[s]['RCs']) 

        model.solve(plp.PULP_CBC_CMD(msg=0)) # solve the model, without printing output; to solve it using another solver: solve(GLPK(msg = 0))
        
        
        self.stts = model.status       
        # print the solution to the output, according to the desired self.verbose level
        if (VERBOSE_RES in self.verbose):
            self.print_sol_res_line_opt (output_file=self.res_file)
            printf (self.res_file, '// status = {}\n' .format (plp.LpStatus[model.status])) 
            if (model.status != 1): 
                print ('Opt failed. Status={}' .format (plp.LpStatus[model.status]))
                print (model, file=open("../res/plp_model.txt", "w")) #$$$$
                exit () #$$$
            sol_cost_by_obj_func = model.objective.value()
            sol_cost_direct = sum ([self.d_var_cpu_cost(d_var)  * d_var.plp_var.value() for d_var in self.d_vars]) + \
                              sum ([self.d_var_link_cost(d_var) * d_var.plp_var.value() for d_var in self.d_vars]) + \
                              sum ([self.d_var_mig_cost(d_var)  * d_var.plp_var.value() for d_var in self.d_vars])
            if (abs (sol_cost_by_obj_func - sol_cost_direct) > 0.1): 
                print ('Error: obj func value of sol={} while sol cost={}' .format (sol_cost_by_obj_func, sol_cost_direct))
                exit ()
            
        # print the solution to the output, according to the desired self.verbose level
        if (VERBOSE_LOG in self.verbose):
            self.print_sol_res_line_opt (output_file=self.log_output_file)
            printf (self.log_output_file, 'tot_cost = {:.2f}\n' .format (model.objective.value())) 
            
        if (VERBOSE_MOVED_RES in self.verbose and not(self.is_first_t)):
            self.print_sol_res_line (output_file=self.moved_res_file)
        if (model.status == 1): # successfully solved
            if (VERBOSE_LOG in self.verbose):            
                self.print_sol_to_log_opt ()
                printf (self.log_output_file, '\nSuccessfully solved in {:.3f} [sec]\n' .format (time.time() - self.last_rt))
        else:
            if (VERBOSE_RES in self.verbose):
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
        
        self.res_file_name = self.gen_res_file_name (mid_str = ('_opt' if self.mode=='opt' else '') ) + ('.expCPU.res' if (self.use_exp_cpu_cost) else '') + ('.expCPU2.res' if (self.use_exp_cpu_cap) else '')
        
        if Path(self.res_file_name).is_file(): # does this res file already exist?
            self.res_file =  open (self.res_file_name,  "a")
        else:
            self.res_file =  open (self.res_file_name,  "w")
            self.print_res_file_prefix (self.res_file)

    def init_moved_res_file (self):
        """
        Open the res file for writing, as follows:
        If a res file with the relevant name already exists - open it for appending.
        Else, open a new res file, and write to it comment header lines, explaining the file's format  
        """
        self.moved_res_file_name = self.gen_res_file_name (mid_str = ('_moved_opt' if self.mode=='opt' else '_moved')) 
        
        if Path('../res/' + self.moved_res_file_name).is_file(): # does this res file already exist?
            self.moved_res_file =  open ('../res/' + self.moved_res_file_name,  "a")
        else:
            self.moved_res_file =  open ('../res/' + self.moved_res_file_name,  "w")
            self.print_res_file_prefix (self.moved_res_file)

    def init_critical_res_file (self):
        """
        Open the res file for writing, as follows:
        If a res file with the relevant name already exists - open it for appending.
        Else, open a new res file, and write to it comment header lines, explaining the file's format  
        """
        self.critical_res_file_name = self.gen_res_file_name (mid_str = ('_critical_opt' if self.mode=='opt' else '_critical')) 
        
        if Path('../res/' + self.critical_res_file_name).is_file(): # does this res file already exist?
            self.critical_res_file =  open ('../res/' + self.critical_res_file_name,  "a")
        else:
            self.critical_res_file =  open ('../res/' + self.critical_res_file_name,  "w")
            self.print_res_file_prefix (self.critical_res_file)

    def print_res_file_prefix (self, res_file):
        """
        Print several header lines, indicating the format and so, to an output res file.
        """
        printf (res_file, '// format: t{T}.{Mode}.cpu{C}.stts{s} | cpu_cost=... | link_cost=... | mig_cost=... | cost=... | ratio=[c,l,m] c, where\n// T is the slot cnt (read from the input file)\n')
        printf (res_file, '// Mode is the algorithm / solver used.\n// C is the num of CPU units used in the leaf\n')
        printf (res_file, '// [c,l,m] are the ratio of the cpu, link, and mig cost out of the total cost, resp.\n\n') 

    def init_log_file (self, overwrite = True):
        """
        Open the log file for writing and write initial comments lines on it
        """
        self.log_file_name = "../res/" + self.ap_file_name.split(".")[0] + '.' + self.mode + ('.detailed' if VERBOSE_ADD_LOG in self.verbose else '') +'.log'  
        self.log_output_file =  open ('../res/' + self.log_file_name,  "w") 
        printf (self.log_output_file, '//RCs = augmented capacity of server s\n' )

    def print_sol_to_log_opt (self):
        """
        print a lp fractional solution to the output log file 
        """
        for s in self.G.nodes():
            printf (self.log_output_file, 's{} RCs={} used cpu={}\n' .format (s, self.G.nodes[s]['RCs'], self.opt_used_cpu_in (s) ))

        if (VERBOSE_ADD_LOG in self.verbose): 
            for d_var in self.d_vars: 
                if d_var.plp_var.value() > 0:
                    printf (self.log_output_file, '\nu {} lvl {:.0f} s {:.0f} val {:.2f}' .format(
                           d_var.usr.id, d_var.lvl, d_var.s, d_var.plp_var.value()))            

    def print_sol_to_log_alg (self):
        """
        print the solution found by alg' for the mig' problem to the output log file 
        """
        for s in self.G.nodes():
            used_cpu_in_s = self.alg_used_cpu_in (s)
            chains_in_s   = [usr.id for usr in self.usrs if usr.nxt_s==s]
            if (used_cpu_in_s > 0):  
                printf (self.log_output_file, 's{} : Rcs={}, a={}, used cpu={:.0f}, num_of_chains={}' .format (
                        s,
                        self.G.nodes[s]['RCs'],
                        self.G.nodes[s]['a'],
                        used_cpu_in_s,
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
        if (self.alg_used_cpu_in(s) + self.G.nodes[s]['a'] != self.G.nodes[s]['RCs']):
            printf (self.log_output_file, 'Error in calculating the cpu utilization of s{}: used_cpu = {}, a={}, Rcs={}' .format 
                    (s, self.alg_used_cpu_in(s), self.G.nodes[s]['a'], self.G.nodes[s]['RCs']))
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
                if (self.G.nodes[usr.S_u[lvl]]['a'] >= usr.B[lvl] and self.chain_cost_alg(usr, lvl) < self.chain_cost_alg(usr, usr.lvl)): # if there's enough available space to move u to level lvl, and this would reduce cost
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
        if (VERBOSE_ADD_LOG in self.verbose):
            printf (self.log_output_file, 'after push-up:\n')

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
    
       
    def gen_parameterized_antloc_tree (self, ap2cell_file_name=''):
        """
        Generate a parameterized tree with specified height and children-per-non-leaf-node. 
        Add leaves for each AP, and prune sub-trees that don't have any descended APs.
        """
        
        # Generate a complete balanced tree. If needed, later we will fix it according to the concrete distribution of APs.
        # Note: after fixing, the tree's height will be 1 level more, as we'll be adding a level for the APs below the current tree.
        self.G                 = nx.generators.classic.balanced_tree (r=self.children_per_node, h=self.tree_height) # Generate a tree of height h where each node has r children.
        
        self.G = self.G.to_directed()
        root             = 0 # In networkx, the ID of the root is 0

        # levelize the tree (assuming a balanced tree)
        self.cell2s           = [] # Will contain a least translating the cell number to the ID of the co-located server. 
        self.num_of_leaves    = 0
        for s in self.G.nodes(): # for every server
            self.G.nodes[s]['id'] = s
            if self.G.out_degree(s)==1 and self.G.in_degree(s)==1: # is it a leaf?
                self.G.nodes[s]['lvl']   = 0 # Yep --> its lvl is 0
                self.cell2s.append (s) 
                self.num_of_leaves += 1

        num_fo_nodes_b4_pruning = self.G.number_of_nodes() # We will later add servers, with increasing IDs
        self.rd_ap2cell_file(ap2cell_file_name)
        
        # nx.draw (self.G, with_labels=True) 
        # plt.show()
        
        # Remove all the servers in cells that don't contain AP
        shortest_path    = nx.shortest_path(self.G)
        for s in range (1, len(self.G.nodes())):
            self.G.nodes[s]['prnt']   = shortest_path[s][root][1]
            self.G.nodes[s]['nChild'] = self.children_per_node # Num of children of this server (==node)
        for cell in range (self.num_of_leaves): # for each cell 
            APs_of_this_cell = list (filter (lambda item : item['cell']==cell, self.APs))
            if (len(APs_of_this_cell)==0): # No APs at this cell
                s2remove = self.cell2s[cell]
                prnt = self.G.nodes[s2remove]['prnt'] # Find the parent of the node-to-remove
                self.G.remove_node(self.cell2s[cell]) # Remove the leaf server handling this cell
                self.num_of_leaves -= 1 # We have just removed one leaf server
                self.cell2s[cell] = -1 # Now, this cell isn't associated with any server
                self.G.nodes[prnt]['nChild'] -= 1 # Dec. the # of children of the parent

                # Remove all parents that don't have any child
                if (self.G.nodes[prnt]['nChild']==0):
                    self.G.remove_node(prnt) # Remove the leaf server handling this cell
                
        # nx.draw (self.G, with_labels=True) 
        # plt.show()

        # Garbage collection: condense all the remaining nodes (==servers), so that they'll have sequencing IDs, starting from 0
        server_ids_to_recycle = set ([s for s in range (num_fo_nodes_b4_pruning) if (s not in self.G.nodes())])
        for s in range(num_fo_nodes_b4_pruning): 
            if (len(server_ids_to_recycle)==0): # No more ids to recycle
                break
            if (s in server_ids_to_recycle): # No server with this id
                continue
            id_to_recycle = min (server_ids_to_recycle)
            if (s > id_to_recycle): # Can decrement the current ID of s, by modifying it to be the current id to recycle
                self.G = nx.relabel_nodes(self.G, {s : id_to_recycle})
                server_ids_to_recycle.remove(id_to_recycle)
                server_ids_to_recycle.add (s)
                
                # Update self.cell2s accordingly
                my_cell_as_list = [i for i, x in enumerate(self.cell2s) if x == s]
                if (len (my_cell_as_list)>0): # This was indeed a server of cell that removed 
                    self.cell2s[my_cell_as_list[0]] = id_to_recycle
                       
        # Add new leaves for the APs below each cell. 
        # To keep the IDs of leaves the greatest in the tree, only after we finished removing all the useless cells.
        self.num_of_srvrs = len(self.G.nodes())
        for ap in [ap for ap in self.APs]: #(filter (lambda item : item['cell']==cell, self.APs)): # for each ap belonging to this cell
            ap['s'] = len(self.G.nodes())   # We'll shortly add a server for this AP, so the id of this AP will be current number of servers+1.
            self.G.add_node (ap['s']) # Add a server co-located with this AP
            self.G.nodes[ap['s']]['lvl'] = 0 # The level of the newly added node is 0 (it's a leaf)
            self.G.add_edge (ap['s'], self.cell2s[ap['cell']]) # Add an edge from the newly added server, to the server handling the cell of this AP, ...
            self.G.add_edge (self.cell2s[ap['cell']], ap['s']) # And vice versa

        self.num_of_leaves    = len(self.APs) # Each AP is co-located with a leaf server
        self.ap2s             = [ap['s'] for ap in self.APs] # Will contain a least translating the AP number (==leaf #) to the ID of the co-located server. 

        # Update the tree height's by the changes made, and set 
        shortest_path    = nx.shortest_path(self.G)
        self.tree_height = len (shortest_path[self.ap2s[0]][root]) - 1
        
        self.CPU_cost_at_lvl   = [2**(self.tree_height - lvl) for lvl in range (self.tree_height+1)] if self.use_exp_cpu_cost else [(1 + self.tree_height - lvl) for lvl in range (self.tree_height+1)]
        self.link_cost_at_lvl  = self.uniform_link_cost * np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of locating a full chain at level i
        self.link_delay_at_lvl = 2 * np.ones (self.tree_height) #self.link_delay_at_lvl[i] is the return delay when locating a full chain at level i 
        
        # overall link cost and link capacity of a Single-Server Placement of a chain at each lvl
        self.link_cost_of_CLP_at_lvl  = [2 * sum([self.link_cost_at_lvl[i]  for i in range (lvl)]) for lvl in range (self.tree_height+1)]
        self.link_delay_of_CLP_at_lvl = [2 * sum([self.link_delay_at_lvl[i] for i in range (lvl)]) for lvl in range (self.tree_height+1)] 

        # Levelize the updated tree
        for leaf in self.ap2s:
            
            for lvl in range (self.tree_height):
                self.G.nodes[shortest_path[leaf][root][lvl]]['lvl']     = np.uint8(lvl) # assume here a balanced tree
        self.G.nodes[0]['lvl'] = self.tree_height

        for s in self.G.nodes(): # for every server
            self.G.nodes[s]['id'] = s
        # nx.draw (self.G, with_labels=True) 
        # plt.show()

        # Find parents of each node (except of the root)
        for s in range (1, len(self.G.nodes())):
            self.G.nodes[s]['prnt'] = shortest_path[s][root][1]

    def gen_parameterized_full_tree (self):
        """
        Generate a parameterized full tree with specified height and children-per-non-leaf-node. 
        """
        
        # Generate a complete balanced tree. If needed, later we will fix it according to the concrete distribution of APs.
        self.G                 = nx.generators.classic.balanced_tree (r=self.children_per_node, h=self.tree_height) # Generate a tree of height h where each node has r children.
        self.CPU_cost_at_lvl   = [2**(self.tree_height - lvl) for lvl in range (self.tree_height+1)] if self.use_exp_cpu_cost else [(1 + self.tree_height - lvl) for lvl in range (self.tree_height+1)]
        self.link_cost_at_lvl  = self.uniform_link_cost * np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of locating a full chain at level i
        self.link_delay_at_lvl = 2 * np.ones (self.tree_height) #self.link_delay_at_lvl[i] is the return delay when locating a full chain at level i 
        
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
                    # # The lines below are for case one likes to vary the link and cpu costs of distinct servers on the same level. 
                    # self.G.nodes[shortest_path[s][root][lvl]]['cpu cost']  = self.CPU_cost_at_lvl[lvl]                
                    # self.G.nodes[shortest_path[s][root][lvl]]['link cost'] = self.link_cost_of_CLP_at_lvl[lvl]
                    # # Iterate over all children of node i
                    # for n in self.G.neighbors(i):
                    #     if (n > i):
                    #         print (n)


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

    def __init__ (self, ap_file_name = 'shorter.ap', verbose = [], tree_height = 4, children_per_node = 4, prob_of_target_delay=0.3, ap2cell_file_name='', 
                  use_exp_cpu_cost=True, use_exp_cpu_cap=False):
        """
        """

        self.verbose                    = verbose

        self.ap_file_name               = ap_file_name #input file containing the APs of all users along the simulation
        self.use_exp_cpu_cost           = use_exp_cpu_cost
        self.use_exp_cpu_cap            = use_exp_cpu_cap
        
        # Network parameters
        self.tree_height                = 2 if (self.ap_file_name=='shorter.ap') else tree_height 
        self.children_per_node          = 2 if (self.ap_file_name=='shorter.ap') else children_per_node
        self.uniform_vm_mig_cost        = 200
        self.Lmax                       = 0
        self.uniform_Tpd                = 2
        self.uniform_link_cost          = 3
        self.uniform_theta_times_lambda = [2, 10, 2] # "1" here means 100MHz
        self.uniform_chain_mig_cost     = self.uniform_vm_mig_cost * len (self.uniform_theta_times_lambda)
        self.uniform_Cu                 = 20 
        self.target_delay               = [10, 100] # in [ms], lowest to highest
        self.prob_of_target_delay       = [prob_of_target_delay]  
        self.warned_about_too_large_ap  = False
        self.usrs                       = []
        random.seed                     (42) # Use a fixed pseudo-number seed 
        
        # Init output files
        if (VERBOSE_COST_COMP in self.verbose):
            self.init_cost_comp () 
        if (VERBOSE_DEBUG in self.verbose):
            self.debug_file = open ('../res/debug.txt', 'w') 
        if (VERBOSE_MOB in self.verbose):
            self.num_of_moved_usrs_in_slot         = [] # self.num_of_moved_usrs_in_slot[t] will hold the num of usrs who moved at slot t.   
            self.num_of_migs_in_slot          = [] # self.num_of_migs[t] will hold the num of chains assigned to migrate in slot t.
            self.num_of_critical_usrs_in_slot = [] 
            self.mig_from_to_lvl      = np.zeros ([self.tree_height+1, self.tree_height+1], dtype='int') # self.mig_from_to_lvl[i][j] will hold the num of migrations from server in lvl i to server in lvl j, along the sim

        self.ap2cell_file_name = ap2cell_file_name
        if (self.ap2cell_file_name==''):
            self.gen_parameterized_full_tree  ()
        else:
            self.gen_parameterized_antloc_tree (self.ap2cell_file_name)
        self.delay_const_sanity_check()

    def init_cost_comp (self):
        """
        Open the output file to which we will write the cost of each component in the sim
        """
        self.cost_comp_file_name            = '../res/cost_comp.res'  
        self.detailed_comp_cost_file_name   = '../res/{}_detailed_cost_comp.res' .format (self.ap_file_name)  
        self.cost_comp_output_file          =  open ('../res/' + self.cost_comp_file_name,           "a") 
        self.detailed_cost_comp_output_file =  open ('../res/' + self.detailed_comp_cost_file_name,  "w") 
        
        self.cpu_cost_in_slot             = []
        self.link_cost_in_slot            = []
        self.num_of_migs_in_slot          = []
        self.num_of_critical_usrs_in_slot = []
        self.num_of_moved_usrs_in_slot    = []

    def delay_const_sanity_check (self):
        """
        Sanity check for the usr parameters' feasibility.
        """
        usr = self.gen_new_usr (usr_id=0)
        self.CPUAll_single_usr (usr) 
        if (len(usr.B)==0):
            print ('Error: cannot satisfy delay constraints of usr {}, even on a leaf. theta_times_lambda={}, target_delay ={}' .format (
                    usr.id, usr.theta_times_lambda, usr.target_delay))
            exit ()       

    def simulate (self, mode, prob_of_target_delay=None, cpu_cap_at_leaf=516, sim_len_in_slots=99999, slot_to_dump=float('inf')): 
        
        """
        Simulate the whole simulation using the chosen alg: LP (for finding an optimal fractional solution), or an algorithm (either our alg, or a benchmark alg' - e.g., first-fit, worst-fit).
        Return value: if the simulation succeeded (found a feasible solution for every time slot during the sim') - return the cpu capacity used in a leaf node.
        Else, return None. 
        """
        
        random.seed                     (42) # Use a fixed pseudo-number seed 
        self.usrs                       = []
        self.slot_to_dump               = slot_to_dump
        self.delay_const_sanity_check()

        self.cpu_cap_at_leaf = 30 if self.ap_file_name == 'shorter.ap' else cpu_cap_at_leaf 
        self.cpu_cap_at_lvl  = self.calc_cpu_capacities (self.cpu_cap_at_leaf)
        self.set_RCs_and_a  (aug_cpu_capacity_at_lvl=self.cpu_cap_at_lvl) # initially, there is no rsrc augmentation, and the capacity and currently-available cpu of each server is exactly its CPU capacity.

        self.usrs                       = [] 

        if (prob_of_target_delay!=None): # If the caller stated a different prob_of_target_delayk, assing it
            self.prob_of_target_delay = [prob_of_target_delay]  

        self.mode              = mode
        self.max_R = 1.2 if (self.mode == 'opt') else 1.3 # Set the upper limit of the binary search. Running opt is much slower, and usually doesn't require much rsrc aug', and therefore we may set for it a lower value.  
        self.sim_len_in_slots = sim_len_in_slots
        self.is_first_t = True # Will indicate that this is the first simulated time slot

        self.init_input_and_output_files()        
                     
        print ('Simulating {}. ap file = {} cpu cap at leaf={}. prob or RT={}' .format (self.mode, self.ap_file_name, self.cpu_cap_at_leaf, self.prob_of_target_delay[0]))
        self.stts     = sccs

        # extract the slot len from the input '.ap' file name
        slot_len_str = self.ap_file_name.split('secs')
        if (len (slot_len_str) > 1): # the input filename contains the string 'secs'
            self.slot_len = int(slot_len_str[0].split('_')[-1])
        else:
            self.slot_len = 1 # assume that by default, slot len is 1
        
        if (self.mode in ['ourAlg', 'wfit', 'ffit', 'cpvnf']):
            self.simulate_algs()
        elif (self.mode == 'opt'):
            self.simulate_lp ();
        else:
            print ('Sorry, mode {} that you selected is not supported' .format (self.mode))
            exit ()
        return self.augmented_cpu_cap_at_leaf () if (self.stts==sccs) else None

    def init_input_and_output_files (self):
        """
        Initialize "self" variables for the file-handlers of input and output files.
        Write several comments explanation files as headers to the output files. 
        """

        # open input file
        self.ap_file  = open ("../res/" + self.ap_file_name, "r")  

        # open output files, and print there initial comments
        if (VERBOSE_RES in self.verbose):
            self.init_res_file()
        if VERBOSE_CALC_RSRC_AUG in self.verbose:
            if (self.use_exp_cpu_cost and self.use_exp_cpu_cap): 
                self.rsrc_aug_file_name = '../res/rsrc_aug_by_RT_prob_exp_cpu^2.res' 
            elif (self.use_exp_cpu_cost):
                self.rsrc_aug_file_name = '../res/rsrc_aug_by_RT_prob_exp_cpu_cost_{}.res' .format (self.ap2cell_file_name)
            else: 
                self.rsrc_aug_file_name = '../res/rsrc_aug_by_RT_prob.res'

            if Path (self.rsrc_aug_file_name).is_file(): # does this res file already exist?
                self.rsrc_aug_file =  open (self.rsrc_aug_file_name,  "a")
            else:
                self.rsrc_aug_file =  open (self.rsrc_aug_file_name,  "w")
            
        if (VERBOSE_MOVED_RES in self.verbose):
            self.init_moved_res_file()
        if (VERBOSE_CRITICAL_RES in self.verbose):
            self.init_critical_res_file()
        if (VERBOSE_LOG in self.verbose):
            self.init_log_file()
        if (VERBOSE_MOB in self.verbose):
            self.mob_file_name   = "../res/" + self.ap_file_name.split(".")[0] + '.' + self.mode.split("_")[1] + '.mob.log'  
            self.mob_output_file =  open ('../res/' + self.mob_file_name,  "w") 
            printf (self.mob_output_file, '// results for running alg_top on input file {}\n' .format (self.ap_file_name))
            printf (self.mob_output_file, '// results for running alg_top on input file shorter.ap with {} leaves\n' .format (self.num_of_leaves))
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
        if (self.mode in ['ourAlg', 'wfit', 'ffit']): # reshuffle the random ids of usrs, to mitigate unfairness due to tie-breaking by the ID, when sorting usrs 
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
                if (self.t >= self.final_slot_to_simulate): # finished the desired simulation time
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
                if (VERBOSE_LOG in self.verbose):
                    self.last_rt = time.time ()
                self.stts = self.alg_top(self.solve_by_plp) # call the top-level alg' that solves the problem, possibly by binary-search, using iterative calls to the given solver (plp LP solver, in our case).
                self.cur_st_params = self.d_vars.copy () #All the decision vars will be referred as "cur st parameters" in the next time slot 
                self.prepare_sim_to_next_slot()
        self.post_processing ()
        
    def rmv_usr_from_all_Hs (self, usr):
        """
        Remove a given usr from all the "Hs" sets of servers.
        An Hs set of server s is the set of usrs for which s is delay feasible.
        Namely, this is the list of usrs that may be placed on s while maintaining the delay constraints. 
        """
        for s in [s for s in self.G.nodes() if usr in self.G.nodes[s]['Hs']]:
            self.G.nodes[s]['Hs'].remove (usr)
            
        
        for s in self.G.nodes():
            if usr in self.G.nodes[s]['Hs']:
                print ('found a bug')
                exit ()
                    
    def prepare_sim_to_next_slot (self):
        """
        To be called after finished handling and solving the problem for the current slot. 
        """
        self.is_first_t = False  # The next slot is surely not the first slot
        self.prev_t     = self.t # save the time of the current time slot as "previous" for the next time slot to come. 
                    
    def rd_usrs_that_left_line (self, splitted_line):
        """
        Read a line of usrs that left from an '.ap' file, and remove them from the data bases.
        """
        if (self.mode != 'ourAlg'):
            for usr in list (filter (lambda usr : usr.id in [int(usr_id) for usr_id in splitted_line[1:] if usr_id!=''], self.usrs)):
                self.usrs.remove (usr)                    

        # Now we know that the alg' running is ourAlg. In this case, we should also remove the usr from all the Hs sets.
        # For a given server s, Hs set is the set of all usrs that may be located on s while satisfying the delay constraints.
        # Namely, for a server s, this is the set of all usrs, for whihc s is delay-feasible.   
        for usr in list (filter (lambda usr : usr.id in [int(usr_id) for usr_id in splitted_line[1:] if usr_id!=''], self.usrs)):
            self.rmv_usr_from_all_Hs(usr)
            self.usrs.remove (usr)                    
    
    def simulate_algs (self):
        """
        Simulate the whole simulation, using an algorithm (NOT a LP solver).
        At each time step:
        - Read and parse from an input ".ap" file the AP cells of each user who moved. 
        - solve the problem using alg_top (our alg). 
        - Write outputs results and/or logs to files.
        - Update self.stts according to the solution's stts.
        - update cur_st = nxt_st
        """
        
        # reset Hs     
        if (self.mode in ['ourAlg']):
            for s in self.G.nodes():
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
                self.rd_usrs_that_left_line (splitted_line)
                continue
        
            elif (splitted_line[0] == "new_usrs:"):              
                self.rd_new_usrs_line (splitted_line[1:])
            elif (splitted_line[0] == "old_usrs:"):  
                self.rd_old_usrs_line (splitted_line[1:])
                if (VERBOSE_ADD_LOG in self.verbose):
                    self.last_rt = time.time ()
                    printf (self.log_output_file, 't={}. beginning alg top\n' .format (self.t))
                    
                # if (self.t == self.slot_to_dump): #$$$
                #     self.dump_usrs_to_ap_file()
                
                # solve the problem using the requested alg'    
                if   (self.mode in ['ourAlg']):
                    self.stts = self.alg_top(self.bottom_up)
                elif (self.mode == 'ffit'):
                    self.stts = self.alg_top(self.first_fit)
                elif (self.mode == 'wfit'):
                    self.stts = self.alg_top(self.worst_fit)
                elif (self.mode == 'cpvnf'):
                    self.stts = self.alg_top(self.cpvnf)                    
                if (self.mode in ['ourAlg'] and self.stts==sccs): # if we ran our alg' (bottom-up), perform now the final step, of push-up 
                    self.push_up (self.usrs) if self.reshuffled else self.push_up(self.critical_usrs) 
                
                self.print_sol_to_res_and_log ()
                if (self.stts!=sccs):
                    return # Once an alg' fails to find a sol even for a single slot, we don't try to further simulate, 
                
                for usr in self.usrs: # The solution found at this time slot is the "cur_state" for next slot
                    usr.cur_s = usr.nxt_s
                self.prepare_sim_to_next_slot()
        
        self.post_processing()
        
    def  post_processing (self):
        """
        Organize, writes and plots the simulation results, after the simulation is done
        """
        if (VERBOSE_MOB in self.verbose):
            self.print_mob ()        
        if (VERBOSE_COST_COMP in self.verbose):
            self.print_sol_cost_components ()
        if (VERBOSE_CALC_RSRC_AUG in self.verbose):
            print ('augmented cpu cap at leaf={}' .format (self.augmented_cpu_cap_at_leaf()))
            self.print_sol_res_line (self.rsrc_aug_file)
    
    def print_mob (self):
        """
        print statistics about the number of usrs who moved, and the num of migrations between every two levels in the tree.
        """

        sim_len = float(self.t - self.first_slot)
        del (self.num_of_migs_in_slot[0]) # remove the mig' recorded in the first slot, which is irrelevant (corner case)
        printf (self.mob_output_file, '// avg num of usrs that moved per slot = {:.0f}\n'   .format (float(sum(self.num_of_moved_usrs_in_slot)) / sim_len))
        printf (self.mob_output_file, '// avg num of usrs who migrated per slot = {:.0f}\n' .format (float(sum(self.num_of_migs_in_slot)) / sim_len))
        avg_num_of_migs_to_from_per_slot = np.divide (self.mig_from_to_lvl, sim_len)
        for lvl_src in range (self.tree_height+1):
            for lvl_dst in range (self.tree_height+1):
                printf (self.mob_output_file, '{:.0f}\t' .format (avg_num_of_migs_to_from_per_slot[lvl_src][lvl_dst]))
            printf (self.mob_output_file, '\n')
        printf (self.mob_output_file, 'moves_in_slot = {}\n' .format (self.num_of_moved_usrs_in_slot))
        printf (self.mob_output_file, 'migs_in_slot = {}\n'  .format (self.num_of_migs_in_slot))

        # plot the mobility
        plt.figure()
        plt.title ('Migrations and mobility at each slot')
        plt.plot (range(int(sim_len)), self.num_of_moved_usrs_in_slot, label='Total vehicles moved to another cell [number/sec]', linestyle='None',  marker='o', markersize = 4)
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
                self.reshuffled = True
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
        
        optional_costs = [self.chain_cost_alg (usr, self.G.nodes[s]['lvl']) for s in avail_delay_feasible_srvrs]
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
                self.reshuffled = True
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
        if (VERBOSE_LOG in self.verbose): 
            printf (self.log_output_file, '\nfailed to locate user {} on S_u={}\n' .format (usr.id, usr.S_u)) 
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
                self.reshuffled = True
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
        Write to the log and to the res file that the solver (either an algorithm, or an LP solver) did not succeed to place all the usrs
        """
        if (VERBOSE_RES in self.verbose):
            printf (self.res_file, 't{:.0f}.{}.stts2' .format (self.t, self.mode))
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_output_file, 't{:.0f}.{}.stts2' .format (self.t, self.mode))

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
    
    
    def alg_top (self, placement_alg):
        """
        Binary search for a feasible sol that minimizes the resource augmentation R.
        The search is done by calling the placement_alg given as input.
        """
               
        # Try to solve the problem by changing the placement or CPU allocation only for the new / moved users
        self.reshuffled = False # As the first run, considering only critical chains, failed, we'll now perform an additional run of bottom-up, while doing a reshuffle (namely, considering all usrs).

        self.stts = placement_alg()
        
        if (VERBOSE_LOG in self.verbose):
            self.print_sol_res_line (self.log_output_file)
            self.print_sol_to_log_alg()
            
        # if (self.t == 30563): #$$$$
        #     self.init_log_file(overwrite=True)
        #     self.print_sol_res_line (self.log_output_file)
        #     self.verbose.append (VERBOSE_LOG)
        #     self.verbose.append (VERBOSE_ADD_LOG)
        #     self.verbose.append (VERBOSE_ADD2_LOG)
        #     self.last_rt = time.time () 
        #     self.print_sol_to_log_alg()
        
        if (self.stts == sccs):
            return sccs
        
        # Now we know that the first run fail. For all the benchmarks, it means that they also made a reshuffle. 
        # However, bottom-up haven't tried a reshuffle yet. So, we give it a try now.
        if (self.mode in ['ourAlg']):
            self.rst_sol()
            self.reshuffled = True # In this run, we'll perform a reshuffle (namely, considering all usrs).
            if (VERBOSE_RES in self.verbose):
                printf (self.res_file, '// reshuffling\n')
            self.stts = self.bottom_up()
            if (VERBOSE_ADD_LOG in self.verbose):
                printf (self.log_output_file, 'after reshuffle:\n')
                self.print_sol_to_log_alg()
                self.print_sol_res_line (self.log_output_file)
            if (self.stts == sccs):
                return sccs

         # Now we know that the first run, without additional resource augmentation, failed. We will progress to a binary search for the required rsrc aug' only if we're requested by the self.verbose attribute. 
        if (VERBOSE_CALC_RSRC_AUG not in self.verbose):
            return self.stts

        # Couldn't solve the problem without additional rsrc aug --> begin a binary search for the amount of rsrc aug' needed.
        if (VERBOSE_LOG in self.verbose):
            printf (self.log_output_file, 'Starting binary search:\n')
        
        # Init the the lower bound (lb) and the upper bound (up) for the binary search.
        # lb, ub are defined the minimal, maximal CPU capacity at the leaves.
        ub = math.ceil (self.max_R * self.cpu_cap_at_leaf) # Maximum allowed capacity at a leaf server
        lb = self.G.nodes[len(self.G.nodes)-1]['RCs'] # Current (possibly augmented) capacity at a leaf server  
        
        # init cur 'RCs' and 'a' of each server to the number of available CPU in each server, assuming maximal rsrc aug'        
        self.set_RCs_and_a (self.calc_cpu_capacities (cpu_cap_at_leaf = ub)) 
        self.rst_sol() # dis-allocate all users
        
        self.stts = placement_alg() 
        if (self.stts != sccs):
            print ('did not find a feasible sol even with maximal rsrc aug')
            exit ()
        
        # Now we know that there exists a feasible sol when using the upper-bound resource aug' 
        
        while True: 
        
            if (ub <= lb+1): # the difference between the lb and the ub is at most 1
                # To avoid corner-case rounding problems, make a last run of the placement alg' with this (upper bound) value 
                self.set_RCs_and_a (self.calc_cpu_capacities (cpu_cap_at_leaf = ub))         
                self.rst_sol()         
                if (placement_alg() == sccs):
                    if (VERBOSE_ADD_LOG in self.verbose):
                        self.print_sol_res_line (self.log_output_file)
                        printf (self.log_output_file, 'successfully finished binary search\n') 
                    return sccs
        
                # We've got a prob', Houston
                print ('Error in the binary search: though I found a feasible sol, but actually this sol is not feasible')
                exit ()
        
            # Now we know that the binary search haven't converged yet
            # Update the augmented capacity, and the available capacity at each server according to the value of resource augmentation for this iteration
            cur_cpu_at_leaf = self.avg_up_and_lb (ub=ub, lb=lb)            
            self.set_RCs_and_a (self.calc_cpu_capacities (cpu_cap_at_leaf = cur_cpu_at_leaf)) # update the (augmented) CPU cap in all servers
            self.rst_sol()
            self.stts = placement_alg()
            if (VERBOSE_LOG in self.verbose):
                    self.print_sol_res_line (self.log_output_file)
        
            if (self.stts == sccs):
                if (VERBOSE_ADD_LOG in self.verbose): 
                    printf (self.log_output_file, 'In binary search IF\n')
                    self.print_sol_to_log_alg()
                    if (VERBOSE_DEBUG in self.verbose):
                        self.check_cpu_usage_all_srvrs()
                ub = cur_cpu_at_leaf       
            else:
                lb = cur_cpu_at_leaf
    
    def bottom_up (self):
        """
        Our bottom-up alg'. 
        Assigns all self.usrs that weren't assigned yet (either new usrs, or old usrs that moved, and now they don't satisfy the target delay).
        Looks for a feasible sol'.
        Returns sccs if a feasible sol was found, fail else.
        """      
        if (self.t == self.slot_to_dump): #$$$
            self.dump_state_to_log_file()  
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
        The input includes a list of usr_entries of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the usr_entries, the function assigns each chain to its relevant list of chains, Hs.  
        """
    
        if (line ==[]):
            return # no new users
    
        splitted_line = line[0].split ("\n")[0].split (")")
    
        for usr_entry in splitted_line:
            if (len(usr_entry) <= 1):
                break
            usr_entry = usr_entry.split("(")
            usr_entry = usr_entry[1].split (',')
    
            usr = usr_lp_c (int(usr_entry[0]), theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.pseudo_random_target_delay (int(usr_entry[0])), C_u=self.uniform_Cu) # generate a new usr, which is assigned as "un-placed" yet (usr.lvl==-1)
            usr.is_new = True
            self.moved_usrs.append(usr)
            self.usrs.append (usr)
            self.CPUAll_single_usr (usr)
            self.update_S_u(usr, AP_id=int(usr_entry[1])) # update the list of delay-feasible servers for this usr

    def rd_new_usrs_line (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file.
        The input includes a list of usr_entries of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the usr_entries, the function assigns each chain to its relevant list of chains, Hs.  
        """
        
        if (line ==[]):
            return # no new users

        splitted_line = line[0].split ("\n")[0].split (")")

        for usr_entry in splitted_line:
            if (len(usr_entry) <= 1):
                break
            usr_entry = usr_entry.split("(")
            usr_entry = usr_entry[1].split (',')
            
            usr = self.gen_new_usr (usr_id=int(usr_entry[0]))
            
            # usr0 = usr_c (id=0, theta_times_lambda=[1, 10, 1], target_delay=5, C_u=100) #$$$
            # usr1 = usr_c (id=1, theta_times_lambda=[1, 10, 1], target_delay=100, C_u=100)
            # self.CPUAll_single_usr (usr0)
            # self.CPUAll_single_usr (usr1)
            # print ('u0.B={}, u1.B={}' .format (usr0.B, usr1.B))
            # exit ()
                       
            self.moved_usrs.append (usr)
            self.critical_usrs.append(usr)

            self.usrs.append (usr)
            self.CPUAll_single_usr (usr) 
            self.update_S_u(usr, AP_id=int(usr_entry[1])) # Update the list of delay-feasible servers for this usr 
            
            # Hs is the list of chains that may be located on each server while satisfying the delay constraint. Only some of the algs' use it
            if (self.mode in ['ourAlg']):
                for s in usr.S_u:
                    self.G.nodes[s]['Hs'].add(usr)                       
                    
    def update_S_u (self, usr, AP_id):
        """
        Update the S_u (list of delay-feasible servers) of a given usr, given the id of its current AP (Access Point server)
        """                    
        usr.S_u = []
        s       = self.ap2s[AP_id]
        usr.S_u.append (s)
        for lvl in (range (len(usr.B)-1)):
            s = self.parent_of(s)
            usr.S_u.append (s)
    
    def rd_old_usrs_line (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file.
        The input includes a list of usr_entries of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the usr entries, the function assigns each chain to its relevant list of chains, Hs.  
        """
        if (line == []): # if the list of old users that moved is empty
            return
        
        splitted_line = self.parse_old_usrs_line(line)
        if (VERBOSE_MOB in self.verbose and self.t > self.first_slot):
            self.num_of_moved_usrs_in_slot.append (len (splitted_line)) # record the num of usrs who moved at this slot  

        for usr_entry in splitted_line:
            if (len(usr_entry) <= 1):
                break
            usr_entry = usr_entry.split("(")
            usr_entry = usr_entry[1].split (',')
            
            list_of_usr = list(filter (lambda usr : usr.id == int(usr_entry[0]), self.usrs))
            usr = list_of_usr[0]
            self.moved_usrs.append (usr)
            usr.cur_cpu = usr.B[usr.lvl]
            
            self.CPUAll_single_usr (usr) # update usr.B by the new requirements of this usr.

            self.update_S_u(usr, AP_id=int(usr_entry[1])) # Add this usr to the Hs of every server to which it belongs in its new location
                        
            # #$$$$ MAYBE THE BUG IS HERE? COMMENTING THESE LINES
            # # Check whether it's possible to comply with the delay constraint of this usr while staying in its cur location and keeping the CPU budget 
            # if (usr.cur_s in usr.S_u and usr.cur_cpu <= usr.B[usr.lvl]): 
            #
            #     # Yep: the delay constraint are satisfied also in the current placement. 
            #     # However, we have to update the 'Hs' (list of usrs in the respective subtree) of the servers in its current and next locations. 
            #
            #     if (self.mode in ['ourAlg'] and usr.cur_s in usr.S_u and usr.cur_cpu <= usr.B[usr.lvl]): 
            #         for s in [s for s in self.G.nodes() if usr in self.G.nodes[s]['Hs']]:
            #             self.G.nodes[s]['Hs'].remove (usr) # Remove the usr from  'Hs' in all locations 
            #         for s in usr.S_u:
            #             self.G.nodes[s]['Hs'].add(usr)     # Add the usr only to the 'Hs' of its delay-feasible servers                          
            #     continue
            
            # Now we know that this is a critical usr, namely a user that needs more CPU and/or migration for satisfying its target delay constraint 
            # dis-place this user (mark it as having nor assigned level, neither assigned server), and free its assigned CPU
            self.critical_usrs.append(usr)
            self.rmv_usr_rsrcs (usr) # Free the resources of this user in its current place            
            usr.lvl   = -1
            usr.nxt_s = -1

            # if the currently-run alg' uses 'Hs', Add the usr to the relevant 'Hs'.
            # Hs is the set of relevant usrs) at each of its delay-feasible server
            if (self.mode in ['ourAlg']):
                for s in usr.S_u:
                    self.G.nodes[s]['Hs'].add(usr)               # Removing the usr from 'Hs' in its previous location is done by self.rmv_usr_rsrcs.

    def rd_old_usrs_line_lp (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".ap" file, when using a LP solver for the problem.
        The input includes a list of usr_entries of the format (usr,ap), where "usr" is the user id, and "ap" is its current access point (AP).
        After reading the usr_entries, the function assigns each chain to its relevant list of chains, Hs.  
        """
        if (line == []): # if the list of old users that moved is empty
            return
        
        splitted_line = self.parse_old_usrs_line(line)

        for usr_entry in splitted_line:
            if (len(usr_entry) <= 1):
                break
            usr_entry = usr_entry.split("(")
            usr_entry = usr_entry[1].split (',')
            
            list_of_usr = list(filter (lambda usr : usr.id == int(usr_entry[0]), self.usrs))
            if (len(list_of_usr) == 0):
                print  ('Error at t={}: input file={}. Did not find old / reslotd usr {}' .format (self.t, self.ap_file_name, usr_entry[0]))
                exit ()
            usr    = list_of_usr[0]
            usr.is_new = False
            self.moved_usrs.append(usr)
            self.CPUAll_single_usr (usr)
            self.update_S_u(usr, AP_id=int(usr_entry[1])) # Add this usr to the Hs of every server to which it belongs at its new location
    
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
        if (self.mode not in ['ourAlg']):
            return 
        
        # Now we know that the alg' that currently runs uses 'Hs'. Hence, we have to clean them.
        self.rmv_usr_from_all_Hs (usr)
                    
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
    
    def dump_state_to_log_file (self):

        log_file = open ('../res/{}.t{}.log' .format (self.ap_file_name, self.t), 'w')

        for s in [s for s in self.G.nodes()]:
            printf (log_file, 's{}: Hs={}\n' .format (s, sorted (list ([usr.id for usr in self.G.nodes[s]['Hs']])))) 

    def dump_usrs_to_ap_file (self):

        output_ap_file = open ('../res/Lux_dump_{}.t{}.ap' .format (self.mode, self.t), 'w')
        printf (output_ap_file, '// Dumping ap file={} ap2cell_file_name={}' .format (self.ap_file_name, self.ap2cell_file_name))
        printf (output_ap_file, '// File format:\n//for each time slot:\n')
        printf (output_ap_file, 't = {}\n' .format(self.t))
        printf (output_ap_file, 'new_usrs: ' .format(self.t))

        for usr in sorted (self.usrs, key = lambda usr : usr.id):
            self.CPUAll_single_usr (usr) 
            printf (output_ap_file, '({},{})' .format (usr.id, usr.S_u[0])) # S_u is the list of delay-feasible servers for that usr. S_u[0] is the leaf server (namely, the PoA) out of them.
        printf (output_ap_file, '\nold_usrs:\n')
    
    def run_dump_usrs (self, slot_to_dump):
        """
        Write the current AP association of all users at a given time slot to an .ap file
        """
        random.seed (42) # Use a fixed pseudo-number seed 
        self.usrs   = []
        self.mode   = 'bypass'
        
        self.is_first_t = True # Will indicate that this is the first simulated time slot
        
        self.input_ap_file = open ("../res/" + self.ap_file_name, "r")  
        for line in self.input_ap_file: 
        
            # Ignore comments lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue
        
            line = line.split ('\n')[0]
            splitted_line = line.split (" ")
        
            if (splitted_line[0] == "t"):
                self.t = int(splitted_line[2])
        
            elif (splitted_line[0] == "usrs_that_left:"):
                self.rd_usrs_that_left_line (splitted_line)
                continue
                    
                continue
        
            elif (splitted_line[0] == "new_usrs:"):              
                new_usrs_line = splitted_line[1:]
                if (new_usrs_line ==[]):
                    continue # no new users
        
                splitted_line = new_usrs_line[0].split ("\n")[0].split (")")
        
                for usr_entry in splitted_line:
                    if (len(usr_entry) <= 1):
                        break
                    usr_entry = usr_entry.split("(")
                    usr_entry = usr_entry[1].split (',')
        
                    usr = self.gen_new_usr (usr_id=int(usr_entry[0]))
                    usr.is_new = True
                    self.CPUAll_single_usr (usr) 
                    self.update_S_u(usr, AP_id=int(usr_entry[1])) # Update the list of delay-feasible servers for this usr 
                    self.usrs.append (usr)
                    
            elif (splitted_line[0] == "old_usrs:"):  
                old_usrs_line = splitted_line[1:]
        
                if (old_usrs_line == []): # if the list of old users that moved is empty
                    return
        
                splitted_line = self.parse_old_usrs_line(old_usrs_line)
                for usr_entry in splitted_line:
                    if (len(usr_entry) <= 1):
                        break
                    usr_entry = usr_entry.split("(")
                    usr_entry = usr_entry[1].split (',')    
                    list_of_usr = list(filter (lambda usr : usr.id == int(usr_entry[0]), self.usrs))
                    usr = list_of_usr[0]
                    self.CPUAll_single_usr (usr) # update usr.B by the new requirements of this usr.
                    self.update_S_u(usr, AP_id=int(usr_entry[1])) # Add this usr to the Hs of every server to which it belongs in its new location

                # if (self.t == slot_to_dump): #$$$
                #     self.dump_usrs_to_ap_file() 
     
    def rd_ap2cell_file (self, ap2cell_file_name):
        """
        Parse an ap2cell file.
        An ap2cell file contains a list of APs, and, for each AP, the cell in which it is located.
        """
        
        self.APs = []
        input_file = open ('../res/' + ap2cell_file_name, 'r')
        
        for line in input_file:
    
            if (line == "\n" or line.split ("//")[0] == ""):
                continue

            line = line.split ("\n")[0]
            splitted_line = line.split ()
            self.APs.append ({'ap' : int(splitted_line[0]), 'cell' : int(splitted_line[1])})
    
    def binary_search_along_full_trace (self, output_file, mode, cpu_cap_at_leaf=200, prob_of_target_delay=0.3, sim_len_in_slots=61):
        """
        Binary-search for the minimal rsrce aug' required for finding a feasible sol.
        A run is considered "sccs" iff it successfully found solutions during the whole trace.
        Returns the lowest cpu cap' at the leaf server which still allows finding a feasible sol. 
        """ 
        res = self.simulate (mode = mode, cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots)
        if (res != None): # found a feasible solution without a binary search 
            print ('wo binary search, cpu_cap_at_leaf={}' .format (cpu_cap_at_leaf))
            self.print_sol_res_line (output_file)
            return res

        lb = cpu_cap_at_leaf
        ub = cpu_cap_at_leaf * self.max_R
        while True:
            if (ub <= lb+1): # the difference between the lb and the ub is at most 1
                cpu_cap_at_leaf = ub
                self.simulate (mode = mode, cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots)
                self.print_sol_res_line (output_file)
                break

            cpu_cap_at_leaf = self.avg_up_and_lb(ub, lb)
            if (self.simulate (mode = mode, cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots) != None): # Sccs
                # self.print_sol_res_line (output_file) #$$$
                ub = cpu_cap_at_leaf
            else: 
                lb = cpu_cap_at_leaf
                # self.print_sol_res_line (output_file) #$$$
        print ('after binary search, cpu_cap_at_leaf={}' .format (cpu_cap_at_leaf))
        return cpu_cap_at_leaf
    
    def run_prob_of_RT_sim (self):
        """
        Run a simulation where the probability of a RT application varies. 
        Output the minimal resource augmentation required by each alg', and the cost obtained, and the cost obtained at each time slot.
        """
        
        output_file      = open ('../res/RT_prob_sim_{}_{}.deter_usr_id.res' .format (ap2cell_file_name, ap_file_name), 'a') 
        mode             = 'ourAlg'    
        sim_len_in_slots = 24

        self.binary_search_along_full_trace (output_file=output_file, mode='ourAlg', cpu_cap_at_leaf=171, prob_of_target_delay=0.5, sim_len_in_slots=sim_len_in_slots)
        # for mode in ['ourAlg']:
        #     for prob_of_target_delay in [0.4, 0.6, 0.7, 0.8, 0.9, 1]:
        #         self.binary_search_along_full_trace (output_file=output_file, mode=mode, cpu_cap_at_leaf=160, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots) 
        
        # for mode in ['cpvnf']:
        #     cpu_cap_at_leaf = 450  #Initial cpu cap at the leaf server
        #     for prob_of_target_delay in [1]: # [0.1*i for i in range (11)]:
        #         # cpu_cap_at_leaf = 
        #         self.binary_search_along_full_trace(output_file=output_file, mode=mode, cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots)
    
        # cpu_cap_at_leaf = 160  #Initial cpu cap at the leaf server
        # for prob_of_target_delay in [0.1*i for i in range (11)]:
        #     cpu_cap_at_leaf = self.binary_search_along_full_trace(output_file=output_file, mode='ourAlg', cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots)

        # cpu_cap_at_leaf = 160  #Initial cpu cap at the leaf server
        # for prob_of_target_delay in [0.1*i for i in range (11)]:
        #     cpu_cap_at_leaf = self.binary_search_along_full_trace(output_file=output_file, mode='opt', cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots)
    
    
#######################################################################################################################################
# Functions that are not part of the class
#######################################################################################################################################


def run_cost_by_rsrc ():
    """
    Run a simulation where the amount of resources varies. 
    Output the cost obtained at each time slot.
    """
    ap_file_name = '0829_0830_1secs_256aps.ap' #'shorter.ap' #
    min_req_cpu = {'opt' : 160, 'ourAlg' : 165, 'ffit' : 341, 'cpvnf' : 407}

    my_simulator = SFC_mig_simulator (ap_file_name=ap_file_name, verbose=[VERBOSE_RES], ap2cell_file_name='Lux.center.post.antloc_256cells.ap2cell')
                
    # for cpu_cap_at_leaf in [int (min_req_cpu['opt']*(1 + 0.1*i)) for i in range(5, 21)]: # simulate for opt's min cpu * [100%, 110%, ...]
    #     my_simulator.simulate (mode = 'opt', cpu_cap_at_leaf=cpu_cap_at_leaf, sim_len_in_slots = 61)
        
    # for cpu_cap_at_leaf in [165, 341, 407]: # simulate for opt's min cpu * [100%, 110%, ...]
    #     my_simulator.simulate (mode = 'opt', cpu_cap_at_leaf=cpu_cap_at_leaf, sim_len_in_slots = 61)
        
    

    # for mode in ['ffit', 'cpvnf', 'ourAlg']: #, 'ourAlg', 'opt']:
    #     for cpu_cap_at_leaf in [int (min_req_cpu['opt']*(1 + 0.1*i)) for i in range(16, 21)]: # simulate for opt's min cpu * [100%, 110%, ... 250%]
    #         my_simulator.simulate (mode = mode, cpu_cap_at_leaf=cpu_cap_at_leaf, sim_len_in_slots = 61)
    #     my_simulator.simulate (mode = mode, cpu_cap_at_leaf=min_req_cpu['cpvnf'], sim_len_in_slots = 61)
    #     if (mode != 'opt'): # if the mode isn't opt, need additional run, for finding the cost at the min req cpu for this mode
    #         my_simulator.simulate (mode = mode, cpu_cap_at_leaf=min_req_cpu[mode], sim_len_in_slots = 61)
    #
    # my_simulator.simulate (mode = 'ffit', cpu_cap_at_leaf=min_req_cpu['cpvnf'], sim_len_in_slots = 61)
    my_simulator.simulate (mode = 'ourAlg', cpu_cap_at_leaf=min_req_cpu['ffit'], sim_len_in_slots = 61)
    
def run_simulator (sim_pickle_file_name):
    """
    loads a simulator, and runs it
    """
    
    my_simulator = pickle.load (open ('../res/' + sim_pickle_file_name, 'rb'))
    my_simulator.simulate (mode = 'ffit', sim_len_in_slots = 61) 

if __name__ == "__main__":

    
    ap_file_name      = '0829_0830_1secs_256aps.ap' #'Lux_dump_ourAlg.t30563.ap' #'0829_0830_1secs_256aps.ap' #'0829_0830_1secs_256aps.ap' #'shorter.ap' #
    # ap_file_name      = 'Lux_dump_ourAlg.t30563.ap' #'0829_0830_1secs_256aps.ap' #'0829_0830_1secs_256aps.ap' #'shorter.ap' #
    ap2cell_file_name = 'Lux.center.post.antloc_256cells.ap2cell'
    my_simulator      = SFC_mig_simulator (ap_file_name=ap_file_name, verbose=[VERBOSE_RES], ap2cell_file_name=ap2cell_file_name)

    
    # my_simulator      = SFC_mig_simulator (ap_file_name=ap_file_name, verbose=[], ap2cell_file_name=ap2cell_file_name)
    # my_simulator.run_dump_usrs (30563)
    my_simulator.simulate (mode = 'ourAlg', cpu_cap_at_leaf=171, prob_of_target_delay=0.5, sim_len_in_slots=24, slot_to_dump=30563) #$$$ 
        # output_file       = open ('../res/RT_prob_sim_{}_{}.res' .format (ap2cell_file_name, ap_file_name), 'a')


    # my_simulator      = SFC_mig_simulator (ap_file_name=ap_file_name, verbose=[], ap2cell_file_name=ap2cell_file_name)
    # my_simulator.print_sol_res_line (output_file)

    # my_simulator.print_sol_res_line (output_file)
    
    # run_cost_by_rsrc ()
