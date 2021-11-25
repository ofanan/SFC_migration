import numpy as np
import math, time, heapq, random, sys, os
import pulp as plp
import networkx as nx
import matplotlib.pyplot as plt
from   pathlib import Path

from usr_c          import usr_c    # class of the users of alg
from usr_lp_c       import usr_lp_c # class of the users, when using LP
from decision_var_c import decision_var_c # class of the decision variables
from printf         import printf ## My own format print functions 
# from scipy._lib import _fpumode

# Levels of verbose / operation modes (which output is generated)
VERBOSE_DEBUG         = 0
VERBOSE_RES           = 1  # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_LOG           = 2  # Write to a ".log" file
VERBOSE_ADD_LOG       = 3  # Write to a detailed ".log" file
VERBOSE_ADD2_LOG      = 4  # Write ever more details to the detailed ".log" file
VERBOSE_MOB           = 5  # Write data about the mobility of usrs, and about the num of migrated chains per slot
VERBOSE_CALC_RSRC_AUG = 7  # Use binary-search to calculate the minimal reseource augmentation required to find a sol. The calculation is done only during a single time slot, and doesn't guarantee that the whole trace would succeed with this rsrc aug. Hence, this way of calculation is good for opt only, as opt searches each time fora solw from sratch.  
VERBOSE_MOVED_RES     = 8  # calculate the cost incurred by the usrs who moved  
VERBOSE_CRITICAL_RES  = 9  # calculate the cost incurred by the critical usrs  
VERBOSE_MIG_ONLY_CRIT = 10 # Disallow the mig' of non-critical chains
 

# Status returned by algorithms solving the prob' 
sccs = 1
fail = 2

# Round a float, and cast it to int
inter = lambda float_num : int (round(float_num))

# Minimum required CPU when prob=0.3, as found by previous runs. Used as a base for run_cost_vs_rsrc
MIN_REQ_CPU = {'Lux'    : {'opt' : 89,  'ourAlg' : 94, 'ffit' : 209, 'cpvnf': 214},     #Old: 'ffit' : 219, 'cpvnf' : 214
               'Monaco' : {'opt' : 840, 'ourAlg' : 842, 'ffit' : 1329, 'cpvnf' : 1329}} #Old: 'ffit' : 1354, 'cpvnf' : 1357

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
    calc_alg_sol_cost = lambda self, usrs: sum ([self.chain_cost_alg (usr, usr.lvl) for usr in usrs])
   
    # calculate the total cost of placing a chain at some level. 
    # The func' assume uniform cost of all links at a certain level; uniform mig' cost per VM; 
    # and uniform cost for all servers at the same layer.
    chain_cost_alg = lambda self, usr, lvl, slot_len=1 : slot_len * (self.link_cost_of_CLP_at_lvl[lvl] + self.cpu_cost_of_usr_at_lvl(usr, lvl)) + self.calc_alg_mig_cost (usr, lvl)
    
    # Returns the CPU cost of locating a given user on a server at a given level 
    cpu_cost_of_usr_at_lvl = lambda self, usr, lvl : self.CPU_cost_at_lvl[lvl] * usr.B[lvl]
    
    # # calculate the migration cost incurred for a usr if placed on a given lvl, assuming a CLP (co-located placement), namely, the whole chain is placed on a single server
    calc_alg_mig_cost = lambda self, usr, lvl : (usr.S_u[lvl] != usr.cur_s and usr.cur_s!=-1) * self.uniform_chain_mig_cost
          
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
    
    # Calculates the cost of a given decision variable, if the decision var is assigned 1.
    # This is when when the current state may be non co-located-placement. That is, distinct VMs (or fractions) of the same chain may be found in several distinct server. 
    d_var_cost = lambda self, d_var: self.d_var_cpu_cost (d_var) + self.d_var_link_cost (d_var) + self.d_var_mig_cost (d_var)

    # Calculates the cpu cost of a given decision variable, if the decision var is assigned 1.
    d_var_cpu_cost  = lambda self, d_var : self.cpu_cost_of_usr_at_lvl (d_var.usr, d_var.lvl) 

    # Calculates the link cost of a given decision variable, if the decision var is assigned 1.
    d_var_link_cost = lambda self, d_var : self.link_cost_of_CLP_at_lvl[d_var.lvl]
    
    # Print a solution for the problem to the output res file  
    print_sol_res_line = lambda self, output_file : self.print_sol_res_line_opt (output_file) if (self.mode == 'opt') else self.print_sol_res_line_alg (output_file)
    
    # parse a line detailing the list of usrs who moved, in an input ".poa" format file
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
    
    # Generate a string for the res file name. The sting will express the settings of this particular run, plus a user-requested string, 'mid_str', in which the caller may detail a concrete setting of 
    # this run (e.g. 'critical_usrs_only'). 
    gen_res_file_name  = lambda self, mid_str : '../res/{}{}_p{}_{}{}_sd{}.res' .format (self.poa_file_name.split(".")[0], mid_str, self.prob_of_target_delay[0], self.mode, self.critical_chains_only_string(), self.seed)

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

    # Returns the total cpu cost in the current time slot, according to the LP solution
    calc_cpu_cost_in_slot_opt = lambda self : sum ([self.d_var_cpu_cost(d_var)  * d_var.plp_var.value() for d_var in self.d_vars])
    
    # Returns the total link cost in the current time slot, according to the LP solution
    calc_link_cost_in_slot_opt = lambda self : sum ([self.d_var_link_cost(d_var) * d_var.plp_var.value() for d_var in self.d_vars])
    
    # Returns the total mig' cost in the current time slot, according to the LP solution
    calc_mig_cost_in_slot_opt = lambda self : sum ([self.d_var_mig_cost(d_var)  * d_var.plp_var.value() for d_var in self.d_vars])

    # Returns a string, detailing the sim' parameters (time, amount of CPU at leaves, probability of RT app' at leaf, status of the solution)
    settings_str = lambda self : 't{}_{}{}_cpu{}_p{:.1f}_sd{}_stts{}' .format(
                              self.t, self.mode, self.critical_chains_only_string(), self.G.nodes[len (self.G.nodes)-1]['RCs'], self.prob_of_target_delay[0], self.seed, self.stts)

    # Generate an identification, clarifying that this sim allows to migrate critical chains only. 
    critical_chains_only_string = lambda self : 'C' if VERBOSE_MIG_ONLY_CRIT in self.verbose else ''
    
    # Print a solution for the problem to the output res file when the solver is an LP solver  
    print_sol_res_line_opt = lambda self, output_file: printf (output_file, '{} | {}\n' .format(
            self.settings_str(), 
            self.sol_cost_str (cpu_cost  = self.calc_cpu_cost_in_slot_opt(),
                               link_cost = self.calc_link_cost_in_slot_opt()
                               ,mig_cost  = self.calc_mig_cost_in_slot_opt())))

    # Print a solution for the problem to the output res file when the solver is an algorithm (not an LP solver)  
    print_sol_res_line_alg = lambda self, output_file: printf (output_file, '{} | {} | num_usrs={} | num_crit_usrs={} \n' .format(
            self.settings_str(), # The settings string, detailing various parameters values used.  
            self.sol_cost_str (cpu_cost  = self.calc_cpu_cost_in_slot_alg(),
                               link_cost = self.calc_link_cost_in_slot_alg(),
                               mig_cost  = self.calc_mig_cost_in_slot_alg()),
            len (self.usrs),
            len (self.critical_usrs)))

    augmented_cpu_cap_at_leaf = lambda self: self.G.nodes[len (self.G.nodes)-1]['RCs']

    # Calculate the server-to-PoA mapping. ASSUMES that the input server s is a leaf server.)
    s2poa  = lambda self, s : self.poa2s.index(s)


    # Generate output file for RT_prob_sim, namely, simulations where we vary the prob' of a usr to be a RT usr, and measure the min' required CPU to find a feasible sol.
    gen_RT_prob_sim_output_file = lambda self, poa2cell_file_name, poa_file_name, mode : open ('../res/RT_prob_sim_{}_{}_{}.res' .format (poa2cell_file_name, poa_file_name, mode), 'a')    

    # Return the ID of the parent of the server given as input
    prnt_of_srvr = lambda self, s : self.G.nodes[s]['prnt']


    # Returns a string, detailing the sim' costs' components
    def sol_cost_str (self, cpu_cost, link_cost, mig_cost):
        tot_cost = cpu_cost + link_cost + mig_cost 
        return 'cpu_cost={:.0f} | link_cost={:.0f} | mig_cost={:.0f} | tot_cost={:.0f} | ratio=[{:.2f},{:.2f},{:.2f}]' .format(
                cpu_cost, link_cost, mig_cost, tot_cost, cpu_cost/tot_cost, link_cost/tot_cost, mig_cost/tot_cost)  

    def d_var_mig_cost (self, d_var): 
        """
        Calculate the mig' cost incurred by the value that the opt (LP) solution gave to the given decision variable.
        """
    
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
                d_var   = decision_var_c (d_var_id=d_var_id, usr=usr, lvl=lvl, s=usr.S_u[lvl], plp_var=plp_var) # generate a decision var, containing the lp var + details about its meaning 
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

        # solve the model, using the default settings. 
        model.solve() if (self.host == 'container') else model.solve(plp.PULP_CBC_CMD(msg=0)) # Suppress plp's output. Unfortunately, suppressing the output this way causes a compilation error 'PULP_CBC_CMD unavailable' while running on Polito's HPC  
        
        self.stts = sccs if (model.status==1) else fail       
        # print the solution to the output, according to the desired self.verbose level
        if (VERBOSE_RES in self.verbose):
            self.print_sol_res_line_opt (output_file=self.res_file)
            if (model.status != 1):
                printf (self.res_file, '// Status={}\n' .format (plp.LpStatus[model.status]))

            sol_cost_by_obj_func = model.objective.value()
            
            if (VERBOSE_DEBUG in self.verbose): 
                sol_cost_direct = sum ([self.d_var_cpu_cost(d_var)  * d_var.plp_var.value() for d_var in self.d_vars]) + \
                                  sum ([self.d_var_link_cost(d_var) * d_var.plp_var.value() for d_var in self.d_vars]) + \
                                  sum ([self.d_var_mig_cost(d_var)  * d_var.plp_var.value() for d_var in self.d_vars])
                if (abs (sol_cost_by_obj_func - sol_cost_direct) > 0.1): 
                    print ('Error: obj func value of sol={} while sol cost={}' .format (sol_cost_by_obj_func, sol_cost_direct))
                    exit ()
            
        # print the solution to the output, according to the desired self.verbose level
        if (VERBOSE_LOG in self.verbose):
            self.print_sol_res_line_opt (output_file=self.log_output_file)
            printf (self.log_output_file, 'tot_cost = {:.0f}\n' .format (model.objective.value())) 
            self.print_sol_to_log_opt ()
            if (model.status == 1): # successfully solved
                printf (self.log_output_file, '\nSuccessfully solved in {:.3f} [sec]\n' .format (time.time() - self.last_rt))
            else:
                printf (self.log_output_file, 'failed. status={}\n' .format(plp.LpStatus[model.status]))
            
        # if (VERBOSE_MOVED_RES in self.verbose and not(self.is_first_t)):
        #     self.print_sol_res_line (output_file=self.moved_res_file)
        return self.stts

    def init_res_file (self):
        """
        Open the res file for writing, as follows:
        If a res file with the relevant name already exists - open it for appending.
        Else, open a new res file, and write to it comment header lines, explaining the file's format  
        """
        
        # In the past, exponential CPU capacities/cost used special suffices, as in the commented line code below.
        #('.expCPU.res' if (self.use_exp_cpu_cost) else '') + ('.expCPU2.res' if (self.use_exp_cpu_cap) else '')
        self.res_file_name = self.gen_res_file_name (mid_str = '')  
        
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
        self.log_file_name = "../res/" + self.poa_file_name.split(".")[0] + '.' + self.mode + ('.detailed' if VERBOSE_ADD_LOG in self.verbose else '') +'.log'  
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
        Used for debugginign only.
        print the id, level and CPU of each user in a heap.
        """
        for usr in self.usrs:
            print ('id = {}, lvl = {}, CPU = {}' .format (usr.id, usr.lvl, usr.B[usr.lvl]))
        print ('')
        
    def push_up (self, usrs):
        """
        Push-up chains: given a feasible solution, greedily push each chain as high as possible in the tree, as long as this reduces the total cost.  
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
        if (self.t == self.slot_to_dump or self.poa_file_name=='shorter.poa' or self.poa_file_name == 'shorter_t2.poa'): #$$$ 
            self.dump_state_to_log_file()  

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
    
       
    def gen_parameterized_antloc_tree (self, poa2cell_file_name):
        """
        Generate a parameterized tree with specified height and children-per-non-leaf-node. 
        Add leaves for each PoA, and prune sub-trees that don't have any descended PoAs.
        """
        
        # Generate a complete balanced tree. If needed, later we will fix it according to the concrete distribution of PoAs.
        # Note: after fixing, the tree's height will be 1 level more, as we'll be adding a level for the PoAs below the current tree.
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
        self.rd_poa2cell_file(poa2cell_file_name)
        
        # Remove all the servers in cells that don't contain PoA
        shortest_path    = nx.shortest_path(self.G)
        
        self.G.nodes[root]['nChild'] = self.children_per_node
        for s in range (1, len(self.G.nodes())):
            self.G.nodes[s]['prnt']   = shortest_path[s][root][1]
            self.G.nodes[s]['nChild'] = self.children_per_node # Num of children of this server (==node)
        for cell in range (self.num_of_leaves): # for each cell 
            PoAs_of_this_cell = list (filter (lambda item : item['cell']==cell, self.PoAs))
            if (len(PoAs_of_this_cell)==0): # No PoAs at this cell
                s2remove = self.cell2s[cell] #server to be removed
                self.G.nodes[self.prnt_of_srvr(s2remove)]['nChild'] -= 1 # Dec. the # of children of the parent
                self.G.remove_node(self.cell2s[cell]) # Remove the leaf server handling this cell
                self.num_of_leaves -= 1 # We have just removed one leaf server
                self.cell2s[cell] = -1 # Now, this cell isn't associated with any server

        srvrs2remove = [s for s in self.G.nodes() if (s>0 and self.G.nodes[s]['nChild']==0)]
        for s in srvrs2remove:
            prnt = self.prnt_of_srvr(s)
            if (prnt!=0): # Don't try to remove the root, as this means there's no tree at all
                self.G.nodes[prnt]['nChild'] -= 1 # Dec. the # of children of the parent
                if (self.G.nodes[prnt]['nChild']==0):
                    srvrs2remove.append(prnt)
            self.G.remove_node(s)
                
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
                       
        # Add new leaves for the PoAs below each cell. 
        # To keep the IDs of leaves the greatest in the tree, we do that only after we finished removing all the useless cells.
        self.num_of_srvrs = len(self.G.nodes())
        for poa in [poa for poa in self.PoAs]: #(filter (lambda item : item['cell']==cell, self.PoAs)): # for each poa belonging to this cell
            poa['s'] = len(self.G.nodes())   # We'll shortly add a server for this PoA, so the id of this PoA will be current number of servers+1.
            self.G.add_node (poa['s']) # Add a server co-located with this PoA
            self.G.nodes[poa['s']]['lvl'] = 0 # The level of the newly added node is 0 (it's a leaf)
            self.G.add_edge (poa['s'], self.cell2s[poa['cell']]) # Add an edge from the newly added server, to the server handling the cell of this PoA, ...
            self.G.add_edge (self.cell2s[poa['cell']], poa['s']) # And vice versa

        self.num_of_leaves    = len(self.PoAs) # Each PoA is co-located with a leaf server
        self.poa2s             = [poa['s'] for poa in self.PoAs] # Will contain a least translating the PoA number (==leaf #) to the ID of the co-located server. 

        # Update the tree height's by the changes made, and set 
        shortest_path    = nx.shortest_path(self.G)
        self.tree_height = len (shortest_path[self.poa2s[0]][root]) - 1
        
        self.CPU_cost_at_lvl   = [2**(self.tree_height - lvl) for lvl in range (self.tree_height+1)] if self.use_exp_cpu_cost else [(1 + self.tree_height - lvl) for lvl in range (self.tree_height+1)]
        self.link_cost_at_lvl  = self.uniform_link_cost * np.ones (self.tree_height) #self.link_cost_at_lvl[i] is the cost of locating a full chain at level i
        self.link_delay_at_lvl = 2 * np.ones (self.tree_height) #self.link_delay_at_lvl[i] is the return delay when locating a full chain at level i 
        
        # overall link cost and link capacity of a Single-Server Placement of a chain at each lvl
        self.link_cost_of_CLP_at_lvl  = [2 * sum([self.link_cost_at_lvl[i]  for i in range (lvl)]) for lvl in range (self.tree_height+1)]
        self.link_delay_of_CLP_at_lvl = [2 * sum([self.link_delay_at_lvl[i] for i in range (lvl)]) for lvl in range (self.tree_height+1)] 

        # Levelize the updated tree
        for leaf in self.poa2s:
            
            for lvl in range (self.tree_height):
                self.G.nodes[shortest_path[leaf][root][lvl]]['lvl']     = np.uint8(lvl) # assume here a balanced tree
        self.G.nodes[0]['lvl'] = self.tree_height

        for s in self.G.nodes(): # for every server
            self.G.nodes[s]['id'] = s

        # Find parents of each node (except of the root)
        for s in range (1, len(self.G.nodes())):
            self.G.nodes[s]['prnt'] = shortest_path[s][root][1]
        
        # self.draw_graph()
        
    
    def draw_graph (self):
        """
        Plotting the graph of servers. Used for debugging / logging only.
        """
        nx.draw(self.G, with_labels=True)
        plt.show()

    def gen_parameterized_full_tree (self):
        """
        Generate a parameterized full tree with specified height and children-per-non-leaf-node. 
        """
        
        # Generate a complete balanced tree. If needed, later we will fix it according to the concrete distribution of PoAs.
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
        self.poa2s             = [] # Will contain a least translating the PoA number (==leaf #) to the ID of the co-located server. 
        root                  = 0 # In networkx, the ID of the root is 0
        self.num_of_leaves    = 0
        for s in self.G.nodes(): # for every server
            self.G.nodes[s]['id'] = s
            if self.G.out_degree(s)==1 and self.G.in_degree(s)==1: # is it a leaf?
                self.G.nodes[s]['lvl']   = 0 # Yep --> its lvl is 0
                self.poa2s.append (s) #[self.num_of_leaves] = s
                self.num_of_leaves += 1
                for lvl in range (self.tree_height+1):
                    self.G.nodes[shortest_path[s][root][lvl]]['lvl']       = np.uint8(lvl) # assume here a balanced tree

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

    def __init__ (self, poa_file_name = 'shorter.poa', # File detailing the PoA of each user (e.g., car, pedestrian) along the trace.  
                        verbose = [], # the type of output produced, e.g.: costs, amount of cpu used, detailed log, debugging. See "VERBOSE_" at this file's header. 
                        tree_height = 4, children_per_node = 4, # topology of the tree (apart from the leaf, which are potentially real antennas). 
                        prob_of_target_delay=0.3, # prob' that a simulated usr has RT requirements 
                        poa2cell_file_name='', # File detailing the attachment of each PoA to a rectangle. In our hierarchy, the leaves are antennas. The leaves' parents are the lowest-level rectangles.
                                               # If no input is given here, the simulation assumes uses no real-data of antenna location, and each leaf in the tree is co-located with a server, covering a rectangled area. 
                        use_exp_cpu_cost=True, # True iff the cost of the CPU exponentially increase when moving down in the tree (costs are 1,2,4,16, lowest cost is in the root. When False, costs are linear (1, 2, 3.,,,).  
                        use_exp_cpu_cap=False  # True iff the cpu capacities exponentially increase when moving up in the tree. (costs are 1,2,4,8, ..., ; lowest capacity is in the roots, When False, capacities are linear (1, 2, 3.,,,).
                        ):
        """
        """

        if ('ofana' in os.getcwd().split ('\\')):
            self.host = 'laptop'
        else:
            self.host = 'container'

        self.verbose                    = verbose

        self.poa_file_name              = poa_file_name #input file containing the PoAs of all users along the simulation
        self.city                       = self.poa_file_name.split('_')[0]
        if (poa2cell_file_name != ''):
            if (poa2cell_file_name.split('.')[0] != self.city):
                print ('Error: the cities specified by poa file and by po2cell file differ. poacell_file_name={}, poa2cell_file_name={}. self.city={}. splitter={}' .format (self.poa_file_name, poa2cell_file_name, self.city, poa2cell_file_name.split('.')[0]))
                exit ()
        self.use_exp_cpu_cost           = use_exp_cpu_cost
        self.use_exp_cpu_cap            = use_exp_cpu_cap
        
        # Network parameters
        if (self.poa_file_name=='shorter.poa' or self.poa_file_name=='shorter_t2.poa'):
            self.tree_height            = 2
        else:
            self.tree_height            = 4 
        self.children_per_node          = 2 if (self.poa_file_name=='shorter.poa' or self.poa_file_name=='shorter_t2.poa') else children_per_node
        self.uniform_vm_mig_cost        = 200
        self.Lmax                       = 0
        self.uniform_Tpd                = 2
        self.uniform_link_cost          = 3
        self.uniform_theta_times_lambda = [2, 10, 2] # "1" here means 100MHz
        self.uniform_chain_mig_cost     = self.uniform_vm_mig_cost * len (self.uniform_theta_times_lambda)
        self.uniform_Cu                 = 20 # Maximum CPU amount that may be allocated to a single user. 
        self.target_delay               = [10, 100] # in [ms], lowest to highest
        self.prob_of_target_delay       = [prob_of_target_delay]  
        self.warned_about_too_large_poa  = False
        self.usrs                       = [] # The list of usrs, to be dynamically change along the sim'.
        
        # Init output files
        if (VERBOSE_DEBUG in self.verbose):
            self.debug_file = open ('../res/debug.txt', 'w') 
        # if (VERBOSE_MOB in self.verbose):
        #     self.num_of_moved_usrs_in_slot         = [] # self.num_of_moved_usrs_in_slot[t] will hold the num of usrs who moved at slot t.   
        #     self.num_of_migs_in_slot          = [] # self.num_of_migs[t] will hold the num of chains assigned to migrate in slot t.
        #     self.num_of_critical_usrs_in_slot = [] 
        #     self.mig_from_to_lvl      = np.zeros ([self.tree_height+1, self.tree_height+1], dtype='int') # self.mig_from_to_lvl[i][j] will hold the num of migrations from server in lvl i to server in lvl j, along the sim

        self.poa2cell_file_name = poa2cell_file_name
        
        # poa_file_used_antloc will be True iff the poa file was generated using real antennas' location data.
        poa_file_used_antloc = True if (self.poa_file_name.split('.poa')[0].split('_')[-1] in ['all', 'orange', 'post', 'Telecom', 'short']) else False
        
        if (self.poa_file_name=='shorter.poa'): # 'shorter.poa' is used only for developing and debugging, and uses no real antennas' location ('.antloc')
            self.gen_parameterized_full_tree  ()
        
        elif (self.poa2cell_file_name==''): # The simulated network is synthetic - includes only rectangles, and not real antenna locations 
            if (poa_file_used_antloc):
                print ('Error: you specified no poa2cell file name, but the .poa file was generated using an antloc file.')
                exit ()
            self.gen_parameterized_full_tree  ()
        else: # The simulated network is synthetic - includes only rectangles, and not real antenna locations
            if (not (poa_file_used_antloc)):
                print ('Error: you specified poa2cell_file_name={}, but the .poa file was not generated using an antloc file.' .format (self.poa2cell_file_name))                
                exit ()
            self.gen_parameterized_antloc_tree (self.poa2cell_file_name)
        self.delay_const_sanity_check()

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

    def simulate (self, mode, prob_of_target_delay=None, cpu_cap_at_leaf=516, sim_len_in_slots=float('inf'), slot_to_dump=float('inf'), seed=42): 
        
        """
        Simulate the whole simulation using the chosen alg: LP (for finding an optimal fractional solution), or an algorithm (either our alg, or a benchmark alg' - e.g., first-fit, worst-fit).
        Return value: if the simulation succeeded (found a feasible solution for every time slot during the sim') - return the cpu capacity used in a leaf node.
        Else, return None. 
        """
        
        self.seed                       = seed
        random.seed                     (self.seed)  
        self.usrs                       = []
        self.slot_to_dump               = slot_to_dump
        self.delay_const_sanity_check()

        self.cpu_cap_at_leaf = 30 if self.poa_file_name == 'shorter.poa' else cpu_cap_at_leaf 
        self.cpu_cap_at_lvl  = self.calc_cpu_capacities (self.cpu_cap_at_leaf)
        self.set_RCs_and_a  (aug_cpu_capacity_at_lvl=self.cpu_cap_at_lvl) # initially, there is no rsrc augmentation, and the capacity and currently-available cpu of each server is exactly its CPU capacity.

        self.usrs                       = [] 

        if (prob_of_target_delay!=None): # If the caller stated a different prob_of_target_delay, assingg it
            self.prob_of_target_delay = [prob_of_target_delay]  

        self.mode              = mode
        
        # Set the upper limit of the binary search. Running opt is much slower, and usually doesn't require much rsrc aug', and therefore we may set for it lower value.
        if (self.mode == 'opt'):
            self.max_R = 1.6 
        elif (self.mode == 'ourAlg'):   
            self.max_R = 1.2 
        else:
            self.max_R = 1.8

        self.sim_len_in_slots = sim_len_in_slots
        self.is_first_t = True # Will indicate that this is the first simulated time slot

        self.init_input_and_output_files()        
                     
        print ('Simulating {}. poa_file = {} cpu_cap_at_leaf={}, prob_of_RT={:.2f}, seed={}' .format (self.mode, self.poa_file_name, self.cpu_cap_at_leaf, self.prob_of_target_delay[0], self.seed))
        self.stts     = sccs

        # extract the slot len from the input '.poa' file name
        slot_len_str = self.poa_file_name.split('secs')
        self.slot_len = int(slot_len_str[0].split('_')[-1]) if (len (slot_len_str) > 1) else 1 # By default, slot len is 1
        
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
        Write several comments at the output files headers. 
        """

        # open input file
        self.poa_file  = open ("../res/poa_files/" + self.poa_file_name, "r")  

        # open output files, and print there initial comments
        if (VERBOSE_RES in self.verbose):
            self.init_res_file()
        # if VERBOSE_CALC_RSRC_AUG in self.verbose:
        #     if (self.use_exp_cpu_cost and self.use_exp_cpu_cap): 
        #         self.rsrc_aug_file_name = '../res/rsrc_aug_by_RT_prob_exp_cpu^2.res' 
        #     elif (self.use_exp_cpu_cost):
        #         self.rsrc_aug_file_name = '../res/rsrc_aug_by_RT_prob_exp_cpu_cost_{}.res' .format (self.poa2cell_file_name)
        #     else: 
        #         self.rsrc_aug_file_name = '../res/rsrc_aug_by_RT_prob.res'
        #
        #     if Path (self.rsrc_aug_file_name).is_file(): # does this res file already exist?
        #         self.rsrc_aug_file =  open (self.rsrc_aug_file_name,  "a")
        #     else:
        #         self.rsrc_aug_file =  open (self.rsrc_aug_file_name,  "w")
            
        # if (VERBOSE_MOVED_RES in self.verbose):
        #     self.init_moved_res_file()
        # if (VERBOSE_CRITICAL_RES in self.verbose):
        #     self.init_critical_res_file()
        if (VERBOSE_LOG in self.verbose):
            self.init_log_file()
        # if (VERBOSE_MOB in self.verbose):
        #     self.mob_file_name   = "../res/" + self.poa_file_name.split(".")[0] + '.' + self.mode.split("_")[1] + '.mob.log'  
        #     self.mob_output_file =  open ('../res/' + self.mob_file_name,  "w") 
        #     printf (self.mob_output_file, '// results for running alg_top on input file {}\n' .format (self.poa_file_name))
        #     printf (self.mob_output_file, '// results for running alg_top on input file shorter.poa with {} leaves\n' .format (self.num_of_leaves))
        #     printf (self.mob_output_file, '// index i,j in the matrices below represent the total num of migs from lvl i to lvl j\n')
    
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
        self.critical_usrs = [] # rst the list of usrs who are critical in this time slot
        # self.moved_usrs    = [] # rst the list of usrs who moved in this time slot 

                    
    def simulate_lp (self):
        """
        Simulate the whole simulation, using a LP fractional solution.
        At each time step:
        - Read and parse from an input ".poa" file the PoA cells of each user who moved. 
        - solve the problem using a LP, using Python's Pulp LP solver. 
        - Write outputs results and/or logs to files.
        - update cur_st = nxt_st
        - If the alg' failed to find a feasible sol', even at a single slot, return with self.stts=fail
        """
        self.cur_st_params = []
        
        for line in self.poa_file: 
        
            # Ignore comments and emtpy lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue
        
            line = line.split ('\n')[0]
            splitted_line = line.split (" ")
        
            if (splitted_line[0] == "t"): # Reached a line indicating a new time slot in the ".poa" input file
                self.rd_t_line (splitted_line[2])
                if (self.t >= self.final_slot_to_simulate): # finished the desired simulation time
                    self.post_processing () # Write statistics concluding the simulation to output files 
                    return 
        
            elif (splitted_line[0] == "usrs_that_left:"): # Reached a line indicating usrs that left the sim in the ".poa" input file
        
                for usr in list (filter (lambda usr : usr.id in [int(usr_id) for usr_id in splitted_line[1:] if usr_id!=''], self.usrs)):
                    self.usrs.remove  (usr) # Remove the usr from the list of usrs
                    self.cur_st_params = list (filter (lambda param : param.usr != usr, self.cur_st_params)) # Remove any parameter corresponding to the current state of this usr (who left)
        
            elif (splitted_line[0] == "new_usrs:"): # Reached a line indicating new usrs that joined the sim in the ".poa" input file
                self.rd_new_usrs_line  (splitted_line[1:])
            elif (splitted_line[0] == "old_usrs:"): # Reached a line indicating 'old' existing usrs that moved in the ".poa" input file
                self.rd_old_usrs_line_lp (splitted_line[1:])
                if (VERBOSE_ADD_LOG in self.verbose):
                    self.last_rt = time.time ()
                self.stts = self.alg_top(self.solve_by_plp) # call the top-level alg' that solves the problem, possibly by binary-search, using iterative calls to the given solver (plp LP solver, in our case).

                if (self.stts!=sccs and VERBOSE_CALC_RSRC_AUG in self.verbose): # opt failed at this slot, but a binary search was requested. Begin a binary search, to find the lowest cpu cap' that opt needs for finding a feasible sol' for this slot. 

                    [self.lb, self.ub] = [self.cpu_cap_at_leaf, self.cpu_cap_at_leaf * self.max_R]
                    while True:
                        if (self.ub <= self.lb+2): # the difference between the lb and the ub is at most 1
                            if (self.cpu_cap_at_leaf == self.ub and self.stts==sccs): # we've already successfully run with this cap' no need to try again
                                break # found a feasible sol for this slot

                            # Need on last run, to verify the value found by binary search, and calculate its cost
                            self.cpu_cap_at_leaf = self.ub
                            self.cpu_cap_at_lvl  = self.calc_cpu_capacities (self.cpu_cap_at_leaf)
                            self.set_RCs_and_a (self.cpu_cap_at_lvl) 
                            self.stts = self.alg_top(self.solve_by_plp)
                            break # found a feasible sol for this slot
            
                        self.cpu_cap_at_leaf = self.avg_up_and_lb(self.ub, self.lb)
                        self.cpu_cap_at_lvl  = self.calc_cpu_capacities (self.cpu_cap_at_leaf)
                        self.set_RCs_and_a (self.cpu_cap_at_lvl) 
                        self.stts = self.alg_top(self.solve_by_plp)
                        if (self.stts==sccs): 
                            self.ub = self.cpu_cap_at_leaf
                        else: 
                            self.lb = self.cpu_cap_at_leaf
                            
                self.cur_st_params = self.d_vars.copy () #All the decision vars will be referred as "cur st parameters" in the next time slot 
                self.is_first_t = False  # The next slot is surely not the first slot

        self.post_processing ()
        
    def rmv_usr_from_all_Hs (self, usr):
        """
        Remove a given usr from all the "Hs" sets of servers.
        An Hs set of server s is the set of usrs for which s is delay feasible.
        Namely, this is the list of usrs that may be placed on s while maintaining the delay constraints. 
        """
        for s in [s for s in self.G.nodes() if usr in self.G.nodes[s]['Hs']]:
            self.G.nodes[s]['Hs'].remove (usr)
            
        
    def simulate_algs (self):
        """
        Simulate using an algorithm (NOT a LP solver).
        At each time step:
        - Read and parse from an input ".poa" file the PoA cells of each user who moved. 
        - solve the problem using alg_top (our alg). 
        - Write outputs results and/or logs to files.
        - Update self.stts according to the solution's stts.
        - update cur_st = nxt_st
        - If the alg' failed to find a feasible sol', even at a single slot, return with self.stts=fail
        """
        
        # reset Hs     
        if (self.mode in ['ourAlg']):
            for s in self.G.nodes():
                self.G.nodes[s]['Hs']  = set() 
        
        for line in self.poa_file: 

            # Ignore comments lines
            if (line == "\n" or line.split ("//")[0] == ""):
                continue

            line = line.split ('\n')[0]
            splitted_line = line.split (" ")

            if (splitted_line[0] == "t"): # Reached a line indicating new usrs that joined the sim in the ".poa" input file
                self.rd_t_line (splitted_line[2])
                if (self.t >= self.final_slot_to_simulate):
                    self.post_processing ()
                    return
                continue
                
            elif (splitted_line[0] == "usrs_that_left:"): # Reached a line indicating usrs that left the sim in the ".poa" input file

                for usr in list (filter (lambda usr : usr.id in [int(usr_id) for usr_id in splitted_line[1:] if usr_id!=''], self.usrs)):

                    self.G.nodes[usr.cur_s]['a'] += usr.B[usr.lvl] # free the CPU units used by the user in the old location
                    if (self.mode in ['ourAlg']):
                        self.rmv_usr_from_all_Hs(usr)
                    self.usrs.remove (usr)                    
                continue
        
            elif (splitted_line[0] == "new_usrs:"): # Reached a line listing the new usrs that joined the sim in the ".poa" input file
                self.rd_new_usrs_line (splitted_line[1:])
            elif (splitted_line[0] == "old_usrs:"): # Reached a line listing the 'old', existing usrs that moved, in the ".poa" input file 
                self.rd_old_usrs_line (splitted_line[1:]) # Read the list of old usrs, and collecting their new PoA assignments
                if (VERBOSE_ADD_LOG in self.verbose):
                    self.last_rt = time.time ()
                    printf (self.log_output_file, 't={}. beginning alg top\n' .format (self.t))
                    
                # solve the prob' using the requested alg'    
                if   (self.mode in ['ourAlg']):
                    self.stts = self.alg_top(self.bottom_up)
                    if (self.stts==sccs): # if we bottom-up succeeded, perform push-up 
                        self.push_up (self.usrs) if self.reshuffled else self.push_up(self.critical_usrs) 
                elif (self.mode == 'ffit'):
                    self.stts = self.alg_top(self.first_fit)
                elif (self.mode == 'wfit'):
                    self.stts = self.alg_top(self.worst_fit)
                elif (self.mode == 'cpvnf'):
                    self.stts = self.alg_top(self.cpvnf)
                else:
                    print ('Sorry, mode {} that you selected is not supported' .format (self.mode))
                    exit ()
        
                self.print_sol_to_res_and_log ()
                if (self.stts!=sccs):
                    return # Once an alg' fails to find a sol even for a single slot, we don't try to further simulate, 
                
                for usr in self.usrs: # The solution found at this time slot is the "cur_state" for next slot
                    usr.cur_s = usr.nxt_s
                self.is_first_t = False  # The next slot is surely not the first slot
        
        self.post_processing()    
        
    def post_processing (self):
        """
        Organize, writes and plots the simulation results, after the simulation is done
        """
        # if (VERBOSE_MOB in self.verbose):
        #     self.print_mob ()        
        # if (VERBOSE_CALC_RSRC_AUG in self.verbose):
        #     print ('augmented cpu cap at leaf={}' .format (self.augmented_cpu_cap_at_leaf()))
        #     self.print_sol_res_line (self.rsrc_aug_file)
    
    # def print_mob (self):
    #     """
    #     print statistics about the number of usrs who moved, and the num of migrations between every two levels in the tree.
    #     """
    #
    #     sim_len = float(self.t - self.first_slot)
    #     del (self.num_of_migs_in_slot[0]) # remove the mig' recorded in the first slot, which is irrelevant (corner case)
    #     printf (self.mob_output_file, '// avg num of usrs that moved per slot = {:.0f}\n'   .format (float(sum(self.num_of_moved_usrs_in_slot)) / sim_len))
    #     printf (self.mob_output_file, '// avg num of usrs who migrated per slot = {:.0f}\n' .format (float(sum(self.num_of_migs_in_slot)) / sim_len))
    #     avg_num_of_migs_to_from_per_slot = np.divide (self.mig_from_to_lvl, sim_len)
    #     for lvl_src in range (self.tree_height+1):
    #         for lvl_dst in range (self.tree_height+1):
    #             printf (self.mob_output_file, '{:.0f}\t' .format (avg_num_of_migs_to_from_per_slot[lvl_src][lvl_dst]))
    #         printf (self.mob_output_file, '\n')
    #     printf (self.mob_output_file, 'moves_in_slot = {}\n' .format (self.num_of_moved_usrs_in_slot))
    #     printf (self.mob_output_file, 'migs_in_slot = {}\n'  .format (self.num_of_migs_in_slot))
    #
    #     # plot the mobility
    #     plt.figure()
    #     plt.title ('Migrations and mobility at each slot')
    #     plt.plot (range(int(sim_len)), self.num_of_moved_usrs_in_slot, label='Total vehicles moved to another cell [number/sec]', linestyle='None',  marker='o', markersize = 4)
    #     plt.plot (range(int(sim_len)), self.num_of_migs_in_slot, label='Total chains migrated to another server [number/sec]', linestyle='None',  marker='.', markersize = 4)
    #     plt.xlabel ('time [seconds, starting at 07:30]')
    #     plt.legend()
    #     plt.savefig ('../res/{}.mob.jpg' .format(self.poa_file_name.split('.')[0]))
    #     plt.clf()
        
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
                if (VERBOSE_MIG_ONLY_CRIT in self.verbose): # Allowed to mig' only critical chains
                    return fail
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
                if (VERBOSE_MIG_ONLY_CRIT in self.verbose): # Allowed to mig' only critical chains
                    return fail
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
                if (VERBOSE_MIG_ONLY_CRIT in self.verbose): # Allowed to mig' only critical chains
                    return fail
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
        self.reshuffled = True if (self.mode=='opt') else False # In the first run of algs (not Opt), we consider only critical chains. Only if this run fails, we'll try a reshuffle.

        self.stts = placement_alg()
        
        if (VERBOSE_LOG in self.verbose):
            self.print_sol_res_line (self.log_output_file)
            self.print_sol_to_log_alg()
            
        if (self.stts == sccs):
            return sccs
        
        if (VERBOSE_MIG_ONLY_CRIT in self.verbose): # Allowed to mig' only critical chains. So, if we haven't succeeded by now - return fail.
            return fail

        # Now we know that the first run fail. For all the benchmarks, it means that they also made a reshuffle. 
        # However, bottom-up haven't tried a reshuffle yet. So, we give it a try now.
        if (self.mode in ['ourAlg']):
            self.rst_sol()
            self.reshuffled = True # In this run, we'll perform a reshuffle (namely, considering all usrs).
            if (VERBOSE_LOG in self.verbose):
                printf (self.res_file, '// reshuffling\n')
            self.stts = self.bottom_up()
            if (VERBOSE_ADD_LOG in self.verbose):
                printf (self.log_output_file, 'after reshuffle:\n')
                self.print_sol_to_log_alg()
                self.print_sol_res_line (self.log_output_file)
            if (self.stts == sccs):
                return sccs

        return self.stts
    
    def bottom_up (self):
        """
        Our bottom-up alg'. 
        Assigns all self.usrs that weren't assigned yet (either new usrs, or old usrs that moved, and now they don't satisfy the target delay).
        Looks for a feasible sol'.
        Returns sccs if a feasible sol was found, fail else.
        """      
        if (self.t == self.slot_to_dump or self.poa_file_name=='shorter.poa' or self.poa_file_name == 'shorter_t2.poa'): #$$$ 
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

    def update_S_u (self, usr, poa_id):
        """
        Update the S_u (list of delay-feasible servers) of a given usr, given the id of its current PoA (Access Point server)
        """                    
        usr.S_u = []
        s       = self.poa2s[poa_id]
        usr.S_u.append (s)
        for _ in (range (len(usr.B)-1)):
            s = self.parent_of(s)
            usr.S_u.append (s)
    
    def rd_old_usrs_line (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".poa" file.
        The input includes a list of usr_entries of the format (usr,poa), where "usr" is the user id, and "poa" is its current access point (PoA).
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
            usr_entry = usr_entry.split("(")[1].split (',')
            
            list_of_usr = list(filter (lambda usr : usr.id == int(usr_entry[0]), self.usrs))
            usr = list_of_usr[0]
            
            # self.moved_usrs.append (usr)
            usr_cur_cpu = usr.B[usr.lvl]
            self.CPUAll_single_usr (usr) # update usr.B by the new requirements of this usr.
            self.update_S_u(usr, poa_id=int(usr_entry[1])) # Add this usr to the Hs of every server to which it belongs in its new location
                        
            if (self.mode in ['ourAlg']):
                self.rmv_usr_from_all_Hs(usr) 
                for s in usr.S_u:
                    self.G.nodes[s]['Hs'].add(usr) # Add the usr only to the 'Hs' (list of usrs that may be hosted on a server) of each of its delay-feasible servers
                                              
            # Is it possible to comply with the delay constraint of this usr while staying in its cur location and keeping the CPU budget 
            if (usr.cur_s in usr.S_u and usr_cur_cpu <= usr.B[usr.lvl]):  
                continue # Yep: the delay constraint are satisfied also in the current placement.
            
            # Now we know that this is a critical usr, namely a user that needs more CPU and/or migration for satisfying its target delay constraint 
            # dis-place this user (mark it as having nor assigned level, neither assigned server), and free its assigned CPU
            self.critical_usrs.append(usr)
            self.G.nodes[usr.cur_s]['a'] += usr.B[usr.lvl] # free the CPU units used by the user in the old location
            usr.lvl   = -1
            usr.nxt_s = -1

    def rd_new_usrs_line (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".poa" file.
        The input includes a list of usr_entries of the format (usr,poa), where "usr" is the user id, and "poa" is its current access point (PoA).
        After reading the usr_entries, the function assigns each chain to its relevant list of chains, Hs.  
        """
        
        if (line ==[]):
            return # no new users

        splitted_line = line[0].split ("\n")[0].split (")")

        for usr_entry in splitted_line:
            if (len(usr_entry) <= 1):
                return
            usr_entry = usr_entry.split("(")[1].split (',')

            if (self.mode == 'opt'):
                usr = usr_lp_c (int(usr_entry[0]), theta_times_lambda=self.uniform_theta_times_lambda, target_delay=self.pseudo_random_target_delay (int(usr_entry[0])), C_u=self.uniform_Cu) # generate a new usr, which is assigned as "un-placed" yet (usr.lvl==-1)
            else: 
                usr = self.gen_new_usr (usr_id=int(usr_entry[0]))
            
            # self.moved_usrs.append (usr)
            self.critical_usrs.append(usr)

            self.usrs.append (usr)
            self.CPUAll_single_usr (usr) 
            self.update_S_u(usr, poa_id=int(usr_entry[1])) # Update the list of delay-feasible servers for this usr 

            if (self.mode == 'ourAlg'):                
                for s in usr.S_u: # Hs is the list of chains that may be located on each server while satisfying the delay constraint. Only some of the algs' use it
                    self.G.nodes[s]['Hs'].add(usr)                       
                    
    def rd_old_usrs_line_lp (self, line):
        """
        Read a line detailing the new usrs just joined the system, in an ".poa" file, when using a LP solver for the problem.
        The input includes a list of usr_entries of the format (usr,poa), where "usr" is the user id, and "poa" is its current access point (PoA).
        After reading the usr_entries, the function assigns each chain to its relevant list of chains, Hs.  
        """
        if (line == []): # if the list of old users that moved is empty
            return
        
        splitted_line = self.parse_old_usrs_line(line)

        for usr_entry in splitted_line:
            if (len(usr_entry) <= 1):
                break
            usr_entry = usr_entry.split("(")[1].split (',')
            
            list_of_usr = list(filter (lambda usr : usr.id == int(usr_entry[0]), self.usrs))
            usr = list_of_usr[0]
            usr.is_new = False
            # self.moved_usrs.append(usr)
            self.CPUAll_single_usr (usr)
            self.update_S_u(usr, poa_id=int(usr_entry[1])) # Add this usr to the Hs of every server to which it belongs at its new location
    
    def parse_usr_entry (self, usr_entry):
        """
        parse an entry of a user in an '.poa' file.
        Returns the corresponding usr from self.usrs.
        This func' is for existing ('old') usrs who moved only.
        """
        
        usr_entry = usr_entry.split("(")[1].split (',')
        # usr_entry = usr_entry[1].split (',')

        
        list_of_usr = list(filter (lambda usr : usr.id == int(usr_entry[0]), self.usrs))
        if (len(list_of_usr) == 0):
            print  ('Error at t={}: input file={}. Did not find old / rescycled usr {}' .format (self.t, self.poa_file_name, usr_entry[0]))
            exit ()
        return list_of_usr[0]
    
    def check_poa_id (self, poa_id):
        """
        For debug only.
        Basic sanity check for the given poa_id (Id of the PoA). 
        """
        if (poa_id >= self.num_of_leaves):
            poa_id = self.num_of_leaves-1
        if (self.warned_about_too_large_poa == False):
            print ('Encountered PoA num {} in the input file, but in the tree there are only {} leaves. Changing the ap to {}' .format (poa_id, self.num_of_leaves, self.num_of_leaves-1))
            exit ()
    
    def inc_array (self, ar, min_val, max_val):
        """
        Currently unused.
        An accessory function for brute-force searches.
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
        """
        Used only for deep log / debugging.
        Dumps data about the current network's state.
        """

        log_file = self.log_output_file 
        printf (log_file, 't={}\n' .format (self.t))
        for usr in self.usrs:
            printf (log_file, 'u.id={}, cur_s={}, nxt_s={}\n' .format(usr.id, usr.cur_s, usr.nxt_s))
        for s in [s for s in self.G.nodes()]:
            printf (log_file, 's{}: Hs={}\n' .format (s, sorted (list ([usr.id for usr in self.G.nodes[s]['Hs']])))) 

    def dump_usrs_to_poa_file (self):
        """
        Used mainly for logging / debugging.
        Dumps the PoA of all usrs at the current time slot.
        """

        output_poa_file = open ('../res/Lux_dump_{}.t{}.poa' .format (self.mode, self.t), 'w')
        printf (output_poa_file, '// Dumping into poa file={} poa2cell_file_name={}' .format (self.poa_file_name, self.poa2cell_file_name))
        printf (output_poa_file, '// File format:\n//for each time slot:\n')
        printf (output_poa_file, 't = {}\n' .format(self.t))
        printf (output_poa_file, 'new_usrs: ' .format(self.t))

        for usr in sorted (self.usrs, key = lambda usr : usr.id):
            self.CPUAll_single_usr (usr) 
            printf (output_poa_file, '({},{})' .format (usr.id, self.s2poa(usr.S_u[0]))) # S_u is the list of delay-feasible servers for that usr. S_u[0] is the leaf server (namely, the PoA) out of them.
        printf (output_poa_file, '\nold_usrs:\n')
    
    # def run_dump_usrs (self, slot_to_dump):
    #     """
    #     Write the current PoA association of all users at a given time slot to an .poa file
    #     """
    #     random.seed (42) # Use a fixed pseudo-number seed 
    #     self.usrs   = []
    #     self.mode   = 'bypass'
    #
    #     self.is_first_t = True # Will indicate that this is the first simulated time slot
    #
    #     self.input_poa_file = open ("../res/" + self.poa_file_name, "r")  
    #     for line in self.input_poa_file: 
    #
    #         # Ignore comments lines
    #         if (line == "\n" or line.split ("//")[0] == ""):
    #             continue
    #
    #         line = line.split ('\n')[0]
    #         splitted_line = line.split (" ")
    #
    #         if (splitted_line[0] == "t"):
    #             self.t = int(splitted_line[2])
    #
    #         elif (splitted_line[0] == "usrs_that_left:"):
    #             self.rd_usrs_that_left_line (splitted_line)
    #             continue
    #
    #             continue
    #
    #         elif (splitted_line[0] == "new_usrs:"):              
    #             new_usrs_line = splitted_line[1:]
    #             if (new_usrs_line ==[]):
    #                 continue # no new users
    #
    #             splitted_line = new_usrs_line[0].split ("\n")[0].split (")")
    #
    #             for usr_entry in splitted_line:
    #                 if (len(usr_entry) <= 1):
    #                     break
    #                 usr_entry = usr_entry.split("(")[1].split (',')
    #
    #                 usr = self.gen_new_usr (usr_id=int(usr_entry[0]))
    #                 usr.is_new = True
    #                 self.CPUAll_single_usr (usr) 
    #                 self.update_S_u(usr, poa_id=int(usr_entry[1])) # Update the list of delay-feasible servers for this usr 
    #                 self.usrs.append (usr)
    #
    #         elif (splitted_line[0] == "old_usrs:"):  
    #             old_usrs_line = splitted_line[1:]
    #
    #             if (old_usrs_line == []): # if the list of old users that moved is empty
    #                 return
    #
    #             splitted_line = self.parse_old_usrs_line(old_usrs_line)
    #             for usr_entry in splitted_line:
    #                 if (len(usr_entry) <= 1):
    #                     break
    #                 usr_entry = usr_entry.split("(")[1].split (',')
    #                 # usr_entry = usr_entry[1].split (',')
    #
    #                 list_of_usr = list(filter (lambda usr : usr.id == int(usr_entry[0]), self.usrs))
    #                 usr = list_of_usr[0]
    #                 self.CPUAll_single_usr (usr) # update usr.B by the new requirements of this usr.
    #                 self.update_S_u(usr, poa_id=int(usr_entry[1])) # Add this usr to the Hs of every server to which it belongs in its new location
    #
    #             # if (self.t == slot_to_dump): 
    #             #     self.dump_usrs_to_poa_file() 
     
    def rd_poa2cell_file (self, poa2cell_file_name):
        """
        Parse an poa2cell file.
        An poa2cell file contains a list of PoAs, and, for each PoA, the cell in which it is located.
        """
        
        self.PoAs = []
        input_file = open ('../res/poa2cell_files/' + poa2cell_file_name, 'r')
        
        for line in input_file:
    
            if (line == "\n" or line.split ("//")[0] == ""):
                continue

            line = line.split ("\n")[0]
            splitted_line = line.split ()
            self.PoAs.append ({'poa' : int(splitted_line[0]), 'cell' : int(splitted_line[1])})
    
    def binary_search_opt (self, output_file, cpu_cap_at_leaf=200, prob_of_target_delay=0.3, sim_len_in_slots=float('inf'), seed=42):
        """
        Binary-search for the minimal rsrce aug' required by an algorithm (not opt) to find a feasible sol.
        As the results of opt are consistent (namely, If a run succeeds with cpu cap X, then in would succeed also with cpu cap X+1), 
        each time opt fails at a single slot, it performs a binary search, until it finds the minimal cpu value required for finding a feasible sol' at that slot.     
        Returns the lowest cpu cap' at the leaf server which still allows finding a feasible sol. 
        """
        self.verbose.append (VERBOSE_CALC_RSRC_AUG) 
        return self.simulate (mode = 'opt', cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots)

    def binary_search_algs (self, output_file, mode, cpu_cap_at_leaf=200, prob_of_target_delay=0.3, sim_len_in_slots=float('inf'), seed=42):
        """
        Binary-search for the minimal rsrce aug' required by an algorithm (not opt) to find a feasible sol.
        As the results for algs' aren't necessarily consistent (namely, a run may succeed with cpu cap X, but fail with cpu cap X+1), 
        for each suggested cpu values, we run the whole trace; a run is considered "sccs" iff it successfully found solutions during the whole trace.    
        Returns the lowest cpu cap' at the leaf server which still allows finding a feasible sol. 
        """ 
        res = self.simulate (mode = mode, cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots, seed=seed)
        if (res != None): # found a feasible solution without a binary search 
            print ('wo binary search, cpu_cap_at_leaf={}' .format (cpu_cap_at_leaf))
            self.print_sol_res_line (output_file)
            return res

        lb = cpu_cap_at_leaf
        ub = inter (cpu_cap_at_leaf * self.max_R)
        res = self.simulate (mode = mode, cpu_cap_at_leaf=ub, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots, seed=seed)
        if (res == None): # found a feasible solution without a binary search 
            print ('Did not find a feasible solution, even with the maximal rsrc aug: cpu_cap_at_leaf={}' .format (ub))
            exit ()
        
        while True:
            if (ub <= lb+1): # the difference between the lb and the ub is at most 1
                cpu_cap_at_leaf = ub
                self.simulate (mode = mode, cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots, seed=seed)
                self.print_sol_res_line (output_file)
                return cpu_cap_at_leaf

            cpu_cap_at_leaf = self.avg_up_and_lb(ub, lb)
            if (self.simulate (mode = mode, cpu_cap_at_leaf=cpu_cap_at_leaf, prob_of_target_delay=prob_of_target_delay, sim_len_in_slots=sim_len_in_slots, seed=seed) != None): 
                ub = cpu_cap_at_leaf
            else: 
                lb = cpu_cap_at_leaf
    
    def run_prob_of_RT_sim_algs (self, poa2cell_file_name, poa_file_name, prob=None):
        """
        Run a simulation where the probability of a RT application varies. 
        Output the minimal resource augmentation required by each alg', and the cost obtained, and the cost obtained at each time slot.
        """       


        print ('Running run_prob_of_RT_sim')
        probabilities = [prob] if (prob!=None) else ([i/10 for i in range (11)])

        output_file = self.gen_RT_prob_sim_output_file (poa2cell_file_name, poa_file_name, 'ourAlg')    
        # To reduce sim' time, lower-bound the required CPU using the values found by sketch pre-runnings 
        if (VERBOSE_MIG_ONLY_CRIT in self.verbose):
            min_cpu_cap_at_leaf_alg = {'Lux'    : {0.0 : 94, 0.1 : 94, 0.2 : 94, 0.3 : 94, 0.4 : 94, 0.5 : 103, 0.6 : 137, 0.7 : 146, 0.8 : 146, 0.9 : 162, 1.0 : 172},
                                       'Monaco' : {0.0 : 936, 0.1 : 936, 0.2 : 972, 0.3 : 961, 0.4 : 1032, 0.5 : 1114, 0.6 : 1317, 0.7 : 1556, 0.8 : 1736, 0.9 : 2000, 1.0 : 2200}} 
        else:
            min_cpu_cap_at_leaf_alg = {'Lux'    : {0.0 : 94, 0.1 : 94, 0.2 : 94, 0.3 : 94, 0.4 : 94, 0.5 : 103, 0.6 : 137, 0.7 : 146, 0.8 : 146, 0.9 : 162, 1.0 : 172},
                                       'Monaco' : {0.0 : 838, 0.1 : 838, 0.2 : 838, 0.3 : 842, 0.4 : 868, 0.5 : 1063, 0.6 : 1283, 0.7 : 1508, 0.8 : 1709, 0.9 : 1989, 1.0 : 2192}} 
        for seed in [40 + delta_sd for delta_sd in range (21) ]:
            for prob_of_target_delay in probabilities:
                self.binary_search_algs(output_file=output_file, mode='ourAlg', cpu_cap_at_leaf=min_cpu_cap_at_leaf_alg[self.city][prob_of_target_delay], prob_of_target_delay=prob_of_target_delay, seed=seed)

        # min_cpu_cap_at_leaf_alg = {'Lux'    : {0.0 : 160, 0.1 : 160, 0.2 : 165, 0.3 : 165, 0.4 : 165, 0.5 : 170, 0.6 : 170, 0.7 : 170, 0.8 : 180, 0.9 : 195, 1.0 : 225},
        #                            'Monaco' : {0.0 : 1100, 0.1 : 1100, 0.2 : 1100, 0.3 : 1100, 0.4 : 1100, 0.5 : 1100, 0.6 : 1100, 0.7 : 1200, 0.8 : 1300, 0.9 : 1600, 1.0 : 1600}} 
        # for seed in [40 + i for i in range (1,4)]:
        #     mode='ffit'
        #     output_file = self.gen_RT_prob_sim_output_file (poa2cell_file_name, poa_file_name, mode=mode)    
        #     for prob_of_target_delay in [i/10 for i in range (11)]:
        #         self.binary_search_algs(output_file=output_file, mode=mode, cpu_cap_at_leaf=min_cpu_cap_at_leaf_alg[self.city][prob_of_target_delay], prob_of_target_delay=prob_of_target_delay, seed=seed)


        # min_cpu_cap_at_leaf_alg = {'Lux'    : {0.0 : 150, 0.1 : 150, 0.2 : 150, 0.3 : 150, 0.4 : 150, 0.5 : 150, 0.6 : 150, 0.7 : 150, 0.8 : 150, 0.9 : 160, 1.0 : 160},
        #                            'Monaco' : {0.0 : 1150, 0.1 : 1150, 0.2 : 1150, 0.3 : 1150, 0.4 : 1150, 0.5 : 1170, 0.6 : 1200, 0.7 : 1400, 0.8 : 1500, 0.9 : 1800, 1.0 : 1800}} 
        # for seed in [40 + i for i in range (6)]:
        #     for mode in ['ffit']: 
        #         output_file = self.gen_RT_prob_sim_output_file (poa2cell_file_name, poa_file_name, mode)    
        #         for prob_of_target_delay in [i/10 for i in range (11)]:
        #             self.binary_search_algs(output_file=output_file, mode=mode, cpu_cap_at_leaf=min_cpu_cap_at_leaf_alg[self.city][prob_of_target_delay], prob_of_target_delay=prob_of_target_delay, seed=seed)

    def run_prob_of_RT_sim_opt (self, poa2cell_file_name, poa_file_name, prob=None):
        """
        Run a simulation where the probability of a RT application varies. 
        If a "prob" input argument is explicitly given, run the simulation with this given prob'
        Else, run the sim for each probability in [0.1, 0.2, ..., 1.0]  
        Output the minimal resource augmentation required by each alg', and the cost obtained, and the cost obtained at each time slot.
        """       

        print ('Running run_prob_of_RT_sim')
        output_file = self.gen_RT_prob_sim_output_file (poa2cell_file_name, poa_file_name, 'opt')
        min_cpu_cap_at_leaf = {'Lux'    : {0.0 : 89, 0.1 : 89, 0.2 : 89, 0.3 : 89, 0.4 : 89, 0.5 : 98, 0.6 : 98, 0.7 : 130, 0.8 : 144, 0.9 : 158, 1.0 : 171},
                               'Monaco' : {0.0 : 836, 0.1 : 836, 0.2 : 836, 0.3 : 840, 0.4 : 866, 0.5 : 1059, 0.6 : 1287, 0.7 : 1505, 0.8 : 1706, 0.9 : 1984, 1.0 : 2188}} 
        probabilities = [prob] if (prob!=None) else ([i/10 for i in range (11)])
        cpu_cap_at_leaf = min_cpu_cap_at_leaf[self.city][0.0]     
        for prob_of_target_delay in probabilities: 
            cpu_cap_at_leaf = self.binary_search_opt(output_file=output_file, cpu_cap_at_leaf=min (cpu_cap_at_leaf, min_cpu_cap_at_leaf[self.city][prob_of_target_delay]), prob_of_target_delay=prob_of_target_delay)
            self.print_sol_res_line (output_file)
    
#######################################################################################################################################
# Functions that are not part of the class
#######################################################################################################################################


def run_cost_vs_rsrc (poa_file_name, poa2cell_file_name, seed=None):
    """
    Run a simulation where the amount of resources varies. 
    Output the cost obtained at each time slot.
    """
    
    print ('Running run_cost_vs_rsrc')
    seeds = [seed] if (seed!=None) else [70 + i for i in range (20)]

    my_simulator = SFC_mig_simulator (poa_file_name=poa_file_name, verbose=[VERBOSE_RES], poa2cell_file_name=poa2cell_file_name)

    # for cpu_cap_at_leaf in [inter (MIN_REQ_CPU[my_simulator.city]['opt']*(1 + i/10)) for i in range(1, 3)]: # simulate for opt's min cpu * [100%, 110%, 120%, ...]
    #     my_simulator.simulate (mode = 'opt', cpu_cap_at_leaf=cpu_cap_at_leaf, seed=40)
    # for cpu_cap_at_leaf in [MIN_REQ_CPU[my_simulator.city]['ourAlg'], MIN_REQ_CPU[my_simulator.city]['ffit'], MIN_REQ_CPU[my_simulator.city]['cpvnf']]:
    #     my_simulator.simulate (mode = 'opt', cpu_cap_at_leaf=cpu_cap_at_leaf)
    
    # for seed in seeds:
    #     for cpu_cap_at_leaf in [inter (MIN_REQ_CPU[my_simulator.city]['opt']*(1 + i/10)) for i in range(21)]: # simulate for opt's min cpu * [100%, 110%, 120%, ...]
    #         my_simulator.simulate (mode = 'ourAlg', cpu_cap_at_leaf=cpu_cap_at_leaf, seed=seed)
    #     for cpu_cap_at_leaf in [MIN_REQ_CPU[my_simulator.city]['ourAlg'], MIN_REQ_CPU[my_simulator.city]['ffit'], MIN_REQ_CPU[my_simulator.city]['cpvnf']]:
    #         my_simulator.simulate (mode = 'ourAlg', cpu_cap_at_leaf=cpu_cap_at_leaf, seed=seed)

    for seed in seeds:
        for mode in ['cpvnf', 'ffit']:
            my_simulator.simulate (mode = mode, cpu_cap_at_leaf=MIN_REQ_CPU[my_simulator.city][mode], seed=seed)
            for cpu_cap_at_leaf in [inter (MIN_REQ_CPU[my_simulator.city]['opt']*(1 + i/10)) for i in range(21)]: # simulate for opt's min cpu * [100%, 110%, 120%, ...]
                if (cpu_cap_at_leaf >= MIN_REQ_CPU[my_simulator.city][mode]):
                    my_simulator.simulate (mode = mode, cpu_cap_at_leaf=cpu_cap_at_leaf, seed=seed)   

def main ():

    poa_file_name      = 'Monaco_0820_0830_1secs_Telecom.poa'       #'Monaco_0730_0830_16secs_Telecom.poa' #'Monaco_0820_0830_1secs_Telecom.poa' #'Lux_0820_0830_1secs_post.poa' #'Monaco_0820_0830_1secs_Telecom.poa' 
    poa2cell_file_name = 'Monaco.Telecom.antloc_192cells.poa2cell'  #'Lux.post.antloc_256cells.poa2cell' #'Monaco.Telecom.antloc_192cells.poa2cell'
    my_simulator = SFC_mig_simulator (poa_file_name=poa_file_name, verbose=[VERBOSE_MIG_ONLY_CRIT], poa2cell_file_name=poa2cell_file_name)
    my_simulator.run_prob_of_RT_sim_algs  (poa_file_name=poa_file_name, poa2cell_file_name=poa2cell_file_name)
    # seed = None
    # if (len (sys.argv)>1):
    #     seed=int(sys.argv[1])   
    # poa_file_name      = 'Monaco_0820_0830_1secs_Telecom.poa'       #'Monaco_0730_0830_16secs_Telecom.poa' #'Monaco_0820_0830_1secs_Telecom.poa' #'Lux_0820_0830_1secs_post.poa' #'Monaco_0820_0830_1secs_Telecom.poa' 
    # poa2cell_file_name = 'Monaco.Telecom.antloc_192cells.poa2cell'  #'Lux.post.antloc_256cells.poa2cell' #'Monaco.Telecom.antloc_192cells.poa2cell'
    # run_cost_vs_rsrc(poa_file_name=poa_file_name, poa2cell_file_name=poa2cell_file_name, seed=seed)

    # print ('Running cost_vs_rsrc')

    # seed=int(sys.argv[1])
    # cpu_cap_at_leaf = inter (MIN_REQ_CPU[my_simulator.city]['opt']*(1 + seed/10)) 
    # my_simulator.simulate (mode = 'ourAlg', cpu_cap_at_leaf=cpu_cap_at_leaf, seed=seed)

    # my_simulator = SFC_mig_simulator (poa_file_name=poa_file_name, verbose=[VERBOSE_RES], poa2cell_file_name=poa2cell_file_name)
    # for cpu_cap_at_leaf in [inter (MIN_REQ_CPU[my_simulator.city]['opt']*(1 + i/10)) for i in range(1, 21)]: # simulate for opt's min cpu * [100%, 110%, 120%, ...]
    #     my_simulator.simulate (mode = 'ourAlg', cpu_cap_at_leaf=cpu_cap_at_leaf, seed=seed)

    # my_simulator = SFC_mig_simulator (poa_file_name=poa_file_name, verbose=[VERBOSE_RES], poa2cell_file_name=poa2cell_file_name)
    # for cpu_cap_at_leaf in [inter (MIN_REQ_CPU[my_simulator.city]['opt']*(1 + i/10)) for i in range(1, 21)]: # simulate for opt's min cpu * [100%, 110%, 120%, ...]
    #     my_simulator.simulate (mode = 'ourAlg', cpu_cap_at_leaf=cpu_cap_at_leaf, seed=seed)
    # my_simulator.simulate (mode = 'ourAlg', cpu_cap_at_leaf=926, seed=99)

if __name__ == "__main__":
    main()
    
