import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

from printf import printf 
import pandas as pd
from pandas._libs.tslibs import period
import pickle

# Indices of fields indicating the settings in a standard ".res" file
t_idx         = 0
mode_idx       = 1
cpu_idx       = 2
prob_idx      = 3
seed_idx      = 4
stts_idx      = 5
num_of_fields = stts_idx+1

num_usrs_idx      = 6
num_crit_usrs_idx = 7

opt_idx   = 0
alg_idx   = 1
ffit_idx  = 2
cpvnf_idx = 3

MARKER_SIZE = 3 #15
LINE_WIDTH  = 3 #4
FONT_SIZE   = 15 #30

class Res_file_parser (object):
    """
    Parse "res" (result) files, and generate plots from them.
    """

    # An inline function. Calculates the total cost at a given time slot.
    # The total cost is the sum of the migration, CPU and link costs.
    # If the length of the slot is 8, we need to multiply the CPU and link cost by 7.5. This is because in 1 minutes (60 seconds), where we ignore the first we have only 7.5 8-seconds solots #$$$ ????        
    calc_cost_of_item = lambda self, item : item['mig_cost'] + (item['cpu_cost'] + item['link_cost']) * (7.5 if self.time_slot_len == 8 else 1)    

    # Calculate the confidence interval, given the avg and the std 
    conf_interval = lambda self, avg, std : [avg - 2*std, avg + 2*std] 

    def __init__ (self):
        """
        Initialize a Res_file_parser, used to parse result files, and generate plots. 
        """
        self.add_plot_opt     = '\t\t\\addplot [color = green, mark=+, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_ourAlg  = '\t\t\\addplot [color = purple, mark=o, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_ffit    = '\t\t\\addplot [color = blue, mark=triangle*, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_cpvnf   = '\t\t\\addplot [color = black, mark = square,      mark options = {mark size = 2, fill = black}, line width = \plotLineWidth] coordinates {\n\t\t'
        self.add_plot_str1    = '\t\t\\addplot [color = blue, mark=square, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.end_add_plot_str = '\n\t\t};'
        self.add_legend_str   = '\n\t\t\\addlegendentry {'
        
        self.add_plot_cpu_cost               = '\n' + self.add_plot_opt 
        self.add_plot_link_cost              = '\n' + self.add_plot_ourAlg
        self.add_plot_mig_cost               = '\n' + self.add_plot_ffit
        self.add_plot_num_of_critical_chains = '\n' + self.add_plot_cpvnf

        self.add_plot_str_dict = {'opt'    : self.add_plot_opt,
                                  'ourAlg' : self.add_plot_ourAlg,
                                  'ffit'   : self.add_plot_ffit,
                                  'cpvnf'  : self.add_plot_cpvnf}

        self.legend_entry_dict = {'opt'    :  'LBound', 
                                  'ourAlg' : 'BUPU', 
                                  'ffit'   : 'F-Fit', #\\ffit',
                                  'cpvnf'  : 'CPVNF'} #\cpvnf'}

        self.color_dict       = {'opt'    : 'green',
                                'ourAlg'  : 'purple',
                                'ffit'    : 'blue',
                                'cpvnf'   : 'black'}
        
        self.markers_dict     = {'opt'    : 'x',
                                'ourAlg'  : 'o',
                                'ffit'    : '^',
                                'cpvnf'   : 's'}
        
        matplotlib.rcParams.update({'font.size': FONT_SIZE})

        self.list_of_dicts   = [] # a list of dictionaries, holding the settings and the results read from result files
      
    def plot_cost_comp_tikz (self):
        """
        Generate a plot of the ratio of critical usrs over time, and of the mig cost over time.   
        """

        # Generate a vector for the x axis (the t line).
        list_of_dicts_of_sd42 = list ([item for item in self.list_of_dicts if item['seed']==42])
        t_min, t_max          = min ([item['t'] for item in list_of_dicts_of_sd42]), max ([item['t'] for item in list_of_dicts_of_sd42])

        num_of_periods     = 10 # number of marker points in the plot 
        period_len         = int( (t_max-t_min+1) / num_of_periods) # Each point will be assigned the avg value, where averaging over period of length period_len
        mig_cost           = np.empty (num_of_periods)
        ratio_of_crit_usrs = np.empty (num_of_periods)
        
        for period in range(num_of_periods): # for every considered period
            res_from_this_period        = list (filter (lambda item : item['t'] >= t_min + period*period_len and item['t'] < t_min + (period+1)*period_len, list_of_dicts_of_sd42))
            if (period==0): # Remove the results of the first slot, which are distorted, as in this slot there cannot be migrations
                del(res_from_this_period[0])
            mig_cost[period]            = np.average ([item['mig_cost'] for item in res_from_this_period])
            ratio_of_crit_usrs [period] = np.average ([item['num_crit_usrs']/item['num_usrs'] for item in res_from_this_period])
            
        self.output_file = open ('../res/cost_comp_{}.dat' .format (self.input_file_name), 'w')
        printf (self.output_file, self.add_plot_mig_cost)
        for period in range (num_of_periods):
            printf (self.output_file, '({:.2f}, {:.2f})' .format (period*period_len, mig_cost[period]))
        printf (self.output_file, '};' + self.add_legend_str + 'mig.}\n')

        printf (self.output_file, self.add_plot_num_of_critical_chains)
        for period in range (num_of_periods):
            printf (self.output_file, '({:.2f}, {:.4f})' .format (period*period_len, ratio_of_crit_usrs[period]))
        printf (self.output_file, '};' + self.add_legend_str + 'Frac. of Critical Chains}\n')
                   
    def gen_vec_for_period (self, vec_for_slot):
        vec_for_period = []
        for period_num in range (self.num_of_periods-1): 
            vec_for_period.append (np.sum([vec_for_slot             [period_num*self.num_of_slots_in_period + i] for i in range (self.num_of_slots_in_period)])) 

        vec_for_period.append (sum (vec_for_slot[self.num_of_slots_in_period*(self.num_of_periods-1):(-1)]))
        return np.array (vec_for_period)         
            
    def parse_vec_line (self, splitted_line):
        """
        parse a vec from an input file, where the vec's format is, e.g.: "[2, 33, 44, 34, 8]" 
        """
        vec      = []
        vec_line = splitted_line.split ("\n")[0]
        vec_line = vec_line.split('[')[1].split(']')[0].split(', ') # remove leading and trailing square brackets
        for item in vec_line:
            vec.append (float(item))
        return np.array(vec)

        
    def parse_file (self, input_file_name, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True):
        """
        Parse a result file, in which each un-ommented line indicates a concrete simulation settings.
        """
        
        self.input_file_name = input_file_name
        self.input_file      = open ("../res/" + input_file_name,  "r")
        lines                = (line.rstrip() for line in self.input_file) # "lines" contains all lines in input file
        lines                = (line for line in lines if line)       # Discard blank lines
        
        for line in lines:
        
            # Discard lines with comments / verbose data
            if (line.split ("//")[0] == ""):
                continue
           
            self.parse_line(line, parse_cost=parse_cost, parse_cost_comps=parse_cost_comps, parse_num_usrs=parse_num_usrs)
            if (self.dict==None): # No new data from this line
                continue
            if (not(self.dict in self.list_of_dicts)):
                self.list_of_dicts.append(self.dict)
                

        self.input_file.close



    def parse_line (self, line, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True):
        """
        Parse a line in a result file. Such a line should begin with a string having several fields, detailing the settings.
        """

        splitted_line = line.split (" | ")
         
        settings          = splitted_line[0]
        splitted_settings = settings.split ("_")

        if len (splitted_settings) != num_of_fields:
            print ("encountered a format error. Splitted line={}\nsplitted settings={}" .format (splitted_line, splitted_settings))
            self.dict = None
            return

        self.dict = {
            "t"         : int   (splitted_settings [t_idx]   .split('t')[1]),
            "mode"       : splitted_settings      [mode_idx],
            "cpu"       : int   (splitted_settings [cpu_idx] .split("cpu")[1]),  
            "prob"      : float (splitted_settings [prob_idx].split("p")   [1]),  
            "seed"      : int   (splitted_settings [seed_idx].split("sd")  [1]),  
            "stts"      : int   (splitted_settings [stts_idx].split("stts")[1]),
        }

        if (parse_cost):
            self.dict["cost"] = float (splitted_line[4].split("=")[1])

        if (parse_cost_comps):        
            self.dict["cpu_cost"]  = float (splitted_line[1].split("=")[1])
            self.dict["link_cost"] = float (splitted_line[2].split("=")[1])
            self.dict["mig_cost" ] = float (splitted_line[3].split("=")[1])            

        if (parse_num_usrs and len(splitted_line) > num_usrs_idx): # Is it a longer line, indicating also the # of usrs, and num of critical usrs?
            self.dict['num_usrs']      = int (splitted_line[num_usrs_idx]     .split("=")[1])
            self.dict['num_crit_usrs'] = int (splitted_line[num_crit_usrs_idx].split("=")[1])

    def gen_filtered_list (self, list_to_filter, min_t = -1, max_t = float('inf'), prob=None, mode = None, cpu = None, stts = -1):
        """
        filters and takes from all the items in a given list (that was read from the res file) only those with the desired parameters value
        The function filters by some parameter only if this parameter is given an input value > 0.
        """
        list_to_filter = list (filter (lambda item : item['t'] >= min_t and item['t'] <= max_t, list_to_filter))    
        if (mode != None):
            list_to_filter = list (filter (lambda item : item['mode']  == mode, list_to_filter))    
        if (cpu != None):
            list_to_filter = list (filter (lambda item : item['cpu']  == cpu, list_to_filter))    
        if (prob != None):
            list_to_filter = list (filter (lambda item : item['prob'] == prob, list_to_filter))    
        if (stts != -1):
            list_to_filter = list (filter (lambda item : item['stts'] == stts, list_to_filter))
        return list_to_filter

    def print_single_tikz_plot (self, list_of_dict, key_to_sort, addplot_str = None, add_legend_str = None, legend_entry = None, y_value = 'cost'):
        """
        Prints a single plot in a tikz format.
        Inputs:
        list_of_dicts - a list of Python dictionaries. 
        key_to_sort - the function sorts the items by this key, e.g.: cache size, uInterval, etc.
        addplot_str - the "add plot" string to be added before each list of points (defining the plot's width, color, etc.).
        addlegend_str - the "add legend" string to be added after each list of points.
        legend_entry - the entry to be written (e.g., 'Opt', 'Alg', etc).
        """
        if (not (addplot_str == None)):
            printf (self.output_file, addplot_str)
        for dict in sorted (list_of_dict, key = lambda i: i[key_to_sort]):
            printf (self.output_file, '({:.4f}, {:.4f})' .format (dict[key_to_sort], dict[y_value]))
        printf (self.output_file, self.end_add_plot_str)
        if (not (add_legend_str == None)): # if the caller requested to print an "add legend" str          
            printf (self.output_file, '\t\t{}{}' .format (self.add_legend_str, legend_entry))    
            printf (self.output_file, '}\n')    
        printf (self.output_file, '\n')    


    def plot_num_of_vehs (self):
        """
        Plot a diagram showing the average num of vehicles in a given area
        """
        # Open input and output files
        input_file  = open ("../res/num_of_vehs_24_every_min_correct.pos", "r")  
        
        t, tot_num_of_vehs, num_of_act_vehs = [], [], []
        for line in input_file:
            
            if (line == "\n" or line.split ("//")[0] == ""):
            
                continue
        
            splitted_line = line.split (" ")
            t.append (float(splitted_line[0].split('=')[1].split(":")[0]))
            tot_num_of_vehs.append (int(splitted_line[1].split('=')[1]))
            num_of_act_vehs.append (int(splitted_line[2].split('=')[1]))
            
        plt.plot (np.array(t)/3600, tot_num_of_vehs)
        plt.show ()
        
    def plot_RT_prob_sim_tikz (self):
        """
        Generating a tikz-plot showing the amount of resource augmentation required, as a function of the probability that a user has tight (RT) delay requirements.
        """
        output_file_name = self.input_file_name + '.dat' 
        self.output_file = open ('../res/{}' .format (output_file_name), 'w')
        for mode in ['ourAlg', 'ffit', 'cpvnf', 'opt']: #['opt', 'ourAlg', 'ffit', 'cpvnf']:
            
            list_of_points = self.gen_filtered_list (self.list_of_dicts, mode=mode, stts=1)
        
            for point in list_of_points:
                point['cpu'] /= 10
            
            self.print_single_tikz_plot (list_of_points, key_to_sort='prob', addplot_str=self.add_plot_str_dict[mode], add_legend_str=self.add_legend_str, legend_entry=self.legend_entry_dict[mode], y_value='cpu')
     
    def plot_RT_prob_sim_python (self, input_file_name=''):
        """
        Generating a python plot showing the amount of resource augmentation required, as a function of the probability that a user has tight (RT) delay requirements.
        Show the conf' intervals.
        """
        
        input_file_name = input_file_name if (input_file_name != '') else self.input_file_name 
        fig, ax = plt.subplots()
        for mode in ['opt', 'ourAlg', 'ffit', 'cpvnf']: #['opt', 'ourAlg', 'ffit', 'cpvnf']:
            
            list_of_points = self.gen_filtered_list(self.list_of_dicts, mode=mode, stts=1) 
        
            x = set () # The x value will hold all the probabilities that appear in the .res file
            for point in list_of_points: # A cpu cap' unit represents 100 MHz --> to represent results by units of GHz, divide the cpu cap' by 10.
                point['cpu'] /= 10
                x.add (point['prob'])
            
            x = sorted (x)
            y = []
            
            for x_val in x: # for each concrete value in the x vector
                
                samples = [item['cpu'] for item in self.gen_filtered_list(list_of_points, prob=x_val)]
                avg = np.average(samples)
                
                [y_lo, y_hi] = self.conf_interval (avg, np.std(samples))
                
                if (x_val==0.3 and mode in ['ffit', 'cpvnf']):
                    print ('mode={}, x_val=0.3, y_hi={:.1f}' .format (mode, y_hi))

                ax.plot ((x_val,x_val), (y_lo, y_hi), color=self.color_dict[mode]) # Plot the confidence interval
                y.append (avg)
            
            ax.plot (x, y, color=self.color_dict[mode], marker=self.markers_dict[mode], markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=self.legend_entry_dict[mode])
        plt.xlabel('Fraction of users with RT requirements')
        plt.ylabel('Min CPU at leaf [GHz]')
        ax.legend () #(loc='upper center', shadow=True, fontsize='x-large')
        plt.xlim (0,1)
            
        plt.savefig ('../res/{}.jpg' .format (input_file_name))
        # plt.show ()            

    def gen_cost_vs_rsrcs_tbl (self, normalize_X = True, slot_len_in_sec=1):
        """
        Plot the cost as a function of the amount of resources (actually, cpu capacity at leaf).
        Possibly normalize the amounts of cpu (the X axis) by either the min' amount of cpu required by opt (LBound) to obtain a feasible sol; 
        and/or normalize the cost (the Y axis) by the costs obtained by opt.   
        """
        
        min_t = 30541
        max_t = 30600
        self.time_slot_len = int(self.input_file_name.split('secs')[0].split('_')[-1])
        prob = 0.3
        self.output_file_name = '../res/{}.dat' .format (self.input_file_name, prob)
        self.output_file      = open (self.output_file_name, "w")
        
        if (normalize_X):
            opt_list = sorted (self.gen_filtered_list (self.list_of_dicts, mode='opt', prob=prob, min_t=min_t, max_t=max_t, stts=1),
                               key = lambda item : item['cpu'])
            cpu_vals = sorted (list (set([item['cpu'] for item in opt_list])))
            if (len (cpu_vals)==0):
                print ('Error: you asked to normalize by opt, but no results of opt exist. Please add first results of opt to the parsed input file.')
                return
            X_norm_factor = cpu_vals[0] # normalize X axis by the minimum cpu
        
        else:
            X_norm_factor = 1

            mode_list = sorted (self.gen_filtered_list (self.list_of_dicts, mode='ourAlg', prob=prob, min_t=min_t, max_t=max_t),key = lambda item : item['cpu'])

        list_of_avg_vals = []        
        
        printf (self.output_file, 'cpu        & LBound        & BUPU        & F-Fit        & CPVNF')        

        for mode in ['opt', 'ourAlg', 'ffit', 'cpvnf']:
            
            mode_list   = sorted (self.gen_filtered_list (self.list_of_dicts, mode=mode, min_t=min_t, max_t=max_t), key = lambda item : item['cpu']) # list of lines with data about this mode
            
            if (len(mode_list)==0): # If no info about this mode - continue
                continue
        
            # Filter-out all results of failed runs 
            failed_runs = [] # failed_runs will include the cpu and seed values for all runs that fail: we've to filter-out these results while calculating the mean cost
            for item in [item for item in mode_list if item['stts']!=1 ]:
                failed_runs.append ({'cpu' : item['cpu'], 'seed' : item['seed']})
            for failed_run in failed_runs: # Remove all results of this failed_run from the list of relevant results 
                mode_list = list (filter (lambda item : not (item['cpu']==failed_run['cpu'] and item['seed']==failed_run['seed']), mode_list))
                        
            for cpu_val in set ([item['cpu'] for item in self.list_of_dicts if item in mode_list]): # list of CPU vals for which the whole run succeeded with this mode' 
                list_of_avg_vals.append ({'mode' : mode, 
                                          'cpu'  : cpu_val, 
                                          'cost' : np.average ([item['cost'] for item in mode_list if item['cpu']==cpu_val]) })

        printf (self.output_file, '\n')
        cpu_vals = sorted (set ([item['cpu'] for item in list_of_avg_vals]))
        min_cpu  = min (cpu_vals)
        for cpu_val in cpu_vals:
            printf (self.output_file, '{:.02f}\t' .format (cpu_val / min_cpu))
            for mode in ['opt', 'ourAlg', 'ffit', 'cpvnf']:
                list_of_val = list (filter (lambda item : item['cpu']==cpu_val and item['mode']==mode, list_of_avg_vals))
                printf (self.output_file, '& $\infty$\t ' if (len(list_of_val)==0) else '& {:.0f}\t ' .format (list_of_val[0]['cost'])) 
            printf (self.output_file, '\\\\ \\hline \n')


    # def calc_n_plot_cost_vs_rsrcs (self, min_t=30541, max_t=30600, prob=0.3, normalize_X = True, slot_len_in_sec=1, min_cpu=None):
    #     """
    #     Calculate the data needed for plotting a graph showing the cost as a function of the amount of resources (actually, cpu capacity at leaf):
    #     * Read the data found in self.list_of_dicts (usually as a result of a previous run of self.parse_file ()).
    #     * Calculate the average cost for each mode and seed during the whole trace for each seed, the confidence intervals, etc. 
    #     * Plot the cost as a function of the amount of resources (actually, cpu capacity at leaf). 
    #       Possibly normalize the amounts of cpu (the X axis) by either the min' amount of cpu required by opt (LBound) to obtain a feasible sol; 
    #       and/or normalize the cost (the Y axis) by the costs obtained by opt.   
    #     """
    #
    #     cost_vs_rsrcs_output_file = open ('../res/{}.dat' .format (self.input_file_name), 'a')
    #     fig, ax = plt.subplots()
    #     for mode in ['opt', 'ourAlg', 'ffit', 'cpvnf']:
    #
    #         mode_list   = sorted (self.gen_filtered_list (self.list_of_dicts, mode=mode, min_t=min_t, max_t=max_t), key = lambda item : item['cpu']) # list of lines with data about this mode
    #
    #         if (len(mode_list)==0): # If no info about this mode - continue
    #             continue
    #
    #         # Filter-out all results of failed runs 
    #         failed_runs = [] # failed_runs will include the cpu and seed values for all runs that fail: we've to filter-out these results while calculating the mean cost
    #         for item in [item for item in mode_list if item['stts']!=1 ]:
    #             failed_runs.append ({'cpu' : item['cpu'], 'seed' : item['seed']})
    #         for failed_run in failed_runs: # Remove all results of this failed_run from the list of relevant results 
    #             mode_list = list (filter (lambda item : not (item['cpu']==failed_run['cpu'] and item['seed']==failed_run['seed']), mode_list))
    #
    #         cpu_vals = sorted (set ([item['cpu'] for item in mode_list]))
    #
    #         if (mode=='opt'):
    #             min_cpu = cpu_vals[0]
    #         x_norm_factor = min_cpu if (normalize_X) else 1 
    #
    #         x = []
    #         y = []
    #         for cpu_val in cpu_vals: 
    #
    #             mode_cpu_list = list (filter (lambda item : item['cpu']==cpu_val, mode_list)) # list of results of runs for this mode, and cpu value
    #             x_val = cpu_val/x_norm_factor # x value for this plotted point / conf' interval
    #             x.append (x_val)               # append the x value of this point to the list of x values of points
    #             avg_cost_of_each_seed = [] # will hold the avg cost of each successful run with a given mode, cpu value, and seed (averaging the cost over all the slots in the trace)
    #             for seed in set ([item['seed'] for item in mode_cpu_list]): # list of seeds for which the whole run succeeded with this mode (algorithm), and this cpu val
    #                 avg_cost_of_each_seed.append (np.average ([item['cost'] for item in mode_cpu_list if item['seed']==seed]))                    
    #
    #             avg_cost_of_all_seeds = np.average (avg_cost_of_each_seed) 
    #             [y_lo, y_hi]          = self.conf_interval (avg_cost_of_all_seeds, np.std (avg_cost_of_each_seed)) # low, high y values for this plotted conf' interval    
    #             printf (cost_vs_rsrcs_output_file, 'mode={}, cpu_val={}, avg_cost_of_each_seed={}, y_lo={:.0f}, y_hi={:.0f}' .format (mode, cpu_val, avg_cost_of_each_seed, y_lo, y_hi))
    #
    #             ax.plot ((x_val,x_val), (y_lo, y_hi), color=self.color_dict[mode]) # Plot the confidence interval for this mode and cpu_val
    #             y.append (avg_cost_of_all_seeds)
    #
    #         ax.plot (x, y, color=self.color_dict[mode], marker=self.markers_dict[mode], markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=self.legend_entry_dict[mode])                
    #
    #
    #
    #     plt.xlabel(r'$C_{cpu} / \hat{C}_{cpu}$')
    #     plt.ylabel('Norm Avg. Cost')
    #     plt.xlim (1,3)
    #     # plt.yscale ('log')
    #     plt.legend ()
    #
    #     plt.tight_layout()
    #     plt.savefig ('../res/cost_by_rsrc_{}.jpg' .format (self.input_file_name), dpi=100)
    #
    #     # plt.ylim (390000,450000)
    #     # plt.xlim (2.4,3.1)
    #     # plt.savefig ('../res/cost_by_rsrc_{}_zoomed.jpg' .format (self.input_file_name), dpi=99)

    def plot_cost_vs_rsrcs (self, pickle_input_file_name, min_t=30541, max_t=30600, prob=0.3, normalize_X = True, slot_len_in_sec=1, min_cpu=None):
        """
        Plot a Python graph showing the cost as a function of the amount of resources (actually, cpu capacity at leaf):
        * Read the required pickled data from an input file.
        * Plot the cost as a function of the amount of resources (actually, cpu capacity at leaf). 
          Possibly normalize the amounts of cpu (the X axis) by either the min' amount of cpu required by opt (LBound) to obtain a feasible sol; 
          and/or normalize the cost (the Y axis) by the costs obtained by opt.   
        """
        
        self.cost_vs_rsrc_data = pd.read_pickle(r'../res/{}' .format (pickle_input_file_name))

        fig, ax = plt.subplots()

        for mode in ['opt', 'ourAlg', 'ffit', 'cpvnf']:
        
        
            mode_list = list (filter (lambda item : item['mode']==mode, self.cost_vs_rsrc_data)) # assign into mode_list all the data for the relevant mode.
        
            if (len(mode_list)==0): # If no info about this mode - continue
                continue
        
            cpu_vals = sorted (set ([item['cpu'] for item in mode_list]))
        
            # If requested, normalize all cpu values (X axis) by the smallest CPU required by opt for finding a feasible sol.
            if (mode=='opt'):  
                min_cpu = cpu_vals[0]
            x_norm_factor = min_cpu if (normalize_X) else 1 

            x, y = [], []        
            for cpu_val in cpu_vals: 
        
                list_of_item = list (filter (lambda item : item['cpu']==cpu_val, mode_list)) # all items with this cpu value, of this mode (the list should usually include a single item)
                if (len(list_of_item)!=1):
                    print ('Warning: len(list_of_item)=={}' .format (len(list_of_item)))
                item  = list_of_item[0]
                x_val = cpu_val/x_norm_factor
                x.append (x_val)
                ax.plot ((x_val, x_val), (item['y_lo'], item['y_hi']), color=self.color_dict[mode]) # Plot the confidence interval for this mode and cpu_val
                y.append (item['y_avg'])
        
            ax.plot (x, y, color=self.color_dict[mode], marker=self.markers_dict[mode], markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=self.legend_entry_dict[mode])                
        
        
        
        plt.xlabel(r'$C_{cpu} / \hat{C}_{cpu}$')
        plt.ylabel('Norm Avg. Cost')
        plt.xlim (1,3)
        # plt.yscale ('log')
        plt.legend ()
        
        plt.tight_layout()
        plt.savefig ('../res/cost_by_rsrc_{}.jpg' .format (pickle_input_file_name), dpi=100)

        # plt.ylim (390000,450000)
        # plt.xlim (2.4,3.1)
        # plt.savefig ('../res/cost_by_rsrc_{}_zoomed.jpg' .format (self.input_file_name), dpi=99)
    def calc_cost_vs_rsrcs (self, pickle_input_file_name='', min_t=30541, max_t=30600, prob=0.3):
        """
        Calculate the data needed for plotting a graph showing the cost as a function of the amount of resources (actually, cpu capacity at leaf):
        * Read the data found in self.list_of_dicts (usually as a result of a previous run of self.parse_file ()).
        * Calculate the average cost for each mode and seed during the whole trace for each seed, the confidence intervals, etc. 
        * Save the processed data into self.cost_vs_rsrc_data. 
        """
    
        if (pickle_input_file_name==''):
            self.cost_vs_rsrc_data = []
        else:
            self.cost_vs_rsrc_data = pd.read_pickle(r'../res/{}' .format (pickle_input_file_name))
    
        for mode in ['opt', 'ourAlg', 'ffit', 'cpvnf']:
    
            cost_vs_rsrc_data_of_this_mode = []
    
            mode_list = sorted (self.gen_filtered_list (self.list_of_dicts, prob=prob, mode=mode, min_t=min_t, max_t=max_t), key = lambda item : item['cpu']) # list of lines with data about this mode
    
            if (len(mode_list)==0): # If no info about this mode - continue
                continue
    
            # Filter-out all results of failed runs 
            failed_runs = [] # failed_runs will include the cpu and seed values for all runs that fail: we've to filter-out these results while calculating the mean cost
            for item in [item for item in mode_list if item['stts']!=1 ]:
                failed_runs.append ({'cpu' : item['cpu'], 'seed' : item['seed']})
            for failed_run in failed_runs: # Remove all results of this failed_run from the list of relevant results 
                mode_list = list (filter (lambda item : not (item['cpu']==failed_run['cpu'] and item['seed']==failed_run['seed']), mode_list))
    
            cpu_vals = sorted (set ([item['cpu'] for item in mode_list]))
    
            for cpu_val in cpu_vals: 
                
                mode_cpu_list = list (filter (lambda item : item['cpu']==cpu_val, mode_list)) # list of results of runs for this mode, and cpu value
                avg_cost_of_each_seed = [] # will hold the avg cost of each successful run with a given mode, cpu value, and seed (averaging the cost over all the slots in the trace)
                for seed in set ([item['seed'] for item in mode_cpu_list]): # list of seeds for which the whole run succeeded with this mode (algorithm), and this cpu val
                    avg_cost_of_each_seed.append (np.average ([item['cost'] for item in mode_cpu_list if item['seed']==seed]))                    
    
                avg_cost_of_all_seeds = np.average (avg_cost_of_each_seed)
                [y_lo, y_hi]          = self.conf_interval (avg_cost_of_all_seeds, np.std (avg_cost_of_each_seed)) # low, high y values for this plotted conf' interval
                
                cost_vs_rsrc_data_of_this_mode.append ({'cpu' : cpu_val, 'y_lo' : y_lo, 'y_hi' : y_hi, 'y_avg' : avg_cost_of_all_seeds,'num_of_seeds' : len(avg_cost_of_each_seed)})
            
            # Add this new calculated point to the ds. Avoid duplications of point.
            for point in sorted (cost_vs_rsrc_data_of_this_mode, key = lambda point : point['cpu']):
                if (point not in self.cost_vs_rsrc_data):
                    self.cost_vs_rsrc_data.append (point)
                point['mode'] = mode
        
        # store the data as binary data stream
        self.cost_vs_rsrc_data_file_name = '{}.data' .format (self.input_file_name.split('.res')[0]) 
        with open('../res/' + self.cost_vs_rsrc_data_file_name, 'wb') as cost_vs_rsrc_data_file:
            pickle.dump(self.cost_vs_rsrc_data, cost_vs_rsrc_data_file)
        # print (self.cost_vs_rsrc_data)


if __name__ == '__main__':

    my_res_file_parser = Res_file_parser ()
    
    # my_res_file_parser.parse_file ('RT_prob_sim_Lux.post.antloc_256cells.poa2cell_Lux_0820_0830_1secs_post.poa.res', parse_cost=False, parse_cost_comps=False, parse_num_usrs=False) #('Monaco_0730_0830_16secs_Telecom_p0.3_ourAlg.res')# ('RT_prob_sim_Monaco.Telecom.antloc_192cells.poa2cell_Monaco_0820_0830_1secs_Telecom.poa.res', parse_cost=True, parse_cost_comps=False, parse_num_usrs=False)
    # my_res_file_parser.plot_RT_prob_sim_python()

    for file_name in ['Monaco_0820_0830_1secs_Telecom_p0.3_ffit.res', 'Monaco_0820_0830_1secs_Telecom_p0.3_cpvnf.res']:
        my_res_file_parser.parse_file (file_name, parse_cost=True, parse_cost_comps=False, parse_num_usrs=False) #('Monaco_0730_0830_16secs_Telecom_p0.3_ourAlg.res')# ('RT_prob_sim_Monaco.Telecom.antloc_192cells.poa2cell_Monaco_0820_0830_1secs_Telecom.poa.res', parse_cost=True, parse_cost_comps=False, parse_num_usrs=False)
    pickle_input_file_name = 'Monaco_0820_0830_1secs_Telecom_p0.3_opt.data' 
    my_res_file_parser.calc_cost_vs_rsrcs (pickle_input_file_name=pickle_input_file_name)
    # my_res_file_parser.plot_cost_vs_rsrcs (pickle_input_file_name=pickle_input_file_name)

    # my_res_file_parser.parse_file ('Monaco_0730_0830_16secs_Telecom_p0.3_ourAlg_sd42.res', parse_cost=True, parse_cost_comps=True, parse_num_usrs=True)  
    # my_res_file_parser.plot_cost_comp_tikz () 
    

    
    # my_res_file_parser.plot_cost_vs_rsrcs (normalize_X=True, slot_len_in_sec=float(input_file_name.split('sec')[0].split('_')[-1]), X_norm_factor=X_norm_factor)
# ncountered a format error. Splitted line=['| num_usrs=8114', 'num_crit_usrs=28']
# splitted settings=['| num', 'usrs=8114']            
    
