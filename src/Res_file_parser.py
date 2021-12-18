import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np, scipy.stats as st, pandas as pd
from pandas._libs.tslibs import period

# import math

from printf import printf, printFigToPdf 
import pickle

# Indices of fields indicating the settings in a standard ".res" file
t_idx         = 0
mode_idx      = 1
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

MARKER_SIZE             = 1 
MARKER_SIZE_SMALL       = 1
LINE_WIDTH              = 3 
LINE_WIDTH_SMALL        = 1 
FONT_SIZE               = 20
FONT_SIZE_SMALL         = 5
LEGEND_FONT_SIZE        = 10
LEGEND_FONT_SIZE_SMALL  = 5 

UNIFORM_CHAIN_MIG_COST = 600

# Parse the len of the time slot simulated, from the given string
find_time_slot_len = lambda string : int(string.split('secs')[0].split('_')[-1])

# Understand which city's data are these, based on the input file name 
parse_city_from_input_file_name = lambda input_file_name : input_file_name.split ('_')[0]

class Res_file_parser (object):
    """
    Parse "res" (result) files, and generate plots from them.
    """

    # Old (wrong?) calculation of the conf' interval
    # conf_interval = lambda self, avg, std : [avg - 2*std, avg + 2*std] 
    
    # Calculate the confidence interval of an array of values ar, given its avg. Based on 
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    conf_interval = lambda self, ar, avg, conf_lvl=0.99 : st.t.interval (conf_lvl, len(ar)-1, loc=avg, scale=st.sem(ar)) if np.std(ar)>0 else [avg, avg]
   
    #mfc='none' makes the markers empty.

    # Understand which city's data are these, based on the input file name 
    # parse_city_from_input_file_name = lambda self, input_file_name : input_file_name.split ('_')[0]

    # Find the length of the time slot based on the input file name
    parse_T_from_input_file_name = lambda self, input_file_name : int (input_file_name.split ('secs')[0].split('_')[-1])

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

        self.legend_entry_dict = {'opt'     :  'LBound', 
                                  'ourAlg'  : 'BUPU', 
                                  'ffit'    : 'F-Fit', #\\ffit',
                                  'cpvnf'   : 'CPVNF', #\cpvnf'}
                                  'ourAlgC' : 'BUPUmoc', 
                                  'ffitC'   : 'F-Fitmoc', #\\ffit',
                                  'cpvnfC'  : 'CPVNFmoc'} #\cpvnf'}

        self.color_dict       = {'opt'    : 'green',
                                'ourAlg'  : 'purple',
                                'ffit'    : 'blue',
                                'cpvnf'   : 'black',
                                'ourAlgC' : 'purple',
                                'ffitC'   : 'blue',
                                'cpvnfC'  : 'black'}
        
        self.markers_dict     = {'opt'    : 'x',
                                'ourAlg'  : 'o',
                                'ffit'    : '^',
                                'cpvnf'   : 's',
                                'ourAlgC' : 'h',
                                'ffitC'   : 'v',
                                'cpvnfC'  : 'd'}
        
        # matplotlib.rcParams.update({'font.size': FONT_SIZE})

        self.list_of_dicts   = [] # a list of dictionaries, holding the settings and the results read from result files
      
    def my_plot (self, ax, x, y, mode='ourAlg', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=None): 
        
        """
        Plot a single x, y, python line, with the required settings (colors, markers etc).
        """ 
    
        color = self.color_dict[mode] if (color==None) else color  
    
        if (mode in ['ourAlgC', 'cpvnfC', 'ffitC']):
            ax.plot (x, y, color=color, marker=self.markers_dict[mode], markersize=markersize, linewidth=linewidth, label=self.legend_entry_dict[mode], mfc='none', linestyle='dashed') 
        else:
            ax.plot (x, y, color=color, marker=self.markers_dict[mode], markersize=markersize, linewidth=linewidth, label=self.legend_entry_dict[mode], mfc='none')


    
    def plot_lin_reg (self, x, y, ax):
        """
        Plot a linear regression for the given scatter 
        """
        m, b = np.polyfit (x, y,  1) # linear reg. parameters: m is the slope, b is the constant
        ax.plot (x, m*x+b, linewidth=LINE_WIDTH_SMALL)

    def plot_tot_num_of_vehs_per_slot (self, pcl_input_file_names):
        """
        Plots the number of vehicles in the simulated area as a function of time.
        """  
        
        t = range (3600)
        
        _, ax = plt.subplots ()
        ax.set_xlabel ('Time[s]')
        ax.set_ylabel ('# of Vehicles')

        for pcl_input_file_name in pcl_input_file_names:
            num_vehs_in_slot = pd.read_pickle(r'../res/{}' .format (pcl_input_file_name))
            city = parse_city_from_input_file_name (pcl_input_file_name)
            
            ax.plot (t, num_vehs_in_slot, color = 'black' if city=='Lux' else 'blue', marker=None, linewidth=LINE_WIDTH, label=city if city=='Monaco' else 'Luxembourg')
            plt.xlim (0, 3600)
            plt.ylim (0)
        
        ax.legend (fontsize=22, loc='center') 
        # plt.show ()

        plt.savefig ('../res/tot_num_of_vehs_0730_0830.pdf', bbox_inches='tight')

    def dump_self_list_of_dicts_to_pcl (self, res_file_name):
        """
        Dump all the data within self.list_of_dicts into a .pcl file, whose name (excluding extenstion) is the same as the given res_file_name
        """             
        with open('../res/{}.pcl' .format (res_file_name.split('.res')[0]), 'wb') as pcl_file:
            pickle.dump(self.list_of_dicts, pcl_file)
      
    def plot_num_crit_n_mig_vs_num_vehs (self, cpu, reshuffle, res_file_to_parse=None, pcl_input_file_name=None):
        """
        Generate a plot of the ratio of critical usrs over time, and of the mig cost over time.
        The output plot may be either tikz, and/or python.
        Inputs: 
        cpu - the plot will consider only result obtained using this cpu value at leaf servers.
        reshuffle - a binary variable. If true, the suffix of the generated output file is 'resh.pdf'. Else, the suffix is 'MOC.pdf' (MOC=Migrate Only Critical chains).
        res_file_to_parse - when given, the function parses this '.res' file, and then used the data for the plot.
        pcl_input_File_name - if no res_file_to_parse is given, the function reads the input from this file name.   
        """
        
        if (res_file_to_parse != None):
            self.parse_file(res_file_to_parse, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True)
            self.dump_self_list_of_dicts_to_pcl (res_file_name = res_file_to_parse)
            self.city = parse_city_from_input_file_name (res_file_to_parse)
        elif (pcl_input_file_name != None):
            self.list_of_dicts = pd.read_pickle(r'../res/{}' .format (pcl_input_file_name))
            self.city = parse_city_from_input_file_name (pcl_input_file_name)

        # filter-out all the data relating to cpu values other than the requested one
        self.list_of_dicts = [item for item in self.list_of_dicts if item['cpu']==cpu]
        print ('cpu={}' .format (cpu))
        
        num_vehs_in_slot     = np.array (pd.read_pickle(r'../res/{}_0730_0830_1secs_num_of_vehs.pcl' .format (self.city)), dtype='int16')
        num_vehs_in_slot_set = set (num_vehs_in_slot)

        matplotlib.rcParams.update({'font.size': FONT_SIZE_SMALL, 
                                    'legend.fontsize': LEGEND_FONT_SIZE_SMALL,
                                    'xtick.labelsize':FONT_SIZE_SMALL,
                                    'ytick.labelsize':FONT_SIZE_SMALL,
                                    'axes.labelsize': FONT_SIZE_SMALL,
                                    'axes.titlesize':FONT_SIZE_SMALL,
                                    })
        _, ax = plt.subplots (4)
        num_crit_sctr, ratio_crit_sctr, num_mig_sctr, ratio_mig_sctr = ax[0], ax[1], ax[2], ax[3]
         
        avg_num_of_migrations    = []    
        avg_ratio_of_migrations  = [] 
        avg_num_of_crit_chains   = [] 
        avg_ratio_of_crit_chains = []
        x                        = []
        
        for num_veh in num_vehs_in_slot_set: # for each distinct value of the number of vehicles
            
            # relevant_slot will hold all the slots in which the number of vehicles is exactly num_vehs. 
            # Add 27001 to the slots, as the array read from the pickle begins by 0, while the first simulated slot with mig' is 27001
            relevant_slots = [(i + 27001) for i, x in enumerate(num_vehs_in_slot) if x == num_veh]  
            num_vehs_in_slot     = [item  for item in num_vehs_in_slot]
            res_from_these_slots = list (filter (lambda item : item['t'] in relevant_slots, self.list_of_dicts))
            if (len (res_from_these_slots)==0): # No results from the relevant slots
                continue

            x. append (num_veh) # append this num of vehs only now, when we're sure that there exist results for the relevant slots
            avg_num_of_crit_chains.  append (np.average ([item['num_crit_usrs']                  for item in res_from_these_slots]))
            avg_ratio_of_crit_chains.append (np.average ([item['num_crit_usrs']/item['num_usrs'] for item in res_from_these_slots]))
            avg_num_of_migrations.   append (np.average ([item['mig_cost']                       for item in res_from_these_slots])  / UNIFORM_CHAIN_MIG_COST)
            avg_ratio_of_migrations. append (np.average ([item['mig_cost']/item['num_usrs']      for item in res_from_these_slots])  / UNIFORM_CHAIN_MIG_COST)

        num_crit_sctr.  scatter (x, avg_num_of_crit_chains,   s=MARKER_SIZE_SMALL, c='black', marker='o')
        ratio_crit_sctr.scatter (x, avg_ratio_of_crit_chains, s=MARKER_SIZE_SMALL, c='black', marker='o')
        num_mig_sctr.   scatter (x, avg_num_of_migrations,    s=MARKER_SIZE_SMALL, c='black', marker='o')
        ratio_mig_sctr. scatter (x, avg_ratio_of_migrations,  s=MARKER_SIZE_SMALL, c='black', marker='o')
        
        plt.locator_params(nbins=4)
        self.plot_lin_reg (x=np.array(x), y=avg_num_of_crit_chains,   ax=num_crit_sctr)
        self.plot_lin_reg (x=np.array(x), y=avg_ratio_of_crit_chains, ax=ratio_crit_sctr)
        self.plot_lin_reg (x=np.array(x), y=avg_num_of_migrations,    ax=num_mig_sctr)
        self.plot_lin_reg (x=np.array(x), y=avg_ratio_of_migrations,  ax=ratio_mig_sctr)

        # ax[0].set (ylabel='# of Critical Chains')
        num_crit_sctr.  set (ylabel='# of Critical Chains')
        ratio_crit_sctr.set (ylabel='Ratio of Critical Chains')
        num_mig_sctr.   set (ylabel='# of Mig. Chains')
        # num_mig_sctr.   set_yticks ([0, 200, 400, 600, 800])
        ratio_mig_sctr. set (ylabel='# of Mig. Chains / # of Chains')
        ratio_mig_sctr. set (xlabel='# of Vehicles')
        printFigToPdf ('../res/{}_mig_vs_num_vehs_{}' .format (self.city, 'resh' if reshuffle else 'MOC'))

    def plot_cost_comp (self, plot_tikz=False, plot_python=True, normalize=False, plot_only_crit=True):
        """
        Generate a plot of the ratio of critical usrs over time, and of the mig cost over time.
        The output plot may be either tikz, and/or python.   
        """

        # Generate a vector for the x axis (the t line).        
        t_min, t_max          = min ([item['t'] for item in self.list_of_dicts]), max ([item['t'] for item in self.list_of_dicts])

        num_of_periods     = 10 # number of marker points in the plot 
        period_len         = int( (t_max-t_min+1) / num_of_periods) # Each point will be assigned the avg value, where averaging over period of length period_len
        
        if (normalize):
            mig_cost, ratio_of_crit_usrs          = np.empty (num_of_periods), np.empty (num_of_periods)
        else:
            num_of_crit_chains, num_of_migrations = np.empty (num_of_periods), np.empty (num_of_periods)
        
        for period in range(num_of_periods): # for every considered period
            res_from_this_period        = list (filter (lambda item : item['t'] >= t_min + period*period_len and item['t'] < t_min + (period+1)*period_len, self.list_of_dicts))
            if (period==0): # Remove the results of the first slot, which are distorted, as in this slot there cannot be migrations
                del(res_from_this_period[0])
            if (normalize):
                mig_cost           [period] = np.average ([item['mig_cost']      / (item['num_usrs']*self.time_slot_len) for item in res_from_this_period])
                ratio_of_crit_usrs [period] = np.average ([item['num_crit_usrs'] / item['num_usrs']                      for item in res_from_this_period])
            else:
                num_of_migrations  [period] = np.sum     ([item['mig_cost']                                              for item in res_from_this_period]) / UNIFORM_CHAIN_MIG_COST
                num_of_crit_chains [period] = np.sum     ([item['num_crit_usrs']                                         for item in res_from_this_period]) 
                
        x = [period*period_len for period in range (num_of_periods)]
        if (plot_python):
            
            _, y1_axis = plt.subplots()
            if (not (plot_only_crit)):
                y2_axis    = y1_axis.twinx()            
                y2_color   = 'blue'
                y2_axis.set_xlabel('Time[s]')
                # y1_axis.set_xlabel('Time[s]',  color=y2_color)
                y2_axis.tick_params (axis='y', colors=y2_color)
            
            if (normalize):
                # Tune maximal values for the Y axes
                Y_LIM_MIG_COST           = {'Lux' : 260, 'Monaco' : 60}
                Y_LIM_RATIO_OF_CRIT_USRS = {'Lux' : 0.5, 'Monaco' : 0.1}
                y1_axis.set_ylabel('Frac. of Critical Chains', color='black')
                line1 = y1_axis.plot (x, ratio_of_crit_usrs, color=y2_color, marker='x', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Critical Chains')
                if (not (plot_only_crit)):
                    y2_axis.set_ylabel('Norm. Mig. Cost', color=y2_color)
                    line2 = y2_axis.plot (x, mig_cost,           color='black',  marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Norm. Mig. Cost')
                    y2_axis.          set_ylim (0, Y_LIM_MIG_COST          [self.city])
                if (self.city=='Monaco'):
                    y1_axis.set_yscale ('log')
                    y1_axis.set_ylim (0.001, Y_LIM_RATIO_OF_CRIT_USRS[self.city])
                else:
                    y1_axis.set_ylim (0, Y_LIM_RATIO_OF_CRIT_USRS[self.city])
                # y1_axis.set_ylim (0, Y_LIM_RATIO_OF_CRIT_USRS[self.city])
    
            else:
                y1_axis.set_ylabel('# of Critical Chains', color='black')
                line1 = y1_axis.plot (x, num_of_crit_chains, color='black',  marker='x', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='# of Critical Chains')
                if (not (plot_only_crit)):
                    y2_axis.set_ylabel('# of Migrated Chains', color=y2_color)
                    line2 = y2_axis.plot (x, num_of_migrations,  color=y2_color, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='# of Migrated Chains')
                
            lines = line1 if (plot_only_crit) else line1 + line2
            plt.xlim (0)
            
            plt.legend (lines, [line.get_label() for line in lines], loc='upper center', fontsize=LEGEND_FONT_SIZE)

            plt.savefig ('../res/cost_comp_{}.pdf' .format (self.input_file_name), bbox_inches='tight')
            
        if (plot_tikz):
            self.output_file = open ('../res/cost_comp_{}.dat' .format (self.input_file_name), 'w')
            printf (self.output_file, self.add_plot_mig_cost)
            for period in range (num_of_periods):
                printf (self.output_file, '({:.2f}, {:.2f})' .format (x[period],mig_cost[period]))
            printf (self.output_file, '};' + self.add_legend_str + 'mig.}\n')
    
            printf (self.output_file, self.add_plot_num_of_critical_chains)
            for period in range (num_of_periods):
                printf (self.output_file, '({:.2f}, {:.4f})' .format (x[period], ratio_of_crit_usrs[period]))
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
        
        self.city = parse_city_from_input_file_name(input_file_name)
        print ('city is ', self.city)
        self.input_file_name = input_file_name
        self.time_slot_len   = find_time_slot_len (self.input_file_name) 
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
            "mode"      : splitted_settings      [mode_idx],
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
        for _ in sorted (list_of_dict, key = lambda i: i[key_to_sort]):
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
        
    # def plot_RT_prob_sim_tikz (self):
    #     """
    #     Generating a tikz-plot showing the amount of resource augmentation required, as a function of the probability that a user has tight (RT) delay requirements.
    #     """
    #     output_file_name = self.input_file_name + '.dat' 
    #     self.output_file = open ('../res/{}' .format (output_file_name), 'w')
    #     for mode in ['ourAlg', 'ffit', 'cpvnf', 'opt']: #['opt', 'ourAlg', 'ffit', 'cpvnf']:
    #
    #         list_of_points = self.gen_filtered_list (self.list_of_dicts, mode=mode, stts=1)
    #
    #         for point in list_of_points:
    #             point['cpu'] /= 10
    #
    #         self.print_single_tikz_plot (list_of_points, key_to_sort='prob', addplot_str=self.add_plot_str_dict[mode], add_legend_str=self.add_legend_str, legend_entry=self.legend_entry_dict[mode], y_value='cpu')
     
    def plot_RT_prob_sim_python (self, input_file_name=None):
        """
        Generating a python plot showing the amount of resource augmentation required, as a function of the probability that a user has tight (RT) delay requirements.
        Show the conf' intervals.
        When given an input file name, parse it first.
        If no input_file_name is given, the function assumes that previously to calling it, an input file was parsed.
        """
        
        if (input_file_name != None):
            self.parse_file(input_file_name, parse_cost=False, parse_cost_comps=False, parse_num_usrs=False)
        input_file_name = input_file_name if (input_file_name != None) else self.input_file_name 
        _, ax = plt.subplots()
        for mode in ['opt', 'ourAlgC', 'ffitC', 'cpvnfC']: #['opt', 'ourAlg', 'ffit', 'cpvnf']:
            
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
                
                [y_lo, y_hi] = self.conf_interval (samples, avg)
                
                if (x_val==0.3 and mode in ['ffit', 'cpvnf', 'ourAlgC']):
                    print ('mode={}, x_val=0.3, y_hi={:.1f}' .format (mode, y_hi))

                ax.plot ((x_val,x_val), (y_lo, y_hi), color=self.color_dict[mode]) # Plot the confidence interval
                
                y.append (avg)
            
            self.my_plot (ax, x, y, mode)
        plt.xlabel('Fraction of Users with RT Requirements')
        plt.ylabel('Min CPU at leaf [GHz]')
        ax.legend (ncol=2, fontsize=LEGEND_FONT_SIZE) #(loc='upper center', shadow=True, fontsize='x-large')
        plt.xlim (0,1)
            
        plt.savefig ('../res/{}.pdf' .format (input_file_name), bbox_inches='tight')

    def gen_cost_vs_rsrc_tbl (self, normalize_X = True, slot_len_in_sec=1):
        """
        Plot the cost as a function of the amount of resources (actually, cpu capacity at leaf).
        Possibly normalize the amounts of cpu (the X axis) by either the min' amount of cpu required by opt (LBound) to obtain a feasible sol; 
        and/or normalize the cost (the Y axis) by the costs obtained by opt.   
        """
        
        min_t = 30541
        max_t = 30600
        self.time_slot_len = find_time_slot_len(self.input_file_name)
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

    def plot_cost_vs_rsrc (self, pcl_input_file_name, normalize_X = True, min_cpu=None):
        """
        Plot a Python graph showing the cost as a function of the amount of resources (actually, cpu capacity at leaf):
        * Read the required pickled data from an input file.
        * Plot the cost as a function of the amount of resources (actually, cpu capacity at leaf). 
          Possibly normalize the amounts of cpu (the X axis) by either the min' amount of cpu required by opt (LBound) to obtain a feasible sol; 
          and/or normalize the cost (the Y axis) by the costs obtained by opt.   
        """
        
        if (find_time_slot_len(pcl_input_file_name)!=1):
            print ('Error: currently, plot_cost_vs_rsrc runs only on slot_len=1 sec')
            return
        
        self.cost_vs_rsrc_data = pd.read_pickle(r'../res/{}' .format (pcl_input_file_name))

        _, ax = plt.subplots()

        for mode in ['opt', 'ourAlg', 'ffit', 'cpvnf']:       
        
            mode_list = list (filter (lambda item : item['mode']==mode, self.cost_vs_rsrc_data)) # assign into mode_list all the data for the relevant mode.
        
            if (len(mode_list)==0): # If no info about this mode - continue
                continue
        
            cpu_vals = sorted (set ([item['cpu'] for item in mode_list]))
        
            # If requested, normalize all cpu values (X axis) by the smallest CPU required by opt for finding a feasible sol.
            if (mode=='opt'):  
                min_cpu = cpu_vals[0]
            x_norm_factor = min_cpu if (normalize_X) else 1 

            x = []
            y = []        
            for cpu_val in cpu_vals: 
        
                list_of_item = list (filter (lambda item : item['cpu']==cpu_val, mode_list)) # all items with this cpu value, of this mode (the list should usually include a single item)
                if (len(list_of_item)!=1):
                    print ('Warning: len(list_of_item)=={}' .format (len(list_of_item)))
                item  = list_of_item[0]
                x_val = cpu_val/x_norm_factor
                x.append (x_val)
                ax.plot ((x_val, x_val), (item['y_lo'], item['y_hi']), color=self.color_dict[mode]) # Plot the confidence interval for this mode and cpu_val
                y.append (item['y_avg'])
        
            self.my_plot (ax, x, y, mode)
        
        plt.xlabel(r'$C_{cpu} / \hat{C}_{cpu}$')
        plt.ylabel('Avg. Cost')
        plt.xlim (1, 3)
        plt.ylim (0, 2000000)
        ax.legend (ncol=2, fontsize=LEGEND_FONT_SIZE, loc='upper right') #(loc='upper center', shadow=True, fontsize='x-large')
        
        plt.tight_layout()
        plt.savefig ('../res/cost_vs_rsrc_{}.pdf' .format (pcl_input_file_name.split('.pcl')[0]), bbox_inches='tight')

    def calc_cost_vs_rsrc (self, pcl_input_file_name=None, res_input_file_names=None, min_t=30001, max_t=30600, prob=0.3):
        """
        Calculate the data needed for plotting a graph showing the cost as a function of the amount of resources (actually, cpu capacity at leaf):
        * Optional inputs: 
            .pcl file, containing self.list_of_dicts (usually as a result of a previous run of self.parse_file ()).
            A list of .res files, containing the results of a run.
            At least one file (either .pcl, or .res file) should be given       
        * Calculate the average cost for each mode and seed during the whole trace for each seed, the confidence intervals, etc. 
        * Save the (pickled) processed data into self.cost_vs_rsrc_data.pcl.
        * Returns the file name to which it saved the pickled results. 
        """
        
        # Caller must provide either a .pcl input file, or a .res input file
        if (pcl_input_file_name==None and res_input_file_names==None):
            print ('Error: calc_cost_vs_rsrc must be called with at least one input file - either a .pcl file, or a .res file')
            return
    
        # If the caller provided a .pcl input file, read the data from it
        if (pcl_input_file_name==None):
            self.cost_vs_rsrc_data = []
        else:
            self.cost_vs_rsrc_data = pd.read_pickle(r'../res/{}' .format (pcl_input_file_name))
    
        # If the caller provided a .res input file, parse the data from it
        for file_name in res_input_file_names:
            self.parse_file(file_name, parse_cost=True, parse_cost_comps=False, parse_num_usrs=False)
    
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
                [y_lo, y_hi]          = self.conf_interval (ar=avg_cost_of_each_seed, avg=avg_cost_of_all_seeds) # low, high y values for this plotted conf' interval
                
                cost_vs_rsrc_data_of_this_mode.append ({'cpu' : cpu_val, 'y_lo' : y_lo, 'y_hi' : y_hi, 'y_avg' : avg_cost_of_all_seeds,'num_of_seeds' : len(avg_cost_of_each_seed)})
            
            # Add this new calculated point to the ds. Avoid duplications of points.
            for point in sorted (cost_vs_rsrc_data_of_this_mode, key = lambda point : point['cpu']):
                
                list_of_item = list (filter (lambda item : item['cpu']==cpu_val and item['mode']==mode, self.cost_vs_rsrc_data)) # all items with this mode, and cpu, already found in self.cost_vs_rsrc_data
                if (point not in self.cost_vs_rsrc_data and len(list_of_item)==0): # insert this new point to the list of points only if it's not already found in self.cost_vs_rsrc_data
                    self.cost_vs_rsrc_data.append (point)
                    point['mode'] = mode
        
        # store the data as binary data stream
        self.pcl_output_file_name = 'cost_vs_rsrc_{}.pcl' .format (self.input_file_name.split('.res')[0]) 
        with open('../res/' + self.pcl_output_file_name, 'wb') as cost_vs_rsrc_data_file:
            pickle.dump(self.cost_vs_rsrc_data, cost_vs_rsrc_data_file)
        return self.pcl_output_file_name 

    def calc_mig_cost_vs_rsrc (self, prob=0.3, res_input_file_names=None, pcl_input_file_name=None):
        """
        Calculate the data needed for plotting a graph showing the migration cost / number of migrated chains / number of reshuffles, as a function of the amount of resources (actually, cpu capacity at leaf):
        * Optional inputs: 
            .pcl file, containing self.list_of_dicts (usually as a result of a previous run of self.parse_file ()).
            A list of .res files, containing the results of a run.
            At least one file (either .pcl, or .res file) should be given       
        * Calculate the average # of migrated chains for each seed during the whole trace for each seed, the confidence intervals, etc. 
        * Save the (pickled) processed data into self.cost_vs_rsrc_data.pcl.
        * Returns the file name to which it saved the pickled results. 
        """
        
        self.mig_vs_rsrc_data = []
        
        # If the caller provided a .res input file, parse the data from it
        if (res_input_file_names != None):
            for file_name in res_input_file_names:
                self.parse_file(file_name, parse_cost=True, parse_cost_comps=True, parse_num_usrs=False)
            self.dump_self_list_of_dicts_to_pcl (res_file_name = res_input_file_names[0])  
        else:
            self.list_of_dicts = pd.read_pickle(r'../res/{}' .format (pcl_input_file_name))
    
        mig_vs_rsrc_data = []

        mode_list = sorted (self.list_of_dicts, key = lambda item : item['cpu']) # list of lines with data about this mode

        # Filter-out all results of failed runs 
        failed_runs = [] # failed_runs will include the cpu and seed values for all runs that fail: we've to filter-out these results while calculating the mean cost
        for item in [item for item in mode_list if item['stts']!=1 ]:
            failed_runs.append ({'cpu' : item['cpu'], 'seed' : item['seed']})
        for failed_run in failed_runs: # Remove all results of this failed_run from the list of relevant results 
            mode_list = list (filter (lambda item : not (item['cpu']==failed_run['cpu'] and item['seed']==failed_run['seed']), mode_list))

        cpu_vals = sorted (set ([item['cpu'] for item in mode_list]))

        for cpu_val in cpu_vals:  
            
            mode_cpu_list = list (filter (lambda item : item['cpu']==cpu_val, mode_list)) # list of results of runs for this mode, and cpu value
            avg_mig_cost_of_each_seed        = [] # will hold the avg mig cost of each successful run with a given mode, cpu value, and seed (averaging the cost over all the slots in the trace)
            for seed in set ([item['seed'] for item in mode_cpu_list]): # list of seeds for which the whole run succeeded with this mode (algorithm), and this cpu val
                avg_mig_cost_of_each_seed.       append (np.average ([item['mig_cost']      for item in mode_cpu_list if item['seed']==seed]))                    

            avg_mig_cost_of_all_seeds        = np.average (avg_mig_cost_of_each_seed)
            [y_lo, y_hi]          = self.conf_interval (ar=avg_mig_cost_of_each_seed, avg=avg_mig_cost_of_all_seeds) # low, high y values for this plotted conf' interval
            
            mig_vs_rsrc_data.append ({'cpu' : cpu_val, 'y_lo' : y_lo, 'y_hi' : y_hi, 'y_avg' : avg_mig_cost_of_all_seeds,'num_of_seeds' : len(avg_mig_cost_of_each_seed)})
        
        # Add this new calculated point to the ds. Avoid duplications of points.
        for point in sorted (mig_vs_rsrc_data, key = lambda point : point['cpu']):
            
            list_of_item = list (filter (lambda item : item['cpu']==cpu_val, self.mig_vs_rsrc_data)) # all items with this mode, and cpu, already found in self.cost_vs_rsrc_data
            if (point not in self.mig_vs_rsrc_data and len(list_of_item)==0): # insert this new point to the list of points only if it's not already found in self.cost_vs_rsrc_data
                self.mig_vs_rsrc_data.append (point)
        
        # store the data in a '.pcl' file (binary data stream)
        pcl_output_file_name = '{}_mig_cost_vs_rsrc.pcl' .format (city) 
        with open('../res/' + pcl_output_file_name, 'wb') as mig_vs_rsrc_data_file:
            pickle.dump(mig_vs_rsrc_data, mig_vs_rsrc_data_file)
        return pcl_output_file_name 
    
    def calc_crit_chains_vs_rsrc (self, res_input_file_names=None, min_t=30001, max_t=30600, prob=0.3, pcl_input_file_name=None):
        """
        Calculate the data needed for plotting a graph showing the migration cost / number of migrated chains / number of reshuffles, as a function of the amount of resources (actually, cpu capacity at leaf):
        * Optional inputs: 
            .pcl file, containing self.list_of_dicts (usually as a result of a previous run of self.parse_file ()).
            A list of .res files, containing the results of a run.
            At least one file (either .pcl, or .res file) should be given       
        * Calculate the average # of migrated chains for each seed during the whole trace for each seed, the confidence intervals, etc. 
        * Save the (pickled) processed data into self.cost_vs_rsrc_data.pcl.
        * Returns the file name to which it saved the pickled results. 
        """
        
        self.mig_vs_rsrc_data = []
        
        # If the caller provided a .res input file, parse the data from it
        for file_name in res_input_file_names:
            self.parse_file(file_name, parse_cost=True, parse_cost_comps=True, parse_num_usrs=False)
    
        mig_vs_rsrc_data = []

        mode_list = sorted (self.gen_filtered_list (self.list_of_dicts, prob=prob, min_t=min_t, max_t=max_t), key = lambda item : item['cpu']) # list of lines with data about this mode

        # Filter-out all results of failed runs 
        failed_runs = [] # failed_runs will include the cpu and seed values for all runs that fail: we've to filter-out these results while calculating the mean cost
        for item in [item for item in mode_list if item['stts']!=1 ]:
            failed_runs.append ({'cpu' : item['cpu'], 'seed' : item['seed']})
        for failed_run in failed_runs: # Remove all results of this failed_run from the list of relevant results 
            mode_list = list (filter (lambda item : not (item['cpu']==failed_run['cpu'] and item['seed']==failed_run['seed']), mode_list))

        cpu_vals = sorted (set ([item['cpu'] for item in mode_list]))

        for cpu_val in cpu_vals:  
            
            mode_cpu_list = list (filter (lambda item : item['cpu']==cpu_val, mode_list)) # list of results of runs for this mode, and cpu value
            avg_num_crit_chains_of_each_seed = [] # will hold the avg num of critical chains migrated of each successful run with a given mode, cpu value, and seed (averaging the cost over all the slots in the trace)
            for seed in set ([item['seed'] for item in mode_cpu_list]): # list of seeds for which the whole run succeeded with this mode (algorithm), and this cpu val
                avg_num_crit_chains_of_each_seed.append (np.average ([item['num_crit_usrs'] for item in mode_cpu_list if item['seed']==seed]))                    

            avg_num_crit_chains_of_all_seeds = np.average (avg_num_crit_chains_of_each_seed)
            [y_lo, y_hi]          = self.conf_interval (ar=avg_num_crit_chains_of_each_seed, avg=avg_num_crit_chains_of_all_seeds) # low, high y values for this plotted conf' interval
            
            mig_vs_rsrc_data.append ({'cpu' : cpu_val, 'y_lo' : y_lo, 'y_hi' : y_hi, 'y_avg' : avg_num_crit_chains_of_all_seeds,'num_of_seeds' : len(avg_num_crit_chains_of_each_seed)})
        
        # Add this new calculated point to the ds. Avoid duplications of points.
        for point in sorted (mig_vs_rsrc_data, key = lambda point : point['cpu']):
            
            list_of_item = list (filter (lambda item : item['cpu']==cpu_val, self.mig_vs_rsrc_data)) # all items with this mode, and cpu, already found in self.cost_vs_rsrc_data
            if (point not in self.mig_vs_rsrc_data and len(list_of_item)==0): # insert this new point to the list of points only if it's not already found in self.cost_vs_rsrc_data
                self.mig_vs_rsrc_data.append (point)
        
        # store the data in a '.pcl' file (binary data stream)
        self.pcl_output_file_name = '{}_num_crit_chains_vs_rsrc.pcl' .format (self.input_file_name.split('.res')[0]) 
        with open('../res/' + self.pcl_output_file_name, 'wb') as mig_vs_rsrc_data_file:
            pickle.dump(mig_vs_rsrc_data, mig_vs_rsrc_data_file)
        return self.pcl_output_file_name 
    
    def plot_mig_vs_rsrc (self, pcl_input_file_name):
        """
        Generate a plot of the number of migrations vs. the cpu capacity at the leaf, for the given city.
        The plot is saved in a .pdf file.
        The plot is based on the data found in the given input .pcl file.
        """

        city             = parse_city_from_input_file_name(pcl_input_file_name)
        mig_vs_rsrc_data = pd.read_pickle(r'../res/{}' .format (pcl_input_file_name))
        matplotlib.rcParams.update({'font.size': FONT_SIZE, 
                                    'legend.fontsize': LEGEND_FONT_SIZE,
                                    'xtick.labelsize':FONT_SIZE,
                                    'ytick.labelsize':FONT_SIZE,
                                    'axes.labelsize': FONT_SIZE,
                                    'axes.titlesize':FONT_SIZE,
                                    })
        _, ax = plt.subplots()
        ax.set_xlabel ('CPU at leaf [GHz]')
        ax.set_ylabel ('# of Mig. Chains')
        ax.plot ([item['cpu']/10 for item in mig_vs_rsrc_data], [item['y_avg']/UNIFORM_CHAIN_MIG_COST for item in mig_vs_rsrc_data], markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        for item in mig_vs_rsrc_data:
            ax.plot ( (item['cpu']/10, item['cpu']/10), (item['y_lo']/UNIFORM_CHAIN_MIG_COST, item['y_hi']/UNIFORM_CHAIN_MIG_COST), color=self.color_dict['ourAlg']) # Plot the confidence interval
        ax.plot ( (item['cpu']/10, item['cpu']/10), (item['y_lo']/UNIFORM_CHAIN_MIG_COST, item['y_hi']/UNIFORM_CHAIN_MIG_COST), color=self.color_dict['ourAlg']) # Plot the confidence interval
        vertical_line_x = 23.5 if (city=='lux') else 130
        plt.axvline (vertical_line_x, color='black', linestyle='dashed')
        # ax.plot ( (vertical_line_x, vertical_line_x), (0, 1000), color='black')
        printFigToPdf ('{}_num_migs_vs_rsrc' .format (city))

    def parse_files (self, filenames):
        """
        Parse each of the given files. Add to its entry in self.list_of_dicts a field, presenting the length of the time slot.
        Save the results in a .pcl file.
        """
    
        full_list_of_dicts = [] # will hold all the dictionary items of the parsed data
        for filename in filenames:
            
            self.list_of_dicts = []
            T = self.parse_T_from_input_file_name (filename)
            self.parse_file (input_file_name = filename, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True)
            
            for item in self.list_of_dicts:
                item['T'] = T
                full_list_of_dicts.append (item)
                
        pcl_output_file_name = '{}_vary_T.pcl' .format (self.city)
        with open ('../res/' + pcl_output_file_name, 'wb') as pcl_output_file:
            pickle.dump (full_list_of_dicts, pcl_output_file)
                
        return pcl_output_file_name
    
    def plot_crit_n_mig_vs_T (self, pcl_input_file_name):
        """
        plot the number of critical chains (on one y-axis), and the migration cost (on the other y-axis), vs. the slot interval, T.
        Input: pcl_input_file_name - a .pcl file, containing a list_of_dicts, namely, a list of dictionaries with all the results. 
        """

        self.list_of_dicts = pd.read_pickle ('../res/{}' .format (pcl_input_file_name))
        
        list_of_Ts = set ([item['T'] for item in self.list_of_dicts]) # list_of_Ts is the list of all slots for which there're results 
        
        matplotlib.rcParams.update({'font.size'       : FONT_SIZE, 
                                    'legend.fontsize' : LEGEND_FONT_SIZE,
                                    'xtick.labelsize' : FONT_SIZE,
                                    'ytick.labelsize' : FONT_SIZE,
                                    'axes.labelsize'  : FONT_SIZE,
                                    'axes.titlesize'  : FONT_SIZE})
        
        _, num_crit_axis = plt.subplots()
        mig_cost_axis = num_crit_axis.twinx ()
        
        x, y_num_crit, y_mig_cost = [], [], []
        for T in list_of_Ts:
            list_of_dicts_T = [item for item in self.list_of_dicts if item['T']==T] # list_of_dicts_T <-- list of results when simulated with time slot==T.
            
            x.append (T) 
            y_num_crit.append (np.average ([item['num_crit_usrs'] for item in list_of_dicts_T]))
            y_mig_cost .append (sum ([item['mig_cost'] for item in list_of_dicts_T]))
                
        mig_color = 'blue'
        num_crit_axis.set_xlabel  ('Time Slot [s]',             fontsize=FONT_SIZE)
        num_crit_axis.set_xscale  ('log')
        num_crit_axis.set_xticks  ( [1, 2, 4, 8, 16])
        num_crit_axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        num_crit_axis.set_ylabel  ('Avg. # of Critical chains', fontsize=FONT_SIZE)
        mig_cost_axis.set_ylabel  ('Total Mig. Cost',           fontsize=FONT_SIZE, color=mig_color)
        mig_cost_axis.tick_params (axis='y', colors=mig_color)
        plt.xlim (1,16)
        self.my_plot (x=x, y=y_num_crit, ax=num_crit_axis,  color='black')
        self.my_plot (x=x, y=y_mig_cost,  ax=mig_cost_axis, color=mig_color)

        plt.show ()        

    def plot_Q (self, pcl_input_file_name):
        """
        plot several plots, showing the weighted cost, which considers the "phi" cost objective function, plus a penalty for the # of critical chains - as a function of the slot T. 
        """

        self.list_of_dicts = pd.read_pickle ('../res/{}' .format (pcl_input_file_name))
        
        list_of_Ts = sorted (set ([item['T'] for item in self.list_of_dicts])) # list_of_Ts is the list of all slots for which there're results 
        
        matplotlib.rcParams.update({'font.size'       : FONT_SIZE, 
                                    'legend.fontsize' : LEGEND_FONT_SIZE,
                                    'xtick.labelsize' : FONT_SIZE,
                                    'ytick.labelsize' : FONT_SIZE,
                                    'axes.labelsize'  : FONT_SIZE,
                                    'axes.titlesize'  : FONT_SIZE})
        
        _, y_axis = plt.plot ()
        
        list_of_costs_n_num_crits = [] 
        
        # First, gather all the data for calculating the plots' values
        for T in list_of_Ts:
            list_of_dicts_T = [item for item in self.list_of_dicts if item['T']==T] # list_of_dicts_T <-- list of results when simulated with time slot==T.
            
            list_of_costs_n_num_crits.append ({'T'                  : T,
                                               'cost_wo_penalty'    : np.average ([item['cost'] for item in list_of_dicts_T]), # value of the objective func'
                                               'num_of_crit_chains' : np.average ([item['num_crit_usrs']  for item in list_of_dicts_T])}) 
                                                                                    
        for Q in [100]:
            y_vals = []
            for T in list_of_Ts:
                list_of_dict = [item for item in list_of_costs_n_num_crits if item['T']==T]
                item = list_of_dict[0]
                y_vals.append (item['cost_wo_penalty'] + Q*item['num_of_cirt_chains'])
            plt.plot (list_of_Ts, y_vals, color='black')
                
        plt.xlabel  ('Time Slot [s]', fontsize=FONT_SIZE)
        plt.ylabel  ('Cost with Penalty', fontsize=FONT_SIZE)
        # plt.xlim (1,16)
        # y_axis.set_xscale  ('log')
        # y_axis.set_xticks  ( [1, 2, 4, 8, 16])
        # y_axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        plt.show ()        


def plot_num_crit_n_mig_vs_num_vehs (city, reshuffle):
    """
    Generate a plot of the number of critical chains, and number of migrated chains vs. the number of vehs. 
    Inputs: 
    city - city for which the plot is generated.
    reshuffle - if True, consider only the version of ourAlg that allows reshuffling, and name the output file with suffix 'resh.pdf'. 
                Else, consider only the MOC (Migrated Only Critical chains) version of ourAlg, and name the output file with suffix 'MOC.pdf'.   
    The plot is saved in a .pdf file.
    """
    
    my_res_file_parser = Res_file_parser ()
    if city=='Monaco':
        res_file_to_parse   = 'Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg.res'
        pcl_input_file_name = 'Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg.pcl'
    elif city=='short': 
        res_file_to_parse = 'Monaco_0730_0830_1secs_short.res'
        city = 'Monsco'
    else:                    
        res_file_to_parse   = 'Lux_0730_0830_1secs_post_p0.3_ourAlg.res' # 1347
        pcl_input_file_name = 'Lux_0730_0830_1secs_post_p0.3_ourAlg.pcl'
    if (city in ['Monaco', 'short']):
        cpu = 842 if reshuffle else 1347
    else:
        cpu = 94 if reshuffle else 250
    my_res_file_parser.plot_num_crit_n_mig_vs_num_vehs (cpu=cpu, reshuffle=reshuffle, pcl_input_file_name=pcl_input_file_name)
    
def plot_mig_vs_rsrc (city): 
    """
    Generate a plot of the number of migrations vs. the cpu capacity at the leaf, for the given city.
    The plot is saved in a .pdf file.
    """
        
    my_res_file_parser = Res_file_parser ()
    # pcl_output_file_name = my_res_file_parser.calc_mig_cost_vs_rsrc(res_input_file_names=['Lux_0730_0830_1secs_post_p0.3_ourAlg.res'] if city=='Lux' else ['Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg.res']) 
    pcl_output_file_name = my_res_file_parser.calc_mig_cost_vs_rsrc (pcl_input_file_name='Lux_0730_0830_1secs_post_p0.3_ourAlg.pcl' if city=='Lux' else 'Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg.pcl')
    my_res_file_parser.plot_mig_vs_rsrc (pcl_input_file_name = pcl_output_file_name) #'Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg_num_mig_chains_vs_rsrc.pcl')
         
if __name__ == '__main__':

    my_res_file_parser = Res_file_parser ()
    my_res_file_parser.plot_Q (pcl_input_file_name='Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg.pcl')
    exit ()
    
    city='Monaco'
    reshuffle=True
    # plot_num_crit_n_mig_vs_num_vehs (city=city, reshuffle=reshuffle)
    plot_mig_vs_rsrc (city=city)
    exit ()
    
    # my_res_file_parser = Res_file_parser ()
    # pcl_output_file_name = my_res_file_parser.parse_files (['Lux_0730_0830_1secs_post_p0.3_ourAlg_cpu103.res', 'Lux_0730_0830_16secs_post_p0.3_ourAlg.res'])
    # my_res_file_parser.plot_crit_n_mig_vs_T (pcl_input_file_name=pcl_output_file_name)
    
    # cost_vs_rsrc_data = pd.read_pickle (r'../res/cost_vs_rsrc_Monaco_0820_0830_1secs_Telecom_p0.3.pcl')
    # cost_vs_rsrc_data = list (filter (lambda item : item['mode']!='cpvnf', cost_vs_rsrc_data))
    # with open('../res/cost_vs_rsrc_Monaco_0820_0830_1secs_Telecom_p0.3.pcl', 'wb') as cost_vs_rsrc_data_file:
    #     pickle.dump (cost_vs_rsrc_data, cost_vs_rsrc_data_file)

    # city = 'Monaco'
    # if (city=='Lux'):            
    #     my_res_file_parser.plot_RT_prob_sim_python ('RT_prob_sim_Lux.post.antloc_256cells.poa2cell_Lux_0820_0830_1secs_post.poa.res')
    # else:
    #     my_res_file_parser.plot_RT_prob_sim_python ('RT_prob_sim_Monaco.Telecom.antloc_192cells.poa2cell_Monaco_0820_0830_1secs_Telecom.poa.res')
    
    # pcl_output_file_name = my_res_file_parser.calc_cost_vs_rsrc (res_input_file_names=['Lux_0820_0830_1secs_post_p0.3_opt.res', 'Lux_0820_0830_1secs_post_p0.3_cpvnf.res', 'Lux_0820_0830_1secs_post_p0.3_ffit.res', 'Lux_0820_0830_1secs_post_p0.3_ourAlg_short.res'])
    # pcl_output_file_name = my_res_file_parser.calc_cost_vs_rsrc (res_input_file_names=['Monaco_0820_0830_1secs_Telecom_p0.3_ourAlg.res'])
    # pcl_output_file_name = my_res_file_parser.calc_cost_vs_rsrc (res_input_file_names=['Monaco_0820_0830_1secs_Telecom_p0.3_opt.res', 'Monaco_0820_0830_1secs_Telecom_p0.3_cpvnf.res', 'Monaco_0820_0830_1secs_Telecom_p0.3_ffit.res', 'Monaco_0820_0830_1secs_Telecom_p0.3_ourAlg.res'])
    # my_res_file_parser.plot_cost_vs_rsrc (pcl_input_file_name=pcl_output_file_name)
    
    # pcl_file_name = my_res_file_parser.calc_mig_vs_rsrc(pcl_input_file_name=None, res_input_file_names=['Lux_0820_0830_1secs_post_p0.3_ourAlg_short.res'])   
    
    # my_res_file_parser.plot_cost_vs_rsrc (normalize_X=True, slot_len_in_sec=float(input_file_name.split('sec')[0].split('_')[-1]), X_norm_factor=X_norm_factor)

    
    # my_res_file_parser.plot_tot_num_of_vehs_per_slot (['Monaco_0730_0830_1secs_cnt.pcl', 'Lux_0730_0830_1secs_cnt.pcl'])
