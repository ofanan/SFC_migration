import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.pylab as pylab
import numpy as np, scipy.stats as st, pandas as pd
from pandas._libs.tslibs import period
from printf import printf, printFigToPdf 
import pickle
# from pandas.tests.extension.test_external_block import df

# This class allows choosing the order of magnitude in plots that use scientific notation 
# class OOMFormatter(matplotlib.ticker.ScalarFormatter):
#     def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
#         self.oom = order
#         self.fformat = fformat
#         matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
#     def _set_order_of_magnitude(self):
#         self.orderOfMagnitude = self.oom
#     def _set_format(self, vmin=None, vmax=None):
#         self.format = self.fformat
#         if self._useMathText:
#             self.format = r'$\mathdefault{}$' .format (s elf.format) 

TREE_HEIGHT    = 6
NUM_DIRECTIONS = (TREE_HEIGHT-1)*2
OVERALL_DIR    = -1


# Indices of fields indicating the settings in a standard ".res" file
t_idx     = 0 # time slot
mode_idx  = 1 # the alg' used, e.g. 'opt',  'ourAlg', ...
cpu_idx   = 2 # ammount of cpu in the leaf
prob_idx  = 3 # prob' that a new request is for a RT service
seed_idx  = 4 # seed used for randomization
stts_idx  = 5 # stts of the run: 1 is sccs. Other values indicate various fails
ad_idx    = 6 # accumulation delay at the leaf, in the dist' async' alg. 
pdd_idx   = 7 # push-down delay at the leaf, in the dist' async' alg.
num_of_fields = stts_idx+1 # num' of fields in a standard .res file
num_of_fields_w_delays = pdd_idx+1 # num' of fields where delays (accumulation delay and push-down delay) are also reported

num_usrs_idx      = 6
num_crit_usrs_idx = 7
reshuffle_idx     = num_crit_usrs_idx+1 

MARKER_SIZE             = 16
MARKER_SIZE_SMALL       = 1
LINE_WIDTH              = 3 
LINE_WIDTH_SMALL        = 1 
FONT_SIZE               = 20
FONT_SIZE_SMALL         = 5
LEGEND_FONT_SIZE        = 12
LEGEND_FONT_SIZE_SMALL  = 5 

UNIFORM_CHAIN_MIG_COST = 600

avg_new_vehs_per_slot = {'Monaco' : [0,5,10,15,20,24,28,33,37,41,46],
                         'Lux'    : [0,8,16,24,32,40,48,55,63,70,78]}

# Parse the len of the time slot simulated, from the given string
find_time_slot_len = lambda string : int(string.split('secs')[0].split('_')[-1])
 
# limit the y axis to values between 0 and 1.05 the max y value
my_y_lim = lambda y : plt.ylim (0, 1.17 * max(y)) 

ISP = {'Lux' : 'post', 'Monaco' : 'Telecom'}

def parse_city_from_input_file_name (input_file_name): 
    """
    Find which city's data are these, based on the input file name
    """
    city = input_file_name.split ('_')[0]
    
    if (city=='RT'): # in the special case of RT_prob_sim, this is not the city's file name, so need further parsing
        city = input_file_name.split ('RT_prob_sim_')[1].split('_')[0].split('.')[0]
    return city

class Res_file_parser (object):
    """
    Parse "res" (result) files, and generate plots from them.
    """

    # Calculate the confidence interval of an array of values ar, given its avg. Based on 
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    conf_interval = lambda self, ar, avg, conf_lvl=0.99 : st.t.interval (conf_lvl, len(ar)-1, loc=avg, scale=st.sem(ar)) if np.std(ar)>0 else [avg, avg]
   

    # Find the length of the time slot based on the input file name
    parse_T_from_input_file_name = lambda self, input_file_name : int (input_file_name.split ('secs')[0].split('_')[-1])

    # Set the parameters of the plot (sizes of fonts, legend, ticks etc.).
    #mfc='none' makes the markers empty.
    set_plt_params = lambda self, size='large' : matplotlib.rcParams.update({'font.size': FONT_SIZE, 
                                                                             'legend.fontsize': LEGEND_FONT_SIZE,
                                                                             'xtick.labelsize':FONT_SIZE,
                                                                             'ytick.labelsize':FONT_SIZE,
                                                                             'axes.labelsize': FONT_SIZE,
                                                                             'axes.titlesize':FONT_SIZE,}) if (size=='large') else matplotlib.rcParams.update({
                                                                             'font.size': FONT_SIZE_SMALL, 
                                                                             'legend.fontsize': LEGEND_FONT_SIZE_SMALL,
                                                                             'xtick.labelsize':FONT_SIZE_SMALL,
                                                                             'ytick.labelsize':FONT_SIZE_SMALL,
                                                                             'axes.labelsize': FONT_SIZE_SMALL,
                                                                             'axes.titlesize':FONT_SIZE_SMALL,
                                                                             })
    
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

        # # List of algorithms' names, used in the plots' legend, for the centralized case
        # self.legend_entry_dict = {'opt'     :  'LBound', 
                                  # 'ourAlg'  : 'BUPUfullOld', 
                                  # 'SyncPartResh' : 'BUPU',
                                  # 'ffit'    : 'F-Fit', 
                                  # 'cpvnf'   : 'CPVNF', 
                                  # 'ms'      : 'MultiScaler',
                                  # 'ourAlgC' : 'BUPUmoc', 
                                  # 'ffitC'   : 'F-Fitmoc', 
                                  # 'cpvnfC'  : 'CPVNFmoc'} 

        # List of algorithms' names, used in the plots' legend, for the dist' case
        self.legend_entry_dict = {'opt'          : 'LBound',
                                  'optInt'       : 'Opt',
                                  'SyncPartResh' : 'BUPU',
                                  'AsyncBlk'     : 'Distributed Blk', 
                                  'AsyncNBlk'    : 'Distributed NBlk', 
                                  'Async'        : 'Old Async; plz check', 
                                  'ffit'         : 'F-Fit',
                                  'cpvnf'        : 'CPVNF',
                                  'ms'           : 'MultiScaler'}

        # # The colors used for each alg's plot, in the centralized case
        # self.color_dict       = {'opt'    : 'green',
                                # 'ourAlg'  : 'yellow',
                                # 'SyncPartResh' : 'purple',
                                # 'ffit'    : 'blue',
                                # 'cpvnf'   : 'black',
                                # 'ourAlgC' : 'purple',
                                # 'ffitC'   : 'blue',
                                # 'cpvnfC'  : 'black',
                                # 'ms'      : 'yellow'}
        
        # The colors used for each alg's plot, in the dist' case
        self.color_dict       = {'opt'          : 'green',
                                 'optInt'       : 'green',
                                'SyncPartResh'  : 'purple',
                                'Async'         : 'brown',
                                'AsyncBlk'      : 'brown',
                                'AsyncNBlk'     : 'brown',
                                'ffit'          : 'blue',
                                'ms'            : 'yellow',
                                'cpvnf'         : 'black'}

        # # The markers used for each alg', in the centralized case
        # self.markers_dict     = {'opt'    : 'x',
                                # 'ourAlg'  : 'v',
                                # 'SyncPartResh' : 'o',
                                # 'ffit'    : '^',
                                # 'cpvnf'   : 's',
                                # 'ourAlgC' : 'h',
                                # 'ffitC'   : 'v',
                                # 'cpvnfC'  : 'd',
                                # 'ms'      : 'v'}
        
        # The markers used for each alg', in the dist' case
        self.markers_dict     = {'opt'          : 'x',
                                 'optInt'       : 'x',
                                'SyncPartResh'  : 'o',
                                'Async'         : 'v',
                                'AsyncBlk'      : 'v',
                                'AsyncNBlk'     : 'v',
                                'ffit'          : '^',
                                'cpvnf'         : 's',
                                'ms'            : 'v'}
        self.list_of_dicts   = [] # a list of dictionaries, holding the settings and the results read from result files
      
    def my_plot (self, ax, x, y, mode='ourAlg', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=None, label=None, marker=None): 
        
        """
        Plot a single x, y, python line, with the required settings (colors, markers etc).
        """ 
    
        color = self.color_dict[mode] if (color==None) else color  
        label=self.legend_entry_dict[mode] if (label==None) else label
    
        if (mode in ['ourAlgC', 'cpvnfC', 'ffitC']):
            ax.plot (x, y, color=color, marker=self.markers_dict[mode], markersize=markersize, linewidth=linewidth, label=label, mfc='none', linestyle='dashed') 
        else:
            ax.plot (x, y, color=color, marker=self.markers_dict[mode] if marker==None else marker, markersize=markersize, linewidth=linewidth, label=label, mfc='none')


    
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

    def dump_self_list_of_dicts_to_pcl (self, pcl_input_file_name=None, res_file_names=None):
        """
        1. Read the given .pcl file (if given)
        2. Parse all the given res_file_names.
        3. Dump all the data within self.list_of_dicts into the a .pcl name.
        The name of the output .pcl file is:
        - If only a res_files are given in the input: the same name as the given res_file_name[0], but with '.pcl' instead of 'res' as an extension.
        - Else: same name as the .pcl input file name.
        """
        
        if (pcl_input_file_name==None and res_file_names==None):
            print ('error: please specify either a pcl_input_file_name and/or res_file_names.')
            exit ()
    
        if (pcl_input_file_name!=None and res_file_names!=None):
            self.city = parse_city_from_input_file_name (pcl_input_file_name)
            res_file_city = parse_city_from_input_file_name (res_file_names[0])
            if (res_file_city != self.city):
                print ('err: res_file_names[0]={} while input_pcl_file_name={}. However, they should belong to the same city' .format (pcl_input_file_name, res_file_names[0]))
                exit ()
        
        if (pcl_input_file_name != None):
            self.list_of_dicts = pd.read_pickle(r'../res/pcl_files/{}' .format (pcl_input_file_name))
            self.city = parse_city_from_input_file_name (pcl_input_file_name)
        
        if (res_file_names!=None):
            for res_file_name in res_file_names:
                self.parse_res_file (input_file_name=res_file_name)
            if (pcl_input_file_name==None):
                pcl_full_path_file_name = '../res/pcl_files/{}.pcl' .format (res_file_names[0].split('.res')[0])
            else:
                pcl_full_path_file_name = '../res/pcl_files/{}' .format (pcl_input_file_name)
                
        with open(pcl_full_path_file_name, 'wb') as pcl_file:
            pickle.dump(self.list_of_dicts, pcl_file)
        return pcl_input_file_name
      
    def plot_num_crit_n_mig_vs_num_vehs (self, cpu, reshuffle, res_file_to_parse=None, pcl_input_file_name=None):
        """
        Generate a plot of the ratio of critical usrs over time, and of the mig cost over time.
        The output plot may be either tikz, and/or python.
        Inputs: 
        cpu - the plot will consider only result obtained using this cpu value at leaf servers.
        reshuffle - a binary variable. If true, the suffix of the generated output file is 'resh.pdf'. Else, the suffix is 'MOC.pdf' (MOC=Migrate Only Critical chains).
        res_file_to_parse - when given, the function parses this '.res' file, and then used the data for the plot.
        pcl_input_file_name - if no res_file_to_parse is given, the function reads the input from this file name.   
        """
        
        if (res_file_to_parse != None):
            self.parse_res_file(res_file_to_parse, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True)
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

        self.set_plt_params (size='small')
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


    def parse_res_file (self, input_file_name, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True, parse_crit_len=False, time_slot_len=None, ignore_worse_lines=False):
        """
        Parse a result file, in which each un-ommented line indicates a concrete simulation settings.
        Inputs:
        parse_cost - when true, parse and save a field also for the total cost.
        parse_cost_comp - when true, parse and save a field also for the cost's component (cpu, link, mig).
        parse_num_usrs - when true, parse and save a field also for the the overall number of the usrs, and the num of crit' usrs.
        parse_crit_len - when true, increase the relevant counters counting the number of chains with the length of duration of being critical.
        ignore_worse_lines - when true, ignore a line for which prob=p, seed=sd, and cpu=c if there already exists an entry with prob=p, seed=sd and cpu<c 
        """
        
        self.city = parse_city_from_input_file_name(input_file_name)
        print ('city is ', self.city)
        self.input_file_name = input_file_name
        if (time_slot_len==None):
            self.time_slot_len   = find_time_slot_len (self.input_file_name)
        else:
            self.time_slot_len = time_slot_len 
        self.input_file      = open ("../res/" + input_file_name,  "r")
        lines                = (line.rstrip() for line in self.input_file) # "lines" contains all lines in input file
        lines                = (line for line in lines if line)       # Discard blank lines
        
        for line in lines:
        
            # Discard lines with comments / verbose data
            if (line.split ("//")[0] == ""):
                continue
           
            self.parse_res_line(line, parse_cost=parse_cost, parse_cost_comps=parse_cost_comps, parse_num_usrs=parse_num_usrs, parse_crit_len=parse_crit_len)
            if (self.dict==None): # No new data from this line
                continue
            if (ignore_worse_lines):
                list_of_dict = [item for item in self.list_of_dicts if 
                                (item['mode']==self.dict['mode'] and item['prob']==self.dict['prob'] and item['t']==self.dict['t'] and item['seed']==self.dict['seed'])]
                if (len(list_of_dict)>0): # there's already an item with the same mode, prob, t and seed in list_of_dicts
                    list_of_dict[0]['cpu'] = min (list_of_dict[0]['cpu'], self.dict['cpu'])
                    # if (self.dict['prob']==0.3 and self.dict['seed']==14):
                    #     print ('dummy')
                    continue
            if (not(self.dict in self.list_of_dicts)):
                self.list_of_dicts.append(self.dict)                

        self.input_file.close


    def parse_res_line (self, line, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True, parse_crit_len=False):
        """
        Parse a line in a ".res" result file, and save the parsed data in the dictionary self.dict. 
        Such a line should begin with a string having several fields, detailing the settings.
        Inputs:
        parse_cost - when true, parse and save a field also for the total cost.
        parse_cost_comp - when true, parse and save a field also for the cost's component (cpu, link, mig).
        parse_num_usrs - when true, parse and save a field also for the the overall number of the usrs, and the num of crit' usrs.
        parse_crit_len - when true, increase the relevant counters counting the number of chains with the length of duration of being critical.
        """

        splitted_line = line.split (" | ")
         
        settings          = splitted_line[0]
        splitted_settings = settings.split ("_")

        if (len (splitted_settings) not in [num_of_fields, num_of_fields_w_delays]):
            print ("encountered a format error. Splitted line={}\nsplitted settings={}" .format (splitted_line, splitted_settings))
            self.dict = None
            return

        stts = int (splitted_settings [stts_idx].split("stts")[1])
        self.dict = {
            "t"         : int   (splitted_settings [t_idx]   .split('t')[1]),
            "mode"      : splitted_settings      [mode_idx],
            "cpu"       : int   (splitted_settings [cpu_idx] .split("cpu")[1]),  
            "prob"      : float (splitted_settings [prob_idx].split("p")  [1]),  
            "seed"      : int   (splitted_settings [seed_idx].split("sd") [1]),  
            "stts"      : stts,
        }
        if (len (splitted_settings)==num_of_fields_w_delays):
            self.dict['ad'] = float (splitted_settings [prob_idx].split("ad") [1])
            self.dict['ad'] = float (splitted_settings [prob_idx].split("pdd")[1])
        
        if (stts!=1): # if the run failed, the other fields are irrelevant
            return

        if (parse_cost):
            self.dict["cost"] = float (splitted_line[4].split("=")[1])

        if (parse_cost_comps):        
            self.dict["cpu_cost"]  = float (splitted_line[1].split("=")[1])
            self.dict["link_cost"] = float (splitted_line[2].split("=")[1])
            self.dict["mig_cost" ] = float (splitted_line[3].split("=")[1])            

        if (parse_num_usrs and len(splitted_line) > num_usrs_idx): # Is it a longer line, indicating also the # of usrs, and num of critical usrs?
            self.dict['num_usrs']      = int (splitted_line[num_usrs_idx]     .split("=")[1])
            self.dict['num_crit_usrs'] = int (splitted_line[num_crit_usrs_idx].split("=")[1])
            if (len(splitted_line) > reshuffle_idx):
                self.dict['resh'] = True if (splitted_line[num_crit_usrs_idx].split("=")[1]=='T') else False
                
        if (parse_crit_len):
            for crit_len in range (1, len(splitted_line)):
                self.crit_len_cnt[crit_len] += int (splitted_line[crit_len].split('=')[1])
            
    def gen_filtered_list (self, list_to_filter, min_t=-1, max_t=float('inf'), prob=None, mode=None, cpu=None, stts=-1, seed=None):
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
        if (seed != None):
            list_to_filter = list (filter (lambda item : item['seed'] == seed, list_to_filter))
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
     
     
    def plot_RT_prob_sim_python (self, pcl_input_file_name=None, res_input_file_name=None, reshuffle=True, dist=False):
        """
        Generating a python plot showing the amount of resource augmentation required, as a function of the probability that a user has tight (RT) delay requirements.
        The plot also presents the conf' intervals.
        The raw data is also printed to an output text file, with the extension .dat
        When given an input file name, parse it first.
        If no input_file_name is given, the function assumes that previously to calling it, an input file was parsed.
        when 'dist' is True, use the settings (colors, legends, output file name etc.) of "distributed" SFC_mig' project.
        """
        
        if (pcl_input_file_name != None):
            self.list_of_dicts = pd.read_pickle(r'../res/pcl_files/{}' .format (pcl_input_file_name))
            self.city = parse_city_from_input_file_name (pcl_input_file_name)
            input_file_name = pcl_input_file_name.split('.pcl')[0]            
        if (res_input_file_name != None):
            self.parse_res_file(res_input_file_name, parse_cost=False, parse_cost_comps=False, parse_num_usrs=False)
            input_file_name = res_input_file_name.split('.res')[0]            
        input_file_name = input_file_name if (input_file_name != None) else self.input_file_name 
        dat_output_file = open ('../res/{}.dat' .format (input_file_name), 'w')

        self.set_plt_params ()
        _, ax = plt.subplots()
        # modes = ['opt', 'ms', 'ffit', 'cpvnf', 'SyncPartResh', 'Async'] if reshuffle else ['opt', 'ourAlgC', 'ffitC', 'cpvnfC'] 
        modes = ['opt', 'SyncPartResh', 'AsyncNBlk', 'ms', 'cpvnf', 'ffit'] if reshuffle else ['opt', 'ourAlgC', 'ffitC', 'cpvnfC'] 
        for mode in modes: 
            
            list_of_points = self.gen_filtered_list(self.list_of_dicts, mode=mode, stts=1) 
        
            x = set () # The x value will hold all the probabilities that appear in the .res file
            for point in list_of_points: # A cpu cap' unit represents 100 MHz --> to represent results by units of GHz, divide the cpu cap' by 10.
                point['cpu'] /= 10
                x.add (point['prob'])
            
            x = sorted (x)
            y = []
            
            for x_val in x: # for each concrete value in the x vector
                
                samples = [item['cpu'] for item in self.gen_filtered_list(list_of_points, prob=x_val)]
                # if (mode not in ['opt', 'ourAlg', 'ourAlgC'] and len(samples)<20):
                    # print ('Note: mode={}, x={}, num of seeds is only {}. seeds are: {}' .format (mode, x_val, len(samples), [item['seed'] for item in self.gen_filtered_list(list_of_points, prob=x_val)]))
                avg = np.average(samples)
                
                [y_lo, y_hi] = self.conf_interval (samples, avg)
                
                # if (x_val==0.3 and mode in ['ffit', 'cpvnf', 'ms', 'ourAlgC']):
                #     print ('mode={}, x_val=0.3, y_hi={:.1f}' .format (mode, y_hi))

                printf (dat_output_file, 'mode={}. x={}, y_lo={:.1f}, y_hi={:.1f}\n' .format (mode, x_val, y_lo, y_hi))                    

                ax.plot ((x_val,x_val), (y_lo, y_hi), color=self.color_dict[mode]) # Plot the confidence interval
                
                y.append (avg)
            
            self.my_plot (ax, x, y, mode)
        plt.xlabel('Fraction of RT Chains')
        plt.ylabel('Min CPU at Leaf [GHz]')
        ax.legend (ncol=2, fontsize=LEGEND_FONT_SIZE) #(loc='upper center', shadow=True, fontsize='x-large')
        plt.xlim (0, 1) #(-0.04,1.04)
        print ('self.city={}' .format (self.city))
        if (dist):
            plt.ylim (0, 33 if self.city=='Lux' else 230)
            plt.savefig ('../res/dist_{}.pdf' .format (input_file_name), bbox_inches='tight')        
        else:
            plt.ylim (0, 33 if self.city=='Lux' else 230)
            plt.savefig ('../res/{}.pdf' .format (input_file_name), bbox_inches='tight')

    def gen_cost_vs_rsrc_tbl (self, city, normalize_X = True, slot_len_in_sec=1, normalize_Y=True, dist=False, pcl_input_file_name=None):
        """
        Plot the cost as a function of the amount of resources (actually, cpu capacity at leaf).
        Possibly normalize the amounts of cpu (the X axis) by either the min' amount of cpu required by opt (LBound) to obtain a feasible sol; 
        and/or normalize the cost (the Y axis) by the costs obtained by opt.   
        """
        modes = ['opt', 'SyncPartResh', 'AsyncBlk', 'AsyncNBlk'] if dist else ['opt', 'optG', 'optInt', 'SyncPartResh','ms', 'ffit', 'cpvnf']
        if (pcl_input_file_name==None):
            pcl_input_file_name='{}_{}cost_vs_rsrc_0820_0830_1secs_p0.3.pcl' .format (city, 'dist_' if dist else '')


        self.cost_vs_rsrc_data = pd.read_pickle(r'../res/pcl_files/{}' .format (pcl_input_file_name))
        self.output_file_name = '../res/{}_{}cost_vs_rsrc.dat' .format (city, 'dist_' if dist else '')
        self.output_file      = open (self.output_file_name, "w")
        
        if (normalize_X):
            opt_list = [item for item in self.cost_vs_rsrc_data if item['mode']=='opt']  

            cpu_vals = sorted (list (set([item['cpu'] for item in opt_list])))
            if (len (cpu_vals)==0):
                print ('Error: you asked to normalize by opt, but no results of opt exist. Please add first results of optInt to the parsed input file.')
                return
            X_norm_factor = cpu_vals[0] # normalize X axis by the minimum cpu
        
        else:
            X_norm_factor = 1

        list_of_avg_vals = []        

        title = 'cpu   &opt & SyncPartResh & Async' if dist else 'cpu   &opt &optG & optInt   & SyncPartResh & MS        & F-Fit        & CPVNF' 
        printf (self.output_file, title)         
            
        for mode in modes:
            
            mode_list = [item for item in self.cost_vs_rsrc_data if item['mode']==mode]  
            
            if (len(mode_list)==0): # If no info about this mode - continue
                continue
        
            for cpu_val in set ([item['cpu'] for item in mode_list]): # list of CPU vals for which the whole run succeeded with this mode' 
                list_of_avg_vals.append ({'mode' : mode, 
                                          'cpu'  : cpu_val, 
                                          'cost' : 'y_avg'})

        printf (self.output_file, '\n')
        cpu_vals = sorted (set ([item['cpu'] for item in list_of_avg_vals]))
        min_cpu  = min (cpu_vals)
        for cpu_val in cpu_vals:
            printf (self.output_file, '{:.02f}\t' .format (cpu_val /  min_cpu))
            print ('normalized={}, abs={}' .format (cpu_val / min_cpu, cpu_val))
            if (normalize_Y):
                list_of_val_opt = [item for item in self.cost_vs_rsrc_data if item['mode']=='opt' and item['cpu']==cpu_val]
                if (len(list_of_val_opt)==0):
                    continue;
                for mode in modes:
                    list_of_val = [item for item in self.cost_vs_rsrc_data if item['mode']==mode and item['cpu']==cpu_val]
                    printf (self.output_file, '& $\infty$\t ' if (len(list_of_val)==0) else '& {:.4f}\t ' .format (list_of_val[0]['y_avg']/list_of_val_opt[0]['y_avg'])) 
            else:
                for mode in modes:
                    list_of_val = [item for item in self.cost_vs_rsrc_data if item['mode']==mode and item['cpu']==cpu_val]
                    printf (self.output_file, '& $\infty$\t ' if (len(list_of_val)==0) else '& {:.4f}\t ' .format (list_of_val[0]['y_avg'])) 
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
        
        self.set_plt_params ()
        # print (self.cost_vs_rsrc_data)
        # exit ()

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

    def print_cost_vs_rsrc (self, res_input_file_names=None, min_t=30001, max_t=30060, prob=0.3, dist=True):
        """
        print (to the screen) data about the cost as a function of the amount of resources (actually, cpu capacity at leaf), for various modes.
        * Optional inputs: 
            A list of .res files, containing the results of a run.
        * Calculate and print the average cost for each mode and seed during the whole trace for each seed, the confidence intervals, etc. 
        """
        
        # If the caller provided a .res input file, parse the data from it
        for file_name in res_input_file_names:
            self.parse_res_file(file_name, parse_cost=True, parse_cost_comps=False, parse_num_usrs=False)
    
        modes = ['opt', 'SyncPartResh', 'AsyncBlk', 'AsyncNBlk'] if dist else ['opt', 'optG', 'optInt', 'ourAlg', 'ms', 'ffit', 'cpvnf', 'SyncPartResh']
        for mode in modes:
    
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
                print ('cpu={}, mode={}, avg_cost={}' .format (cpu_val, mode, avg_cost_of_all_seeds)) 

            # # Add this new calculated point to the ds. Avoid duplications of points.
            # for point in sorted (cost_vs_rsrc_data_of_this_mode, key = lambda point : point['cpu']):
            #
            #     list_of_item = list (filter (lambda item : item['cpu']==cpu_val, cost_vs_rsrc_data_of_this_mode)) # all items with this mode, and cpu, already found in self.cost_vs_rsrc_data
            #     print ('cpu={}, mode={}', 'avg_cost={}' .format (list_of_item[0]['cpu'], mode, point['y_avg'])) 

    def parse_comoh_file (self, input_file_name, city=None, numDirections=NUM_DIRECTIONS, stdout=False):
        """
        Parse a .comoh file (files which details the communication overhead - e.g., number and overall size of packets).
        Inputs:
        numDirections - the number of directions in which packets may be sent.
        Assuming a tree whose highest level is lvlOfRoot, the directions are:
        directions 0, 1, ..., lvlOfRoot-1 --> pkts from lvl 0, 1, ..., lvlOfRoot-1 to the level above.
        directions lvlOfRoot, .., 2*lvlOfRoot-1--> pkts from lvl=2*lvlOfRoot-1, ..., lvlOfRoot to the level below.
        stdout: when True, prints to stdout info about the parsed lines
        """
        
        self.city = parse_city_from_input_file_name(input_file_name) if (city==None) else city 
        print ('city is ', self.city)
        self.input_file_name = input_file_name
        self.input_file      = open ("../res/" + input_file_name,  "r")
        lines                = (line.rstrip() for line in self.input_file) # "lines" contains all lines in input file
        lines                = (line for line in lines if line)       # Discard blank lines
        
        for line in lines:
        
            # Discard lines with comments / verbose data
            if (line.split ("//")[0] == ""):
                continue
           
            splitted_line = line.split (" | ")
             
            settings          = splitted_line[0]
            splitted_settings = settings.split ("_")
    
            if len (splitted_settings) != num_of_fields_w_delays:
                print ("encountered a format error. Splitted line={}\nsplitted settings={}" .format (splitted_line, splitted_settings))
                self.dict = None
                return
    
            stts = int (splitted_settings [stts_idx].split("stts")[1])
            self.dict = {
                "t"         : int   (splitted_settings [t_idx]   .split('t')[1]),
                "mode"      : splitted_settings      [mode_idx],
                "cpu"       : int   (splitted_settings [cpu_idx] .split("cpu")[1]),  
                "prob"      : float (splitted_settings [prob_idx].split("p")   [1]),  
                "seed"      : int   (splitted_settings [seed_idx].split("sd")  [1]),  
                "ad"        : float (splitted_settings [ad_idx]  .split("ad")   [1]),  
                "pdd"       : float (splitted_settings [pdd_idx] .split("pdd")  [1]),  
                "stts"      : stts,
            }
            
            if (stts!=1): # if the run failed, the other fields are irrelevant
                continue
    
            for direction in range (numDirections): 
                self.dict['nPkts{}'  .format (direction)] = int (splitted_line[direction+1].split("=")[1])
                self.dict['nBytes{}' .format (direction)] = int (splitted_line[numDirections+direction+1].split("=")[1])
                       
            self.dict['numCritNNewRtUsrs']    = int (splitted_line[2*numDirections+1].split("=")[1])
            self.dict['numCritNNewNonRtUsrs'] = int (splitted_line[2*numDirections+2].split("=")[1])

            if (not(self.dict in self.list_of_dicts)):
                self.list_of_dicts.append(self.dict)                
            
            if (stdout):
                print ('cpu{}_p{}_sd{} normNBytes={:.0f}' .format (self.dict['cpu'], self.dict['prob'], self.dict['seed'], 
                    sum ([self.dict['nBytes{}' .format (direction)] for direction in range(numDirections)])/(self.dict['numCritNNewRtUsrs']+self.dict['numCritNNewNonRtUsrs'])))
        self.input_file.close

    def plot_rsrc_by_ad_pdd (self, city, res_input_file_names):
        """
        Calculate the data needed for plotting a graph showing the required cpu at leaf for finding a feasible sol', as a func' of the acc delay and push-down delay.
        Then, plot a graph, and save it.
        """
        self.set_plt_params ()
        ax = plt.gca()

        for file_name in res_input_file_names:
            self.parse_res_file(input_file_name=file_name, parse_cost=False, parse_cost_comps=False, parse_num_usrs=False, parse_crit_len=False, ignore_worse_lines=True)
                   
        acc_delay_vals = sorted (set ([item['ad']  for item in self.list_of_dicts]))  # values of accumulation delay in the input files
        colors  = ['blue', 'green', 'brown', 'purple', 'black', 'cyan', 'yellow']
        markers = ['x', 'o', 'v', '^', 's', 'h', 'd']
        color_idx = 0

        for pdd2ad_ratio in [1,2,4]:
            avg_cpu_for_this_ratio = []
            for acc_delay in acc_delay_vals:
                data_of_this_ad_n_pdd = list (filter (lambda item : item['ad']==acc_delay and item['pdd']==pdd2ad_ratio*acc_delay, self.list_of_dicts)) #list of results of runs for this accum delay and push-down delay values
                cpu_for_this_ad_n_pdd = [item['cpu'] for item in data_of_this_ad_n_pdd]
                avg_cpu_for_this_ad_n_pdd = np.average(cpu_for_this_ad_n_pdd)
                avg_cpu_for_this_ratio.append (avg_cpu_for_this_ad_n_pdd)
                # [y_lo, y_hi] = (self.conf_interval (ar=cpu_for_this_ad_n_pdd, avg=avg_cpu_for_this_ad_n_pdd))   
                
            self.my_plot (ax=ax, x=acc_delay_vals, y=avg_cpu_for_this_ratio, mode='AsyncNBlk', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=colors[color_idx], marker=markers[color_idx], label='PD delay={:.0f}*Acc delay' .format (acc_delay))
            color_idx += 1
        ax.legend (ncol=2, fontsize=LEGEND_FONT_SIZE) #  loc='upper right') 
        plt.ylabel('Min Cpu at Leaf [GHz]')
        plt.xlabel('Accumulation Delay [us]')
        plt.savefig ('../res/{}_cpu_by_delays.pdf' .format (city), bbox_inches='tight')
        plt.cla()

    
    def plot_comoh_by_Rt_prob (self, city, comoh_input_file_names, numDirections=NUM_DIRECTIONS):
        """
        Calculate the data needed for plotting a graph showing the communication overhead as a func' of the RT prob'.
        Then, plot a graph, and save it.
        """
        self.set_plt_params ()
        ax = plt.gca()

        self.comoh_data = [] # this field will hold all the data to be parsed from the comoh input files
        for file_name in comoh_input_file_names:
            self.parse_comoh_file (file_name, city=city, numDirections=numDirections)
            
        
        acc_delay_vals = sorted (set ([item['ad']  for item in self.list_of_dicts]))  # values of accumulation delay in the input files
        colors  = ['blue', 'green', 'brown', 'purple', 'black', 'cyan', 'yellow']
        markers = ['x', 'o', 'v', '^', 's', 'h', 'd']
        color_idx = 0

        for acc_delay in [acc_delay for acc_delay in acc_delay_vals if acc_delay in [0, 1, 10, 100]]:
            data_of_this_ad = list (filter (lambda item : item['ad']==acc_delay, self.list_of_dicts)) #list of results of runs for this accum delay value

            prob_vals = sorted (set ([item['prob'] for item in data_of_this_ad])) # values of prob' collected for this acc delay

            avg_nBytes_per_req_for_this_ad = []
            for prob in prob_vals:  
            
                data_of_this_ad_n_prob = list (filter (lambda item : item['prob']==prob, data_of_this_ad)) #list of results of runs for this acc delay and prob values
                seeds = [item['seed'] for item in data_of_this_ad_n_prob]
                nBytes_per_req_for_this_ad_n_prob = [] # will hold a list of overall nBytes/pkts per request, in all directions, for each simulated seed.
                for seed in seeds:
                    
                    nBytes_per_req_for_this_ad_n_prob.append (sum ([item['nBytes{}' .format (direction)]/(item['numCritNNewRtUsrs']+item['numCritNNewNonRtUsrs']) 
                                                             for direction in range(numDirections) for item in data_of_this_ad_n_prob if item['seed']==seed]))
                avg_nBytes_per_req_for_this_ad_n_prob = np.average(nBytes_per_req_for_this_ad_n_prob)
                avg_nBytes_per_req_for_this_ad.append (avg_nBytes_per_req_for_this_ad_n_prob)
                [y_lo, y_hi] = (self.conf_interval (ar=nBytes_per_req_for_this_ad_n_prob, avg=avg_nBytes_per_req_for_this_ad_n_prob))
                self.comoh_data.append ({'type' : type,
                                         'y_avg' : avg_nBytes_per_req_for_this_ad_n_prob, 
                                         'y_lo' : y_lo, 
                                         'y_hi' : y_hi, 
                                         'num_of_seeds' : len(seeds), 
                                         'dir' : OVERALL_DIR})

                
            self.my_plot (ax=ax, x=prob_vals, y=avg_nBytes_per_req_for_this_ad, mode='AsyncNBlk', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=colors[color_idx], marker=markers[color_idx], label='acc delay={:.0f}[us]' .format (acc_delay))
            color_idx += 1
        ax.legend (ncol=2, fontsize=LEGEND_FONT_SIZE) #  loc='upper right') 
        plt.xlim(0,1)
        plt.ylabel('Control Bytes/Request')
        plt.xlabel('Fraction of RT Chains')
        plt.savefig ('../res/{}_comoh.pdf' .format (city), bbox_inches='tight')
        plt.cla()
    
    def calc_comoh_by_cpu (self, city, pcl_output_file_name, pcl_input_file_name=None, res_input_file_names=None, prob=0.3, numDirections=NUM_DIRECTIONS):
        """
        Calculate the data needed for plotting a graph showing the communication overhead as a func' of the cpu at the leaf.
        * Optional inputs: 
            .pcl file, containing self.list_of_dicts (usually as a result of a previous run of self.parse_res_file ()).
            A list of .res files, containing the results of a run.
            At least one file (either .pcl, or .res file) should be given       
        * Calculate the average cost for setting. 
        * Save the (pickled) processed data into pcl_output_file_name..
         """
        if (pcl_input_file_name==None and res_input_file_names==None):
            print ('Error: calc_cost_vs_rsrc must be called with at least one input file - either a .pcl file, or a .res file')
            return
    
        # If the caller provided a .pcl input file, read the data from it
        if (pcl_input_file_name==None):
            self.comoh_data = []
        else:
            self.comoh_data = pd.read_pickle(r'../res/pcl_files/{}' .format (pcl_input_file_name))
    
        # If the caller provided a .res input file, parse the data from it
        for file_name in res_input_file_names:
            self.parse_comoh_file (file_name, city=city, numDirections=numDirections)

        cpu_vals = sorted (set ([item['cpu'] for item in self.list_of_dicts]))

        for cpu_val in cpu_vals:  
            
            data_of_this_cpu = list (filter (lambda item : item['cpu']==cpu_val, self.list_of_dicts)) #list of results of runs for this cpu value
            seeds = [item['seed'] for item in data_of_this_cpu]
            for type in ['nPkts', 'nBytes']:
                comoh_per_req_for_this_cpu_n_type = []
                for seed in seeds:
                    # comoh_per_req_for_this_cpu_n_type will hold a list of overall nBytes/pkts per request, in all directions, for each simulated seed.
                    comoh_per_req_for_this_cpu_n_type.append (sum ([item['{}{}' .format (type, direction)]/(item['numCritNNewRtUsrs']+item['numCritNNewNonRtUsrs']) 
                                                             for direction in range(numDirections) for item in data_of_this_cpu if item['seed']==seed]))
                avg_comoh_per_req_for_this_cpu_n_type = np.average(comoh_per_req_for_this_cpu_n_type)
                [y_lo, y_hi] = (self.conf_interval (ar=comoh_per_req_for_this_cpu_n_type, avg=avg_comoh_per_req_for_this_cpu_n_type ))
                self.comoh_data.append ({'cpu' : cpu_val, 
                                         'type' : type,
                                         'y_avg' : avg_comoh_per_req_for_this_cpu_n_type, 
                                         'y_lo' : y_lo, 
                                         'y_hi' : y_hi, 
                                         'num_of_seeds' : len(seeds), 
                                         'dir' : OVERALL_DIR})

            # for direction in range(numDirections):
            #     for type in ['nPkts', 'nBytes']:
            #         data_of_this_cpu_and_dir = [item['{}{}' .format (type, direction)] for item in data_of_this_cpu]
            #         cur_avg = int (round (np.average (data_of_this_cpu_and_dir)))
            #         [y_lo, y_hi] = (self.conf_interval (ar=data_of_this_cpu_and_dir, avg=cur_avg)) # low, high y values for this plotted conf' interval
            #         self.comoh_data.append ({'cpu' : cpu_val, 'y_lo' : y_lo, 'y_hi' : y_hi, 'y_avg' : cur_avg, 'num_of_seeds' : len(data_of_this_cpu_and_dir), 'type' : type, 'dir' : direction })
        
        
        # store the data as binary data stream
        with open('../res/pcl_files/' + pcl_output_file_name, 'wb') as comoh_data_file:
            pickle.dump(self.comoh_data, comoh_data_file)

    
    def plot_comoh_by_cpu (self, pcl_input_file_name):
        """
        Plot the comm' o/h (num of pkts / num of bytes), as a function of the cpu.
        """
        self.set_plt_params ()
        self.comoh_data = pd.read_pickle(r'../res/pcl_files/{}' .format (pcl_input_file_name))
        ax = plt.gca()
        
        cpu_norm_factor = 89 if self.city=='Lux' else 840 # normalization factor for x axis: the minimal cpu for which opt finds a feasible sol
        cpu_vals = sorted (list (set ([item['cpu'] for item in self.comoh_data]))) # list of cpu vals for which there exist data
        normalized_cpu_vals, overall_nPkts, overall_nBytes = [], [], []
        plot_types = ['nPkts', 'nBytes']
        overall = {plot_types[0] : [], plot_types[1] : []}
        for cpu_val in cpu_vals:
            cpu_val_data = [item for item in self.comoh_data if item['cpu']==cpu_val] 
            normalized_cpu_val = cpu_val/cpu_norm_factor
            normalized_cpu_vals.append (normalized_cpu_val)
            for type in plot_types: 
                list_of_item = [item for item in cpu_val_data if item['type']==type]
                if (type in ['nPkts', 'nBytes']):
                    list_of_item = [item for item in list_of_item if item['dir']==OVERALL_DIR]
                if (len(list_of_item)<1):
                    print ('error in plot_comoh_by_cpu: could not find entry for overall {}' .format (type))
                    exit () 
                item = list_of_item[0]
                overall[type].append(item['y_avg'])

        # nonRtUsrChainOh = 20 
        for type in ['nPkts', 'nBytes']:
            self.my_plot (ax=ax, x=normalized_cpu_vals, y=overall[type], mode='AsyncNBlk', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=None, label='Communication Overhead per Request') 
                #  'label='Number of {} per Request' .format ('Bytes' if type=='nBytes' else 'Packets'))
            ax.legend (ncol=2, fontsize=LEGEND_FONT_SIZE, loc='upper right') 
            plt.ylabel('Communication Overhead {}' .format ('[Bytes]' if type=='nBytes' else '[# Packets]'))
            plt.xlabel(r'$C_{cpu} / \hat{C}_{cpu}$')
            # if (type=='nBytes'):
            #     ax.plot (normalized_cpu_vals, [item*nonRtUsrChainOh*(TREE_HEIGHT-1)   for item in overall['critNNewNonRtUsrs']], color=self.color_dict['opt'],   marker=self.markers_dict['opt'],   markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Intuitive LBound')
            #     ax.plot (normalized_cpu_vals, [item*nonRtUsrChainOh*(TREE_HEIGHT-1)*4 for item in overall['critNNewNonRtUsrs']], color=self.color_dict['cpvnf'], marker=self.markers_dict['cpvnf'], markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Intuitive HBound', markerfacecolor='none')
            ax.legend (ncol=1, fontsize=LEGEND_FONT_SIZE, loc='upper right') 
            plt.savefig ('../res/{}_p0.3_{}.pdf' .format (city, type), bbox_inches='tight')
            plt.cla()
            cpu_val_list = [item for item in self.comoh_data if item['cpu']==cpu_val]
            nPkts_list   = [item for item in cpu_val_list if item['type']=='nPkts']
            nByts_list   = [item for item in cpu_val_list if item['type']=='nPkts']
            overall_nPkts .append (sum (item['y_avg'] for item in nPkts_list))
            overall_nBytes.append (sum (item['y_avg'] for item in nByts_list))

        # self.my_plot (ax=ax, x=[item/cpu_norm_factor for item in  cpu_vals], y=overall_nPkts, mode='AsyncBlk', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=None) 
        # self.my_plot (ax=ax, x=[item/cpu_norm_factor for item in  cpu_vals], y=overall_nBytes, mode='AsyncBlk', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=None) 

        # ax.plot (cpu_vals, overall_nPkts)#, color = 'black') #, marker=None, linewidth=LINE_WIDTH, label=city if city=='Monaco' else 'Luxembourg')
 
            # print ('overall_nPkts={}' .format (overall_nPkts))
            
        # my_plot (ax, x, y, mode='ourAlg', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=None): 
        # ax.plot (x, m*x+b, linewidth=LINE_WIDTH_SMALL)
        # plt.xlim (0, 3600)
        # plt.ylim (0)
        # ax.legend (fontsize=22, loc='center') 
        # ax.plot ((cpu_val ,cpu_val), (y_lo, y_hi), color=self.color_dict[mode]) # Plot the confidence interval
        # plt.show ()

        # plt.savefig ('../res/tot_num_of_vehs_0730_0830.pdf', bbox_inches='tight')

    def calc_cost_vs_rsrc (self, pcl_input_file_name=None, res_input_file_names=None, min_t=30001, max_t=30600, prob=0.3, dist=True):
        """
        Calculate the data needed for plotting a graph showing the cost as a function of the amount of resources (actually, cpu capacity at leaf):
        * Optional inputs: 
            .pcl file, containing self.list_of_dicts (usually as a result of a previous run of self.parse_res_file ()).
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
            self.cost_vs_rsrc_data = pd.read_pickle(r'../res/pcl_files/{}' .format (pcl_input_file_name))
    
        # If the caller provided a .res input file, parse the data from it
        for file_name in res_input_file_names:
            self.parse_res_file(file_name, parse_cost=True, parse_cost_comps=False, parse_num_usrs=False)

        modes = ['opt', 'SyncPartResh', 'AsyncBlk', 'AsyncNBlk'] if dist else ['opt', 'optG', 'optInt', 'ourAlg', 'ms', 'ffit', 'cpvnf', 'SyncPartResh']   
        for mode in modes:
    
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
        self.pcl_output_file_name = 'cost_vs_rsrc_{}.pcl' .format (self.input_file_name.split('.res')[0]) if (pcl_input_file_name==None) else pcl_input_file_name 
        with open('../res/pcl_files/' + self.pcl_output_file_name, 'wb') as cost_vs_rsrc_data_file:
            pickle.dump(self.cost_vs_rsrc_data, cost_vs_rsrc_data_file)
        return self.pcl_output_file_name 

    def calc_mig_cost_vs_rsrc (self, prob=0.3, res_input_file_names=None, pcl_input_file_name=None):
        """
        Calculate the data needed for plotting a graph showing the migration cost / number of migrated chains / number of reshuffles, as a function of the amount of resources (actually, cpu capacity at leaf):
        * Optional inputs: 
            .pcl file, containing self.list_of_dicts (usually as a result of a previous run of self.parse_res_file ()).
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
                self.parse_res_file(file_name, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True)
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
            
            mode_cpu_list             = list (filter (lambda item : item['cpu']==cpu_val, mode_list)) # list of results of runs for this mode, and cpu value
            avg_mig_cost_of_each_seed = [] # will hold the avg mig cost of each successful run with a given mode, cpu value, and seed (averaging the cost over all the slots in the trace)
            avg_num_crit_of_each_seed = [] # will hold the avg num of critical chains of each successful run with a given mode, cpu value, and seed (averaging the cost over all the slots in the trace)
            for seed in set ([item['seed'] for item in mode_cpu_list]): # list of seeds for which the whole run succeeded with this mode (algorithm), and this cpu val
                avg_mig_cost_of_each_seed.append (np.average ([item['mig_cost'     ] for item in mode_cpu_list if item['seed']==seed]))                    
                avg_num_crit_of_each_seed.append (np.average ([item['num_crit_usrs'] for item in mode_cpu_list if item['seed']==seed]))
            avg_mig_cost_of_all_seeds = np.average (avg_mig_cost_of_each_seed)
            [y_lo, y_hi]              = self.conf_interval (ar=avg_mig_cost_of_each_seed, avg=avg_mig_cost_of_all_seeds) # low, high y values for this plotted conf' interval
            
            mig_vs_rsrc_data.append ({'cpu'          : cpu_val, 
                                      'y_lo'         : y_lo, 
                                      'y_hi'         : y_hi, 
                                      'y_avg'        : avg_mig_cost_of_all_seeds,
                                      'num_of_crit'  : np.average (avg_num_crit_of_each_seed), 
                                      'num_of_seeds' : len(avg_mig_cost_of_each_seed)})
        
        # Add this new calculated point to the ds. Avoid duplications of points.
        for point in sorted (mig_vs_rsrc_data, key = lambda point : point['cpu']):
            
            list_of_item = list (filter (lambda item : item['cpu']==cpu_val, self.mig_vs_rsrc_data)) # all items with this mode, and cpu, already found in self.cost_vs_rsrc_data
            if (point not in self.mig_vs_rsrc_data and len(list_of_item)==0): # insert this new point to the list of points only if it's not already found in self.cost_vs_rsrc_data
                self.mig_vs_rsrc_data.append (point)
        
        # store the data in a '.pcl' file (binary data stream)
        pcl_output_file_name = '{}_mig_cost_vs_rsrc.pcl' .format (self.city if pcl_input_file_name==None else parse_city_from_input_file_name (pcl_input_file_name)) 
        with open('../res/' + pcl_output_file_name, 'wb') as mig_vs_rsrc_data_file:
            pickle.dump(mig_vs_rsrc_data, mig_vs_rsrc_data_file)
        return pcl_output_file_name 
    
    def calc_crit_chains_vs_rsrc (self, res_input_file_names=None, min_t=30001, max_t=30600, prob=0.3, pcl_input_file_name=None):
        """
        Calculate the data needed for plotting a graph showing the migration cost / number of migrated chains / number of reshuffles, as a function of the amount of resources (actually, cpu capacity at leaf):
        * Optional inputs: 
            .pcl file, containing self.list_of_dicts (usually as a result of a previous run of self.parse_res_file ()).
            A list of .res files, containing the results of a run.
            At least one file (either .pcl, or .res file) should be given       
        * Calculate the average # of migrated chains for each seed during the whole trace for each seed, the confidence intervals, etc. 
        * Save the (pickled) processed data into self.cost_vs_rsrc_data.pcl.
        * Returns the file name to which it saved the pickled results. 
        """
        
        self.mig_vs_rsrc_data = []
        
        # If the caller provided a .res input file, parse the data from it
        for file_name in res_input_file_names:
            self.parse_res_file(file_name, parse_cost=True, parse_cost_comps=True, parse_num_usrs=False)
    
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
    
    def gen_mig_vs_rsrc_tbl (self, city=None, pcl_file_name=None):
        """
        Generate a table of the number of the percentage of migrated chains that are non-critical, out of all migrated chains.
        Input is either the city (and then, a default pcl file name is saerhced); or the pcl_file_name (in which case, the city's
        name is parsed from the pcl_file_name).
        """
        
        if (city==None and pcl_file_name==None):
            print ("error: plz specify either a city and/or a pcl file name.")
            exit ()
        if (pcl_file_name==None):
            pcl_file_name = '{}_mig_cost_vs_rsrc.pcl' .format (city)
        else:
            city = self.parse_city_from_input_file_name (pcl_file_name)
        
        mig_vs_rsrc_data = pd.read_pickle(r'../res/pcl_files/{}' .format (pcl_file_name))
        
        # NUM_OF_SLOTS = 600
        CPU_VALS_NORM_FACTOR = {'Lux' : 89, 'Monaco' : 840}
        nomalized_cpu        = [item['cpu']/CPU_VALS_NORM_FACTOR[city] for item in mig_vs_rsrc_data]
        all_migs_num         = [item['y_avg']/UNIFORM_CHAIN_MIG_COST for item in mig_vs_rsrc_data]
        averall_mig_cost     = [item['y_avg'] for item in mig_vs_rsrc_data]
        # crit_migs_num        = np.array ([item['num_crit_usrs']                for item in mig_vs_rsrc_data]) - avg_new_vehs_per_slot[city]
        # non_crit_percent     = [100*(all_migs_num[i] - crit_migs_num[i])/all_migs_num[i] for i in range(len(all_migs_num))]
             
        print ('city is {}' .format (city))
        for i in range(len(all_migs_num)):
            print ('{}\t{:.0f}' .format (nomalized_cpu[i], averall_mig_cost[i]))
            # print ('{}\t{}' .format (nomalized_cpu, non_crit_percent))

    def plot_mig_vs_rsrc (self, pcl_input_file_name):
        """
        Generate a plot of the number of migrations vs. the cpu capacity at the leaf, for the given city.
        The plot is saved in a .pdf file.
        The plot is based on the data found in the given input .pcl file.
        """

        city             = parse_city_from_input_file_name(pcl_input_file_name)
        mig_vs_rsrc_data = pd.read_pickle(r'../res/{}' .format (pcl_input_file_name))
        self.set_plt_params ()
        _, ax = plt.subplots()
        ax.set_xlabel ('CPU at leaf [GHz]')
        ax.set_ylabel ('# of Mig. Chains')
        
        # Plot the avg # of migration as a func' of the rsrc (cpu at leaf) 
        ax.plot ([item['cpu']/10 for item in mig_vs_rsrc_data], [item['y_avg']/UNIFORM_CHAIN_MIG_COST for item in mig_vs_rsrc_data], markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

        # Plot the conf' intervals for the # of migration as a func' of the rsrc (cpu at leaf) 
        for item in mig_vs_rsrc_data:
            ax.plot ( (item['cpu']/10, item['cpu']/10), (item['y_lo']/UNIFORM_CHAIN_MIG_COST, item['y_hi']/UNIFORM_CHAIN_MIG_COST), color=self.color_dict['ourAlg']) # Plot the confidence interval
        ax.plot ( (item['cpu']/10, item['cpu']/10), (item['y_lo']/UNIFORM_CHAIN_MIG_COST, item['y_hi']/UNIFORM_CHAIN_MIG_COST), color=self.color_dict['ourAlg']) # Plot the confidence interval
        vertical_line_x = 23.5 if (city=='Lux') else 130
        plt.axvline (vertical_line_x, color='black', linestyle='dashed')
        plt.xlim (0, np.max ([item['cpu']/10 for item in mig_vs_rsrc_data]))
        printFigToPdf ('{}_num_migs_vs_rsrc' .format (city))

    def parse_res_files_w_distinct_T (self, input_res_filenames, input_pcl_file_name=None):
        """
        Parse each of the given files. Add to its entry in self.list_of_dicts a field, presenting the length of the time slot.
        Save the results in a .pcl file.
        """
    
        if (input_pcl_file_name==None): # No pcl input file was given --> start the data structure from scratch
            full_list_of_dicts = [] # will hold all the dictionary items of the parsed data
        else: # A pcl input file was given --> read it, and incrementally add to it the data parsed from the ".res" input files
            full_list_of_dicts = pd.read_pickle ('../res/'+ input_pcl_file_name)
        for filename in input_res_filenames:
            
            self.list_of_dicts = []
            T = self.parse_T_from_input_file_name (filename)
            self.parse_res_file (input_file_name = filename, parse_cost=True, parse_cost_comps=True, parse_num_usrs=True)
            
            for item in self.list_of_dicts:
                item['T'] = T
                if (not(item in full_list_of_dicts)):
                    full_list_of_dicts.append(item)                
                full_list_of_dicts.append (item)
                
        pcl_output_file_name = '{}_vary_T.pcl' .format (self.city)
        with open ('../res/' + pcl_output_file_name, 'wb') as pcl_output_file:
            pickle.dump (full_list_of_dicts, pcl_output_file)
                
        return pcl_output_file_name
    
    def plot_crit_n_mig_vs_T (self, pcl_input_file_name, y_axis='num_crit', per_slot=True, plot_conf_intervals=False):
        """
        plot the number of critical chains (on one y-axis), and the migration cost (on the other y-axis), vs. the slot interval, T.
        Inputs: 
        pcl_input_file_name - a .pcl file, containing a list_of_dicts, namely, a list of dictionaries with all the results.
        y_axis - can be either:
            - 'crit_num' --> plot only the # of critical chains.
            - 'mig_cost' --> plot only the mig' cost.
        resh - when False, consider only mig' cost of critical chains (not of those happened due to reshuffle).
        per_slot - when True, plot the mig' cost per slot. When False, plot the overall mig' cost in the trace.  
        """

        self.list_of_dicts = pd.read_pickle ('../res/{}' .format (pcl_input_file_name))
        
        list_of_Ts = sorted (set ([item['T'] for item in self.list_of_dicts])) # list_of_Ts is the list of all slots for which there're results 
        
        self.set_plt_params ()
        
        city = parse_city_from_input_file_name (pcl_input_file_name)
        cpu = 103 if city=='Lux' else 926
        
        plt.xlabel  ('Decision Period Duration [s]', fontsize=FONT_SIZE)        
        y_num_crit, y_mig_cost, y_mig_cost_wo_resh = [], [], []
        for T in list_of_Ts:
            list_of_dicts_T = [item for item in self.list_of_dicts if item['T']==T and item['cpu']==cpu] # list_of_dicts_T <-- list of results when simulated with time slot==T.
            
            seeds = set ([item['seed'] for item in list_of_dicts_T])
            
            avg_num_crit_chains_per_sd, mig_cost_per_sd = [], []
            for sd in seeds:
                list_of_dicts_T_sd = [item for item in list_of_dicts_T if item['seed']==sd] # list_of_dicts_T_sd is the list of items with this time slot length, and this seed
                avg_num_crit_chains_per_sd.append (np.average ([item['num_crit_usrs'] for item in list_of_dicts_T_sd]))
                mig_cost_per_sd.           append (np.sum     ([item['mig_cost']      for item in list_of_dicts_T_sd]))
            
            per_slot_norm_factor = len (list_of_dicts_T_sd) if per_slot else 1 # If plotting in a "per_slot" manner, we will normalize by the real # of slots. Else, "normalize" by 1 (actually, do nothing)  
            
            avg_num_crit_chains = np.average (avg_num_crit_chains_per_sd)
            mig_cost            = np.average (mig_cost_per_sd)
            
            avg_num_crit_chains_conf_interval = self.conf_interval (avg_num_crit_chains_per_sd, avg_num_crit_chains)                
            mig_cost_conf_interval            = self.conf_interval (mig_cost_per_sd,        mig_cost)
            mig_cost_wo_resh_conf_interval    = (np.array (avg_num_crit_chains_conf_interval) - avg_new_vehs_per_slot[city][T]) * UNIFORM_CHAIN_MIG_COST * len (list_of_dicts_T_sd) 
            
            if (plot_conf_intervals):
                if (y_axis=='num_crit'):
                    plt.plot ((T, T), avg_num_crit_chains_conf_interval, color='black') # Plot the confidence interval
                elif (y_axis=='mig_cost'):  
                    plt.plot ((T, T), np.array(mig_cost_conf_interval)/per_slot_norm_factor, color='blue') # Plot the confidence interval
                    plt.plot ((T, T), np.array(mig_cost_wo_resh_conf_interval)/per_slot_norm_factor, color='black') # Plot the confidence interval                

            # We calculate the cost of migrations excluding reshuffles as follows:
            # avg. num of critical chains that are not new per slot * num of slots * per-chain mig' cost 
            y_mig_cost_wo_resh.append ((avg_num_crit_chains - avg_new_vehs_per_slot[city][T]) * UNIFORM_CHAIN_MIG_COST * len (list_of_dicts_T_sd) / per_slot_norm_factor)
            y_mig_cost.        append (mig_cost / per_slot_norm_factor) #(mig_cost) 
            y_num_crit.        append (avg_num_crit_chains)
                
        self.set_plt_params ()
        plt.xlim (min(list_of_Ts), max(list_of_Ts))
        if (y_axis =='mig_cost'):
            plt.ylabel  ('Mig. Cost per Decision Period' if per_slot else 'Total Mig. Cost', fontsize=FONT_SIZE, labelpad=16.5 if (per_slot and city=='Monaco') else 4)
            plt.tick_params (axis='y')
            
            my_y_lim (y_mig_cost)
            plt.fill_between (x=list_of_Ts, y1=0,y2=y_mig_cost_wo_resh, color='blue', label='Critical chains') 
            plt.fill_between (x=list_of_Ts, y1=y_mig_cost_wo_resh, y2=y_mig_cost, color='deepskyblue', label='Non-critical chains')
            if (per_slot):
                plt.plot (list_of_Ts, y_mig_cost_wo_resh[0]*np.array (list_of_Ts),  
                                    label='Pessimistic', mfc='none', color='black')
            my_y_lim (y_mig_cost)
            
            # Force using the desired scientific notation format
            MIG_COST_EXP, MIG_COST_PER_SLOT_EXP = 8, 5 
            plt.ticklabel_format (axis="y", style="sci", scilimits=(MIG_COST_PER_SLOT_EXP,MIG_COST_PER_SLOT_EXP) if per_slot else (MIG_COST_EXP,MIG_COST_EXP))
            if (not (per_slot) and city=='Monaco'):
                # plt.yticks (np.array ([0 , 4, 8, 12])*100000000)
                plt.yticks (np.array ([0 , 4, 8, 12])*10000000)
            plt.legend (loc='lower right' if (city=='Lux') else 'lower right', fontsize=LEGEND_FONT_SIZE)

        else:
            # plt.ylabel  ('Avg. # of Critical Chains', fontsize=FONT_SIZE)
            plt.fill_between (x=list_of_Ts, y1=0, y2=y_num_crit, color='blue', label='Avg. # of Critical Chains')
            plt.plot (list_of_Ts, y_num_crit[0]*np.array (list_of_Ts), label='Pessimistic', mfc='none', color='black')
            plt.legend (loc='upper left', fontsize=LEGEND_FONT_SIZE)
            
            plt.ylim (0, (max (max (y_num_crit[0]*np.array (list_of_Ts)), max (y_num_crit)))*1.05)
        
        if (y_axis=='num_crit'):
            printFigToPdf('{}_crit_vs_T' .format (city))
        else:
            printFigToPdf('{}_mig_cost_{}vs_T' .format (city, 'per_slot_' if per_slot else ''))
        plt.cla ()

    def plot_crit_len (self, city):
        """
        Calculate a histogram of the overall # of vehs for each duration of being critical 
        """
        list_of_hists = pd.read_pickle ('../res/{}_crit_len.pcl' .format (city)) 

        list_of_Ts = [item['slot_len'] for item in list_of_hists]
        
        self.set_plt_params ()
        
        crit_len, num_of_crit_chains = [], []
        # print ('all hists={}' .format (list_of_hists))
        for T in list_of_Ts:
            hist_of_this_T = ([item['hist'] for item in list_of_hists if item['slot_len']==T])[0]
            t_range =  range(len(hist_of_this_T))
            num_of_crit_chains.append (np.sum (hist_of_this_T))
            crit_len.append (np.sum([hist_of_this_T[t]*t for t in t_range]) / np.sum (hist_of_this_T)) 
        
        _, ax = plt.subplots ()
        ax.plot (list_of_Ts, crit_len, color='black', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE, mfc='none')#, label=self.legend_entry_dict[mode], mfc='none')
        plt.xlim (min(list_of_Ts), max(list_of_Ts))
        plt.ylim (0, max(list_of_Ts)/2+1)
        ax.set_xlabel ('Decision Period T [s]')
        ax.set_ylabel ('Avg. Violation Time [s]') 
        printFigToPdf('{}_crit_len' .format (city))

        _, ax = plt.subplots ()
        ax.plot (list_of_Ts, num_of_crit_chains, color='black', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE)#, label=self.legend_entry_dict[mode], mfc='none')
        # plt.xlim (min(list_of_Ts), max(list_of_Ts))
        ax.set_xlabel ('Decision Period T [s]')
        ax.set_ylabel ('# of Critical Chains') 
        printFigToPdf('{}_num_crit_chains' .format (city))

    def calc_crit_len (self, city):
        """
        Calculate a histogram of the overall # of vehs for each duration of being critical 
        """
    
        list_of_Ts = range (1, 11)
        list_of_hists = [] # list of historgrams of the distribution of the durations of each critical chain being critical.  
        pcl_output_file_name = '{}_crit_len.pcl' .format (city)
        for T in list_of_Ts:
            self.crit_len_cnt = np.zeros (T+1) # self.crit_len_cnt[T] will hold the overall # of critical chains that were critical for T seconds. 
            self.parse_res_file (input_file_name='{}_crit_len_T{}.res' .format (city, T), parse_cost=False, parse_cost_comps=False, parse_num_usrs=False, parse_crit_len=True, time_slot_len=T)
            list_of_hists.append ({'slot_len' : T, 'hist' : self.crit_len_cnt})
    
        with open ('../res/{}' .format (pcl_output_file_name), 'wb') as pcl_file:
            pickle.dump (list_of_hists, pcl_file)
            
        return pcl_output_file_name
 
    def erase_from_pcl (self, pcl_input_file_name, mode_to_erase='Async'):
        """
        Read a .pcl file, erase all the entries of a given mode, and over-write into the same .pcl file
        """
        
        pcl_full_path_file_name = '../res/pcl_files/{}' .format (pcl_input_file_name)
        self.list_of_dicts = pd.read_pickle(r'{}' .format (pcl_full_path_file_name))
        
        self.list_of_dicts = [item for item in self.list_of_dicts if item['mode']!=mode_to_erase]
        with open(pcl_full_path_file_name, 'wb') as pcl_file:
            pickle.dump(self.list_of_dicts, pcl_file)


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
    # pcl_output_file_name = my_res_file_parser.calc_mig_cost_vs_rsrc (pcl_input_file_name='Lux_0730_0830_1secs_post_p0.3_ourAlg.pcl' if city=='Lux' else 'Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg.pcl')
    # my_res_file_parser.plot_mig_vs_rsrc (pcl_input_file_name = pcl_output_file_name) #'Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg_num_mig_chains_vs_rsrc.pcl')
    my_res_file_parser.plot_mig_vs_rsrc (pcl_input_file_name = '{}_mig_cost_vs_rsrc.pcl' .format (city))
         
def plot_crit_n_mig_vs_T (city, y_axis='mig_cost', per_slot=True, prepare_new_pcl_file=False):

    """
    Plot the number of critical chains and the number of migrations as a function of the time slot, T.
    """
    
    my_res_file_parser = Res_file_parser ()
    
    if (prepare_new_pcl_file):
        input_res_filenames = []
        if (city=='Lux'):
            for T in range (1, 11):
                input_res_filenames.append ('Lux_0730_0830_{}secs_post_SyncPartResh.res' .format (T))
        else:
            for T in range (1, 11):
                input_res_filenames.append ('Monaco_0730_0830_{}secs_Telecom_SyncPartResh.res' .format (T))
        my_res_file_parser.parse_res_files_w_distinct_T(input_res_filenames)
    my_res_file_parser.plot_crit_n_mig_vs_T (pcl_input_file_name='{}_vary_T.pcl' .format (city), y_axis=y_axis, per_slot=per_slot)

def plot_RT_prob_sim (city):
    """
    Generate a plot of the minimal required resources to find a feasible sol', as a function of the ratio of requests with RT requirements.
    The plot is saved as a .pdf in the '../res' directory.    
    """
    my_res_file_parser = Res_file_parser ()
    if (city=='Lux'):            
        my_res_file_parser.plot_RT_prob_sim_python ('RT_prob_sim_Lux.post.antloc_256cells.poa2cell_Lux_0820_0830_1secs_post.poa.res')
    else:
        my_res_file_parser.plot_RT_prob_sim_python ('RT_prob_sim_Monaco.Telecom.antloc_192cells.poa2cell_Monaco_0820_0830_1secs_Telecom.poa.res')


def plot_cost_vs_rsrc (city):
    """
    Generate a plot of the cost, as a function of the resources. 
    """

    my_res_file_parser = Res_file_parser ()
    if (city=='Monaco'):
        pcl_input_file_name = my_res_file_parser.calc_cost_vs_rsrc (res_input_file_names=['Monaco_0820_0830_1secs_Telecom_p0.3_opt.res']), 
    pcl_input_file_name = 'cost_vs_rsrc_Monaco_0820_0830_1secs_Telecom_p0.3.pcl' if (city=='Monaco') else 'cost_vs_rsrc_Lux_0820_0830_1secs_post_p0.3.pcl'
    my_res_file_parser.plot_cost_vs_rsrc (pcl_input_file_name=pcl_input_file_name)
       
if __name__ == '__main__':

    city = 'Lux'
    my_res_file_parser = Res_file_parser ()
    my_res_file_parser.plot_rsrc_by_ad_pdd(city=city, res_input_file_names='{}_cpu_by_delays.res' .format (city))
    # my_res_file_parser.plot_comoh_by_Rt_prob(city=city, comoh_input_file_names=['{}.comoh' .format (city)])
    exit ()
    # Generate a Rt_prob_sim plot
    city = 'Monaco'
    my_res_file_parser = Res_file_parser ()
    pcl_input_file_name = '{}_RtProb_0820_0830_1secs.pcl' .format (city)
    my_res_file_parser.dump_self_list_of_dicts_to_pcl (pcl_input_file_name=pcl_input_file_name, res_file_names=['{}_RtProb_AsyncNBlk_1secs.res' .format (city)])
    my_res_file_parser.plot_RT_prob_sim_python (pcl_input_file_name=pcl_input_file_name, dist=True)
    
    # plot_crit_n_mig_vs_T (city=city, y_axis='mig_cost', per_slot=False)
    # city = 'Lux'
    # my_res_file_parser = Res_file_parser ()
    # comoh_file = '{}.comoh' .format (city)
    # res_input_file_name = 'Lux_p0.0_hdr0B_NonRt20B.comoh' #{}.comoh' .format (city)
    # pcl_output_file_name='{}_0hdr_20BnonRt_p0.0.comoh.pcl' #'{}.comoh.pk' .format (city)
    # my_res_file_parser.calc_comoh_by_cpu (city=city, pcl_output_file_name=pcl_output_file_name, pcl_input_file_name=None, res_input_file_names=[res_input_file_name], prob=0.3)
    # my_res_file_parser.plot_comoh_by_cpu (pcl_input_file_name=pcl_output_file_name)

    # city = 'Monaco'
    # my_res_file_parser = Res_file_parser ()
    # my_res_file_parser.parse_comoh_file(input_file_name='Monaco_0.5_0.5_acc_delay.comoh', city=city, numDirections=NUM_DIRECTIONS, stdout=True)
    # exit ()
    # comoh_file = '{}.comoh' .format (city)
    # my_res_file_parser.calc_comoh_by_cpu (city=city, pcl_output_file_name='{}.comoh.pcl' .format (city), pcl_input_file_name=None, res_input_file_names=['{}.comoh' .format (city)], prob=0.3)
    # my_res_file_parser.plot_comoh_by_cpu (pcl_input_file_name='{}.comoh.pcl' .format (city))

    # city = 'Monaco'
    # my_res_file_parser = Res_file_parser ()
    # my_res_file_parser.erase_from_pcl(pcl_input_file_name='Monaco_dist_cost_vs_rsrc_0820_0830_1secs_p0.3.pcl')
    
    # res_input_file_name = '{}_0820_0830_1secs_p0.3_Async.res' .format (city)
    # pcl_input_file_name = '{}_dist_cost_vs_rsrc_0820_0830_1secs_p0.3.pcl' .format (city)
    # my_res_file_parser.print_cost_vs_rsrc (res_input_file_names=[res_input_file_name])
    # pcl_input_file_name='{}_dist_cost_vs_rsrc_0820_0830_1secs_p0.3.pcl' .format (city)
    # pcl_input_file_name = my_res_file_parser.calc_cost_vs_rsrc (pcl_input_file_name = pcl_input_file_name, res_input_file_names=[res_input_file_name])
    # my_res_file_parser.gen_cost_vs_rsrc_tbl (city=city, normalize_Y=True, dist=True, pcl_input_file_name=pcl_input_file_name)
    # my_res_file_parser = Res_file_parser ()
    # my_res_file_parser.plot_cost_vs_rsrc (normalize_X=True, slot_len_in_sec=float(input_file_name.split('sec')[0].split('_')[-1]), X_norm_factor=X_norm_factor)

    
    # pcl_file_name = my_res_file_parser.calc_mig_cost_vs_rsrc (pcl_input_file_name=None, res_input_file_names=['Monaco_0820_0830_1secs_Telecom_p0.3_opt.res'])
    # , 'Monaco_0820_0830_1secs_SyncPartResh.res', 'Monaco_0820_0830_1secs_Async.res'])   
    # my_res_file_parser.gen_mig_vs_rsrc_tbl (city, pcl_file_name)

    # city = 'Monaco'
    # my_res_file_parser = Res_file_parser ()
    # pcl_input_file_name = '{}_RtProb_0820_0830_1secs.pcl' .format (city)
    # my_res_file_parser.dump_self_list_of_dicts_to_pcl (pcl_input_file_name=pcl_input_file_name, res_file_name='Monaco_Rt_Prob_1secs.res') 
    # my_res_file_parser.plot_RT_prob_sim_python (pcl_input_file_name=pcl_input_file_name, dist=True)
    # plot_crit_n_mig_vs_T (city=city, y_axis='mig_cost', per_slot=False)
    
    # my_res_file_parser = Res_file_parser ()
    # ar=np.array ([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 110])
    # print ('the conf interval is ', my_res_file_parser.conf_interval (ar, avg=np.average(ar)))
        
    # my_res_file_parser.plot_RT_prob_sim_python ('RT_prob_sim_Monaco.Telecom.antloc_192cells.poa2cell_Monaco_0820_0830_1secs_Telecom.poa.res')
    # my_res_file_parser.plot_RT_prob_sim_python ('RT_prob_sim_Monaco.Telecom.antloc_192cells.poa2cell_Monaco_0820_0830_1secs_Telecom.poa.res')
    # my_res_file_parser.city = 'Lux'
    # pcl_file_name = my_res_file_parser.calc_mig_cost_vs_rsrc (pcl_input_file_name=None, res_input_file_names=['Monaco_0730_0830_1secs_Telecom_SyncPartResh_mig_vs_rsrc.res'])   
    # pcl_file_name = my_res_file_parser.calc_mig_cost_vs_rsrc(pcl_input_file_name=None, res_input_file_names=['Monaco_0820_0830_1secs_Telecom_p0.3_ourAlg.res'])   
    # my_res_file_parser.gen_mig_vs_rsrc_tbl (city)
    # city = 'Monaco'
    # for city in ['Lux', 'Monaco']:
    #     my_res_file_parser = Res_file_parser ()
    #     my_res_file_parser.plot_crit_len (city=city)
    
    # city='Monaco'
    # reshuffle=True
    # plot_mig_vs_rsrc (city=city)
    # plot_num_crit_n_mig_vs_num_vehs (city=city, reshuffle=reshuffle)
    # exit ()
        
    
    
    # my_res_file_parser.plot_cost_vs_rsrc (normalize_X=True, slot_len_in_sec=float(input_file_name.split('sec')[0].split('_')[-1]), X_norm_factor=X_norm_factor)

    
    # my_res_file_parser.plot_tot_num_of_vehs_per_slot (['Monaco_0730_0830_1secs_cnt.pcl', 'Lux_0730_0830_1secs_cnt.pcl'])
    # pcl_output_file_name = my_res_file_parser.calc_mig_cost_vs_rsrc(res_input_file_names=['Lux_0730_0830_1secs_post_p0.3_ourAlg.res'] if city=='Lux' else ['Monaco_0730_0830_1secs_Telecom_p0.3_ourAlg.res']) 

    # plot_crit_n_mig_vs_T (city=city, y_axis='mig_cost', per_slot=True)
    # city = 'Monaco'
    # my_res_file_parser = Res_file_parser ()
    # my_res_file_parser.print_cost_vs_rsrc (res_input_file_names=['Monaco_0820_0830_1secs_Telecom_p0.3_optInt_sdG.res', 'Monaco_0820_0830_1secs_Telecom_p0.3_opt_sdG.res', 'Monaco_0820_0830_1secs_Telecom_p0.3_opt.res'])
    # my_res_file_parser.calc_cost_vs_rsrc (pcl_input_file_name='cost_vs_rsrc_Lux_0820_0830_1secs_post_p0.3.pcl', res_input_file_names=['Lux_0820_0830_1secs_post_p0.3_opt.res'])
    # my_res_file_parser.gen_cost_vs_rsrc_tbl (city=city, normalize_Y=False)

