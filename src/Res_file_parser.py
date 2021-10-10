import matplotlib.pyplot as plt
import numpy as np
import math

from printf import printf 

# Indices of fields indicating the settings in a standard ".res" file
t_idx         = 0
alg_idx       = 1
cpu_idx       = 2
prob_idx      = 3
seed_idx      = 4
stts_idx      = 5
num_of_fields = stts_idx+1

opt_idx   = 0
alg_idx   = 1
ffit_idx  = 2
cpvnf_idx = 3

class Res_file_parser (object):
    """
    Parse "res" (result) files, and generate plots from them.
    """

    # An inline function. Calculates the total cost at a given time slot.
    # The total cost is the sum of the migration, CPU and link costs.
    # If the length of the slot is 8, we need to multiply the CPU and link cost by 7.5. This is because in 1 minutes (60 seconds), where we ignore the first we have only 7.5 8-seconds solots #$$$ ????        
    calc_cost_of_item = lambda self, item : item['mig_cost'] + (item['cpu_cost'] + item['link_cost']) * (7.5 if self.time_slot_len == 8 else 1)    

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
                                  'ffit'   : '\\ffit',
                                  'cpvnf'  : '\cpvnf'}

        self.color_dict       = {'opt'    : 'green',
                                'ourAlg'  : 'purple',
                                'ffit'    : 'blue',
                                'cpvnf'   : 'black'}
        
        self.markers_dict     = {'opt'    : '+',
                                'ourAlg'  : 'o',
                                'ffit'    : '^',
                                'cpvnf'   : 's'}

        self.marker_size = 5
        
    def parse_detailed_cost_comp_file (self, input_file_name):
        """
        Parse a result file containing the detailed costs, with its components (link, cpu and mig' cost) for each ran algorithm. 
        """
        self.input_file_name = input_file_name
        self.input_file      = open ("../res/" + input_file_name,  "r")

        lines                = (line.rstrip() for line in self.input_file) # "lines" contains all lines in input file
        lines                = (line for line in lines if line)       # Discard blank lines
        
        for line in lines:
            
            # Discard lines with comments / verbose data
            if (line.split ("//")[0] == ""):
                continue
           
            splitted_line = line.split ("=")
            if (splitted_line[0]=='cpu_cost_in_slot'):
                cpu_cost_in_slot = np.array (self.parse_vec_line(splitted_line[1]))
            elif (splitted_line[0]=='link_cost_in_slot'):
                link_cost_in_slot = np.array (self.parse_vec_line(splitted_line[1]))
            elif (splitted_line[0]=='num_of_migs_in_slot'):            
                num_of_migs_in_slot = np.array (self.parse_vec_line(splitted_line[1]))
            elif (splitted_line[0]=='num_of_critical_usrs_in_slot'):
                num_of_critical_usrs_in_slot = np.array (self.parse_vec_line(splitted_line[1]))
            elif (splitted_line[0]=='num_of_usrs_in_slot'):
                num_of_usrs_in_slot = np.array (self.parse_vec_line(splitted_line[1]))
        
        self.sim_len              = 3600
        self.period_len           = 400
        self.cost_of_single_chain = 600
        self.num_of_periods    = int (self.sim_len / self.period_len)
        self.time_slot_len     = int(self.input_file_name.split('secs')[0].split('_')[-1])
        self.num_of_slots_in_period = int (self.period_len / self.time_slot_len)
        
        cpu_cost_in_period             = self.gen_vec_for_period(cpu_cost_in_slot) * self.time_slot_len 
        link_cost_in_period            = self.gen_vec_for_period(link_cost_in_slot) * self.time_slot_len
        num_of_migs_in_period          = self.gen_vec_for_period(num_of_migs_in_slot) * self.cost_of_single_chain
        num_of_critical_usrs_in_period = self.gen_vec_for_period(num_of_critical_usrs_in_slot)
        num_of_usrs_in_period          = self.gen_vec_for_period(num_of_usrs_in_slot)
        num_of_critical_usrs_per_slot  = num_of_critical_usrs_in_period / self.num_of_slots_in_period
        num_of_usrs_per_slot            = num_of_usrs_in_period / self.num_of_slots_in_period
        
        output_file_name = self.input_file_name + '.dat' 
        self.output_file = open ('../res/{}' .format (output_file_name), 'w')
        
        x = [self.period_len * (i+1) for i in range (len(cpu_cost_in_period))]
        
        printf (self.output_file, self.add_plot_cpu_cost)
        for i in range (len(cpu_cost_in_period)):
            printf (self.output_file, '({:.2f}, {:.2f})' .format (x[i], cpu_cost_in_period[i]))
        printf (self.output_file, '};' + self.add_legend_str + 'cpu}\n')

        printf (self.output_file, self.add_plot_link_cost)
        for i in range (len(cpu_cost_in_period)):
            printf (self.output_file, '({:.2f}, {:.2f})' .format (x[i], link_cost_in_period[i]))
        printf (self.output_file, '};' + self.add_legend_str + 'link}\n')

        printf (self.output_file, self.add_plot_mig_cost)
        for i in range (len(cpu_cost_in_period)):
            printf (self.output_file, '({:.2f}, {:.2f})' .format (x[i], num_of_migs_in_period[i]))
        printf (self.output_file, '};' + self.add_legend_str + 'mig.}\n')

        printf (self.output_file, self.add_plot_num_of_critical_chains)
        for i in range (len(cpu_cost_in_period)):
            printf (self.output_file, '({:.2f}, {:.2f})' .format (x[i], num_of_critical_usrs_per_slot[i]))
        printf (self.output_file, '};' + self.add_legend_str + '\# of critical chains per slot}\n')

        printf (self.output_file, self.add_plot_num_of_critical_chains)
        for i in range (len(cpu_cost_in_period)):
            printf (self.output_file, '({:.4f}, {:.4f})' .format (x[i], num_of_critical_usrs_per_slot[i]/num_of_usrs_per_slot[i]))
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

        
    def parse_file (self, input_file_name):
        """
        Parse a result file, in which each un-ommented line indicates a concrete simulation settings.
        """
        
        self.input_file_name = input_file_name
        self.input_file      = open ("../res/" + input_file_name,  "r")
        lines                = (line.rstrip() for line in self.input_file) # "lines" contains all lines in input file
        lines                = (line for line in lines if line)       # Discard blank lines
        self.list_of_dicts   = [] # a list of dictionaries, holding the settings and the results read from result files
        
        for line in lines:
        
            # Discard lines with comments / verbose data
            if (line.split ("//")[0] == ""):
                continue
           
            self.parse_line(line)
            if ( not(self.dict in self.list_of_dicts)):
                self.list_of_dicts.append(self.dict)
                

        self.input_file.close

    def parse_line (self, line):
        """
        Parse a line in a result file. Such a line should begin with a string having several fields, detailing the settings.
        """

        splitted_line = line.split (" | ")
         
        settings          = splitted_line[0]
        cost              = float(splitted_line[1].split("=")[1])
        splitted_settings = settings.split ("_")

        if len (splitted_settings) != num_of_fields:
            print ("encountered a format error.\nSplitted line={}\nsplitted settings={}" .format (splitted_line, splitted_settings))
            exit ()
               
        self.dict = {
            "t"         : int   (splitted_settings [t_idx]   .split('t')[1]),
            "alg"       : splitted_settings      [alg_idx],
            "cpu"       : int   (splitted_settings [cpu_idx] .split("cpu")[1]),  
            "prob"      : float (splitted_settings [prob_idx].split("p")   [1]),  
            "seed"      : int   (splitted_settings [seed_idx].split("sd")  [1]),  
            "stts"      : int   (splitted_settings [stts_idx].split("stts")[1]),  
            "cpu_cost"  : float (splitted_line[1].split("=")[1]),
            "link_cost" : float (splitted_line[2].split("=")[1]),
            "mig_cost"  : float (splitted_line[3].split("=")[1]),            
            "cost"      : float (splitted_line[4].split("=")[1])
            }

    def gen_filtered_list (self, list_to_filter, min_t = -1, max_t = float('inf'), prob=None, alg = None, cpu = None, stts = -1):
        """
        filters and takes from all the items in a given list (that was read from the res file) only those with the desired parameters value
        The function filters by some parameter only if this parameter is given an input value > 0.
        """
        list_to_filter = list (filter (lambda item : item['t'] >= min_t and item['t'] <= max_t, list_to_filter))    
        if (alg != None):
            list_to_filter = list (filter (lambda item : item['alg']  == alg, list_to_filter))    
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


    def plot_cost_vs_rsrcs (self, normalize_X = True, normalize_Y = False, slot_len_in_sec=1, X_norm_factor=1):
        """
        Plot the cost as a function of the amount of resources (actually, cpu capacity at leaf).
        Possibly normalize the amounts of cpu (the X axis) by either the min' amount of cpu required by opt (LBound) to obtain a feasible sol; 
        and/or normalize the cost (the Y axis) by the costs obtained by opt.   
        """
        
        max_t = 30600
        self.time_slot_len = int(self.input_file_name.split('secs')[0].split('_')[-1])
        min_t = 30545 if (self.time_slot_len==1) else 30541
        prob = 0.3
        Y_units_factor = 1 # a factor added for showing the cost, e.g., in units of K (thousands)
        self.output_file_name = '../res/{}.dat' .format (self.input_file_name, prob)
        self.output_file      = open (self.output_file_name, "w")
        
        # if (normalize_X):
        #     opt_list = sorted (self.gen_filtered_list (self.list_of_dicts, alg='opt', prob=prob, min_t=min_t, max_t=max_t, stts=1),
        #                        key = lambda item : item['cpu'])
        #     cpu_vals = sorted (list (set([item['cpu'] for item in opt_list])))
        #     X_norm_factor = cpu_vals[0] # normalize X axis by the minimum cpu
        #
        #     if (normalize_Y):
        #         opt_avg_list = []
        #         for cpu in cpu_vals:
        #             opt_avg_list.append (np.average ([item['cost'] for item in list (filter (lambda item : item['cpu']==cpu, opt_list) )]))
        # else:
        #     X_norm_factor = 1
        #     alg_list = sorted (self.gen_filtered_list (self.list_of_dicts, alg='ourAlg', prob=prob, min_t=min_t, max_t=max_t, stts=1),
        #                        key = lambda item : item['cpu'])
        
        # Y_norm_factor = opt_avg_list[-1] if normalize_Y else 1 # Calculate the normalization factor of the Y axis
        
        Y_norm_factor = 1 #$$$
        
        for alg in ['ourAlg', 'ffit', 'cpvnf', 'opt']: #['opt', 'ourAlg', 'ffit', 'cpvnf']:
            
            alg_list = sorted (self.gen_filtered_list (self.list_of_dicts, alg=alg, min_t=min_t, max_t=max_t, stts=1), key = lambda item : item['cpu'])
            
            cpu_vals = set ([item['cpu'] for item in self.list_of_dicts if item in alg_list])
            
            if (len(alg_list)==0):
                continue

            alg_avg_list = []
            
            for cpu in cpu_vals:
                
                alg_vals_for_this_cpu = list (filter (lambda item : item['cpu']==cpu, alg_list) )
                
                if (len(alg_vals_for_this_cpu)< math.floor( (max_t - min_t)/slot_len_in_sec)):
                    print ('Warning: there are too few samples\nalg={}, cpu={}, num of smpls={}' .format(alg, cpu, len(alg_vals_for_this_cpu)))
                    continue
                
                alg_avg_list.append ({'cpu'  : (cpu / X_norm_factor) if normalize_X else (cpu / X_norm_factor), 
                                      'cost' : np.average ([self.calc_cost_of_item(item) for item in alg_vals_for_this_cpu])* Y_units_factor / Y_norm_factor })

                if (len(alg_avg_list)==0):
                    continue
        
            self.print_single_tikz_plot (alg_avg_list, key_to_sort='cpu', addplot_str=self.add_plot_str_dict[alg], add_legend_str=self.add_legend_str, legend_entry=self.legend_entry_dict[alg]) 

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
        for alg in ['ourAlg', 'ffit', 'cpvnf', 'opt']: #['opt', 'ourAlg', 'ffit', 'cpvnf']:
            
            list_of_points = self.gen_filtered_list (self.list_of_dicts, alg=alg, stts=1)
        
            for point in list_of_points:
                point['cpu'] /= 10
            
            self.print_single_tikz_plot (list_of_points, key_to_sort='prob', addplot_str=self.add_plot_str_dict[alg], add_legend_str=self.add_legend_str, legend_entry=self.legend_entry_dict[alg], y_value='cpu')
     
    def conf_interval (self, vec):
        """
        Input: a vector
        Output: [y_low, y_min], that are the lower and lowest values of the 95%-confidence interval for this vec
        """ 

        avg = np.average (vec)
        std = np.std(vec)
        return [avg - 2*std, avg + 2*std]
       
    def plot_RT_prob_sim_python (self):
        """
        Generating a python plot showing the amount of resource augmentation required, as a function of the probability that a user has tight (RT) delay requirements.
        Show the conf' intervals.
        """
        
        output_file_name = self.input_file_name + '.dat' 
        self.output_file = open ('../res/{}' .format (output_file_name), 'w')

        
        for alg in ['ourAlg', 'ffit', 'cpvnf', 'opt']: #['opt', 'ourAlg', 'ffit', 'cpvnf']:
            
            list_of_points = self.gen_filtered_list(self.list_of_dicts, alg=alg, stts=1) 
        
            x = set () # The x value will hold all the probabilities that appear in the .res file
            for point in list_of_points: # A cpu cap' unit represents 100 MHz --> to represent results by units of GHz, divide the cpu cap' by 10.
                point['cpu'] /= 10
                x.add (point['prob'])
            
            x = sorted (x)
            
            for x_val in x: # for each concrete value in the x vector
                [y_lo, y_hi] = self.conf_interval ([item['cpu'] for item in self.gen_filtered_list(list_of_points, prob=x_val)])

                plt.plot (x_val, (y_lo+y_hi)/2, color=self.color_dict[alg], marker=self.markers_dict[alg], markersize=self.marker_size)
                plt.plot ((x_val,x_val), (y_lo, y_hi), color=self.color_dict[alg]) # Plot the confidence interval
            plt.show ()
            
if __name__ == '__main__':
    
    my_res_file_parser = Res_file_parser ()
    
    input_file_name = 'RT_prob_sim_Lux.center.post.antloc_256cells.ap2cell_0829_0830_1secs_256aps.ap.res' 
    my_res_file_parser.parse_file (input_file_name) 
    # my_res_file_parser.plot_RT_prob_sim()
    # my_res_file_parser.parse_detailed_cost_comp_file(input_file_name)
    my_res_file_parser.plot_RT_prob_sim_python()
    
    # # X_norm_factor values: Lux post: 160. Lux rect: 208 
    # X_norm_factor = 160 
    # input_file_name = '0829_0830_1secs_256aps_p0.3.res.expCPU_POST.res' #'RT_prob_sim_Lux.center.post.antloc_256cells.ap2cell_0829_0830_1secs_256aps.ap_deter_usr_id.res'
    # my_res_file_parser.plot_cost_vs_rsrcs (normalize_X=True, slot_len_in_sec=float(input_file_name.split('sec')[0].split('_')[-1]), X_norm_factor=X_norm_factor)        
    
