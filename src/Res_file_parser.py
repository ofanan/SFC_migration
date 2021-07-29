import matplotlib.pyplot as plt
import numpy as np
import math

from printf import printf 

t_idx         = 0
alg_idx       = 1
cpu_idx       = 2
prob_idx      = 3
stts_idx      = 4
num_of_fields = stts_idx+1

opt_idx   = 0
alg_idx   = 1
ffit_idx  = 2
cpvnf_idx = 3

class Res_file_parser (object):  

    def __init__ (self):
        """
        """
        self.add_plot_str1    = '\t\t\\addplot [color = blue, mark=square, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_opt     = '\t\t\\addplot [color = green, mark=+, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_ourAlg = '\t\t\\addplot [color = purple, mark=o, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_ourAlgShortPushUp = '\t\t\\addplot [color = red, mark=o, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_ffit    = '\t\t\\addplot [color = blue, mark=triangle*, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_cpvnf   = '\t\t\\addplot [color = black, mark = square,      mark options = {mark size = 2, fill = black}, line width = \plotLineWidth] coordinates {\n\t\t'
        self.end_add_plot_str = '\n\t\t};'
        self.add_legend_str = '\n\t\t\\addlegendentry {'
        # self.add_plot_str_vec = [self.add_plot_opt, self.add_plot_alg, self.add_plot_ffit, self.add_plot_cpvnf]

        self.add_plot_str_dict = {'opt'     : self.add_plot_opt,
                                  'ourAlg' : self.add_plot_ourAlg,
                                  'ourAlgShortPushUp' : self.add_plot_ourAlgShortPushUp,
                                  'ffit'    : self.add_plot_ffit,
                                  'cpvnf'   : self.add_plot_cpvnf}

        self.legend_entry_dict = {'opt'     :  '\opt', 
                                  'ourAlg' : '\\algtop', 
                                  'ffit'    : '\\ffit',
                                  'cpvnf'   : '\cpvnf'}
        
    def parse_file (self, input_file_name):
        
        self.input_file_name = input_file_name
        self.input_file     = open ("../res/" + input_file_name,  "r")
        lines               = (line.rstrip() for line in self.input_file) # "lines" contains all lines in input file
        lines               = (line for line in lines if line)       # Discard blank lines
        self.list_of_dicts  = [] # a list of dictionaries, holding the settings and the results read from result files
        
        for line in lines:
        
            # Discard lines with comments / verbose data
            if (line.split ("//")[0] == ""):
                continue
           
            self.parse_line(line)
            # if (not(self.dict in self.list_of_dicts)): # verify that such an item doesn't already exist in the list. However, if using list_of_dicts, no need for this check 
            if ( not(self.dict in self.list_of_dicts)):
                self.list_of_dicts.append(self.dict)
                

        self.input_file.close

    def parse_line (self, line):

        splitted_line = line.split (" | ")
         
        settings          = splitted_line[0]
        cost              = float(splitted_line[1].split("=")[1])
        splitted_settings = settings.split ("_")

        if len (splitted_settings) != num_of_fields:
            print ("encountered a format error.\nSplitted line={}\nsplitted settings={}" .format (splitted_line, splitted_settings))
            exit ()
               
        self.dict = {
            "t"     : int   (splitted_settings [t_idx]   .split('t')[1]),
            "alg"   : splitted_settings      [alg_idx],
            "cpu"   : int   (splitted_settings [cpu_idx] .split("cpu")[1]),  
            "prob"  : float (splitted_settings [prob_idx].split("p")   [1]),  
            "stts"  : int   (splitted_settings [stts_idx].split("stts")[1]),  
            "cost"  : float (splitted_line[1].split(" = ")[1])
            }

    def gen_filtered_list (self, list_to_filter, min_t = -1, max_t = float('inf'), prob=0, alg = None, cpu = -1, stts = -1):
        """
        filters and takes from all the items in a given list (that was read from the res file) only those with the desired parameters value
        The function filters by some parameter only if this parameter is given an input value > 0.
        """
        list_to_filter = list (filter (lambda item : item['t'] >= min_t and item['t'] <= max_t, list_to_filter))    
        if (alg != None):
            list_to_filter = list (filter (lambda item : item['alg']  == alg, list_to_filter))    
        if (cpu != -1):
            list_to_filter = list (filter (lambda item : item['cpu']  == cpu, list_to_filter))    
        if (prob > 0):
            list_to_filter = list (filter (lambda item : item['prob'] == prob, list_to_filter))    
        if (stts != -1):
            list_to_filter = list (filter (lambda item : item['stts'] == stts, list_to_filter))    
        return list_to_filter

    def print_single_tikz_plot (self, list_of_dict, key_to_sort, addplot_str = None, add_legend_str = None, legend_entry = None):
        """
        Prints a single plot in a tikz format.
        Inputs:
        The "x" value is the one which the user asks to sort the inputs (e.g., "x" value may be the cache size, uInterval, etc).
        The "y" value is the cost for this "x". 
        list_of_dicts - a list of Python dictionaries. 
        key_to_sort - the function sorts the items by this key, e.g.: cache size, uInterval, etc.
        addplot_str - the "add plot" string to be added before each list of points (defining the plot's width, color, etc.).
        addlegend_str - the "add legend" string to be added after each list of points.
        legend_entry - the entry to be written (e.g., 'Opt', 'Alg', etc).
        """
        if (not (addplot_str == None)):
            printf (self.output_file, addplot_str)
        for dict in sorted (list_of_dict, key = lambda i: i[key_to_sort]):
            printf (self.output_file, '({:.4f}, {:.4f})' .format (dict[key_to_sort], dict['cost']))
        printf (self.output_file, self.end_add_plot_str)
        if (not (add_legend_str == None)): # if the caller requested to print an "add legend" str          
            printf (self.output_file, '\t\t{}{}' .format (self.add_legend_str, legend_entry))    
            printf (self.output_file, '}\n')    
        printf (self.output_file, '\n\n')    


    def plot_cost_vs_rsrcs (self, normalize_X = True, normalize_Y = False, slot_len_in_sec=1):
        min_t, max_t = 30541, 30600
        prob = 0.3
        Y_units_factor = 1 # a factor added for showing the cost, e.g., in units of K (thousands)
        self.output_file_name = '../res/{}.dat' .format (self.input_file_name, prob)
        self.output_file      = open (self.output_file_name, "w")
        
        if (normalize_X):
            opt_list = sorted (self.gen_filtered_list (self.list_of_dicts, alg='opt', prob=prob, min_t=min_t, max_t=max_t, stts=1),
                               key = lambda item : item['cpu'])
            # cpu_vals = sorted (list (set([item['cpu'] for item in opt_list])))
            X_norm_factor = cpu_vals[0] # normalize X axis by the minimum cpu
            
            opt_avg_list = []
            for cpu in cpu_vals:
                opt_avg_list.append (np.average ([item['cost'] for item in 
                                     list (filter (lambda item : item['cpu']==cpu, opt_list) )]))
        else:
            X_norm_factor = 1
            alg_list = sorted (self.gen_filtered_list (self.list_of_dicts, alg='ourAlg', prob=prob, min_t=min_t, max_t=max_t, stts=1),
                               key = lambda item : item['cpu'])
            # cpu_vals = sorted (list (set([item['cpu'] for item in alg_list])))
        
        Y_norm_factor = opt_avg_list[-1] if normalize_Y else 1 # normalize Y axis by the maximum cost

        for alg in ['ourAlg', 'ffit', 'cpvnf']: #['opt', 'ourAlg', 'ffit', 'cpvnf']:
            
            alg_list = sorted (self.gen_filtered_list (self.list_of_dicts, alg=alg, min_t=min_t, max_t=max_t, stts=1),
                           key = lambda item : item['cpu'])
            
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
                                      'cost' : np.average ([item['cost'] for item in alg_vals_for_this_cpu])* Y_units_factor / Y_norm_factor })

                if (len(alg_avg_list)==0):
                    continue
        
            self.print_single_tikz_plot (alg_avg_list, key_to_sort='cpu', addplot_str=self.add_plot_str_dict[alg], add_legend_str=self.add_legend_str, legend_entry=self.legend_entry_dict[alg]) 

    def plot_num_of_vehs (self):
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
     
if __name__ == '__main__':
    my_res_file_parser = Res_file_parser ()
    input_file_name = '0829_0830_1secs_256aps_p0.3.res'
    my_res_file_parser.parse_file (input_file_name) # ('shorter.res')
    my_res_file_parser.plot_cost_vs_rsrcs (normalize_X=False, slot_len_in_sec=float(input_file_name.split('sec')[0].split('_')[-1]))        
    # my_res_file_parser.compare_algs()  
    