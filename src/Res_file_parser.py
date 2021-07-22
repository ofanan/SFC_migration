import matplotlib.pyplot as plt
import numpy as np

from printf import printf 

t_idx         = 0
alg_idx       = 1
cpu_idx       = 2
stts_idx      = 3
num_of_fields = 3

class Res_file_parser (object):  

    def __init__ (self):
        """
        """
        self.add_plot_opt   = '\t\t\\addplot [color = green, mark=+, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_str1  = '\t\t\\addplot [color = blue, mark=square, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_fno1  = '\t\t\\addplot [color = purple, mark=o, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_fna1  = '\t\t\\addplot [color = red, mark=triangle*, line width = \\plotLineWidth] coordinates {\n\t\t'
        self.add_plot_fno2  = '\t\t\\addplot [color = black, mark = square,      mark options = {mark size = 2, fill = black}, line width = \plotLineWidth] coordinates {\n\t\t'
        self.add_plot_fna2  = '\t\t\\addplot [color = blue,  mark = *, mark options = {mark size = 2, fill = blue},  line width = \plotLineWidth] coordinates {\n\t\t'
        self.end_add_plot_str = '\n\t\t};'
        self.add_legend_str = '\n\t\t\\addlegendentry {'
        self.add_plot_str_dict = {'Opt' : self.add_plot_opt, 'FNAA' : self.add_plot_fna2, 'FNOA' : self.add_plot_fno2}
        self.legend_entry_dict = {'Opt' : '\\opt', 
                                  'FNOA' : '\\pgmfno'}

    def parse_file (self, input_file_name):
    
        self.input_file     = open ("../res/" + input_file_name,  "r")
        self.output_file    = open ("../res/" + input_file_name.split(".")[0] + ".dat", "w")
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
        splitted_settings = settings.split (".")

        if len (splitted_settings) < num_of_fields:
            print ("encountered a format error. Splitted line is {}" .format (splitted_line))
            exit ()
               
        self.dict = {
            "t"     : int (splitted_settings [t_idx]   .split('t')[1]),
            "alg"   : splitted_settings      [alg_idx],
            "cpu"   : int (splitted_settings [cpu_idx] .split("cpu")[1]),  
            "stts"  : int (splitted_settings [stts_idx].split("stts")[1]),  
            "cost"  : float(splitted_line[1].split(" = ")[1])
            }

    def gen_filtered_list (self, list_to_filter, t = -1, alg = None, cpu = -1, stts = -1):
        """
        filters and takes from all the items in a given list (that was read from the res file) only those with the desired parameters value
        The function filters by some parameter only if this parameter is given an input value > 0.
        """
        if (t != -1):
            list_to_filter = list (filter (lambda item : item['t'] == t, list_to_filter))    
        if (alg != None):
            list_to_filter = list (filter (lambda item : item['alg'] == alg, list_to_filter))    
        if (cpu != -1):
            list_to_filter = list (filter (lambda item : item['cpu'] == cpu, list_to_filter))    
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
            printf (self.output_file, '({:.0f}, {:.0f})' .format (dict[key_to_sort], dict['cost']))
        printf (self.output_file, self.end_add_plot_str)
        if (not (add_legend_str == None)): # if the caller requested to print an "add legend" str          
            printf (self.output_file, '\t\t{}{}' .format (self.add_legend_str, legend_entry))    
            printf (self.output_file, '}\n')    
        printf (self.output_file, '\n\n')    


    def plot_cost_vs_rsrcs_normalized (self):
        opt_list     = self.gen_filtered_list (self.list_of_dicts, alg='lp',      t=30601, stts=1)
        norm_factor  = opt_list[-1]['cost']
        our_alg_list = self.gen_filtered_list (self.list_of_dicts, alg='our_alg', t=30601, stts=1)
        
        # for alg in  
        # self.print_single_tikz_plot ([opt_list[i] / opt_list[0] for i in range (len (opt_list))], key_to_sort='cpu')
        #
        #
        # self.print_single_tikz_plot(our_alg_list, key_to_sort='cpu')

    def compare_algs (self):
        # lp_list_of_dicts  = sorted (list (filter (lambda item : item['alg'] == 'lp', self.list_of_dicts)), key = lambda item : item['t'])
        # alg_list_of_dicts = sorted (list (filter (lambda item : item['alg'] == 'lp', self.list_of_dicts)), key = lambda item : item['t'])
        opt_cost  = np.array ([item['cost'] for item in sorted (list (filter (lambda item : item['alg'] == 'lp',  self.list_of_dicts)), key = lambda item : item['t'])] )
        alg_cost = np.array ([item['cost'] for item in sorted (list (filter (lambda item : item['alg'] == 'our_alg', self.list_of_dicts)), key = lambda item : item['t'])])
        ratio     = np.divide (alg_cost, opt_cost)
        print ('max_ratio = {}, avg ratio = {}' .format (np.max (ratio), np.average(ratio)))
        print (ratio)

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
    my_res_file_parser.parse_file ('vehicles_n_speed_0830_0831.res') # ('shorter.res')
    my_res_file_parser.plot_cost_vs_rsrcs_normalized ()        
    # my_res_file_parser.compare_algs()  
    