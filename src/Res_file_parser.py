import matplotlib.pyplot as plt
import numpy as np

from printf import printf 

t_idx         =  0
solver_idx    =  1
stts_idx      = -1
num_of_fields = 2

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
        self.list_of_dicts  = []
        
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

        if len (splitted_line) < num_of_fields:
            print ("encountered a format error. Splitted line is {}" .format (splitted_line))
            return False
               
        self.dict = {
            "t"          : int (splitted_settings [t_idx].split('t')[1]),
            "solver"     : splitted_settings      [solver_idx],
            "cost"       : float(splitted_line[1].split(" = ")[1])
            }

    def gen_filtered_list (self, list_to_filter, alg_mode = None):
        """
        filters and takes from all the items in a given list (that was read from the res file) only those with the desired parameters value
        The function filters by some parameter only if this parameter is given an input value > 0.
        """
        if (not (alg_mode == None)):
            list_to_filter = list (filter (lambda item : item['alg_mode'] == alg_mode, list_to_filter))    
        return list_to_filter

    def compare_algs (self):
        # lp_list_of_dicts  = sorted (list (filter (lambda item : item['alg'] == 'lp', self.list_of_dicts)), key = lambda item : item['t'])
        # alg_list_of_dicts = sorted (list (filter (lambda item : item['alg'] == 'lp', self.list_of_dicts)), key = lambda item : item['t'])
        lp_cost  = np.array ([item['cost'] for item in sorted (list (filter (lambda item : item['solver'] == 'lp',  self.list_of_dicts)), key = lambda item : item['t'])] )
        alg_cost = np.array ([item['cost'] for item in sorted (list (filter (lambda item : item['solver'] == 'alg', self.list_of_dicts)), key = lambda item : item['t'])])
        ratio     = np.divide (alg_cost, lp_cost)
        print (ratio)
        print ('max_ratio = {}' .format (np.max (ratio)))

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
    my_res_file_parser.parse_file ('vehicles_n_speed_0730.res') # ('shorter.res')  
    my_res_file_parser.compare_algs ()        
    