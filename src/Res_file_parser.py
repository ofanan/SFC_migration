import matplotlib.pyplot as plt
import numpy as np

from printf import printf 

t_idx   = 0
alg_idx = 1
stts_idx = 2

class Res_file_parser (object):  

    def __init__ (self):
        print ('')

    def parse_line (self, line):
        splitted_line = line.split ("|")
         
        settings        = splitted_line[0]
        cost            = float(splitted_line[1].split(" = ")[1])
        splitted_line   = settings.split (".")

        if len (splitted_line) < num_of_fields:
            print ("encountered a format error. Splitted line is {}" .format (splitted_line))
            return False
        self.dict = {
            "t"          : int (splitted_line   [t_idx].split('t')[1]),
            "alg"        : splitted_line        [alg_idx],   
            }

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
     
    def plot_num_of_vehs_per_ap (self):
        # Open input and output files
        input_file  = open ("../res/num_of_vehs_per_ap.ap", "r")  
        
        num_of_vehs_per_ap_per_t = []
        for line in input_file:
            
            if (line == "\n" or line.split ("//")[0] == ""):
                continue
        
            num_of_vehs_in_cur_ap = []
            line = line.split ("\n")[0]
            splitted_line = line.split (":")
            # ap_num = splitted_line[0].split("_")[-1]
            splitted_line = splitted_line[1].split('[')[1].split(']')[0].split(' ')
            for cur_num_of_vehs_in_this_ap in splitted_line:
                num_of_vehs_in_cur_ap.append (int(cur_num_of_vehs_in_this_ap))
            
            num_of_vehs_per_ap_per_t.append (num_of_vehs_in_cur_ap)            
                
        for plot_num in range (16):
            for ap in range (4*plot_num, 4*(plot_num+1)):
                plt.title ('Number of vehicles in each cell')
                plt.plot (range(len(num_of_vehs_per_ap_per_t[ap])), num_of_vehs_per_ap_per_t[ap], label='cell {}' .format(ap))
                plt.ylabel ('Number of vehicles')
            plt.legend()
            plt.savefig ('../res/num_of_vehs_per_cell_plot{}.jpg' .format(plot_num))
            plt.clf()
        # plt.plot (np.array(t)/3600, tot_num_of_vehs)
        # plt.show ()    

if __name__ == '__main__':
    my_res_file_parser = Res_file_parser ()
    my_res_file_parser.plot_num_of_vehs_per_ap ()
            
    