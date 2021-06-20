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

    def gen_num_of_behs_plot (self):
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
     
        
        