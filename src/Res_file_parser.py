import matplotlib.pyplot as plt
import numpy as np

from printf import printf 

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
# plt.plot (t, num_of_act_vehs)
plt.show ()
 
    
    