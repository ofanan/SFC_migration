import numpy as np
import math
import itertools 
import time
import matplotlib.pyplot as plt

from usr_c import usr_c # class of the users
from printf import printf

VERBOSE_AP         = 1
VERBOSE_CNT        = 2
VERBOSE_AP_AND_CNT = 3

class loc2ap_c (object):
    """
    Accept as input a .loc file (a file detailing the locations of all the new users / users who moved at each slot).
    Output a .ap file (a file detailing the Access Points of all the new users / users who moved at each slot).
    """

    
    # inline function for the files location-to-AP mapping
    # currently this function always maps assuming square cells.
    loc2ap = lambda self, x, y : self.loc2ap_using_sq_cells (x, y)
    
    # inline function for formatted-printing the AP of a single user
    print_usr_ap = lambda self, usr: printf(self.ap_file, "({},{})" .format (usr['id'], usr['nxt ap']))   

    def __init__(self, use_sq_cells = True, max_power_of_4=3, verbose = VERBOSE_AP):

        self.max_x, self.max_y = 12000, 12000 # size of the square cell of each AP, in meters.
        self.verbose           = verbose 
        self.usrs              = []
        self.use_sq_cells      = use_sq_cells
        if (self.use_sq_cells):
            self.max_power_of_4    = max_power_of_4
            self.num_of_APs        = 4**max_power_of_4
        if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
            self.num_of_vehs_in_ap = np.empty (4**self.max_power_of_4, dtype = 'object')
            for ap in range(self.num_of_APs): 
                self.num_of_vehs_in_ap[ap] = []
        if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
            self.num_of_vehs_output_file = open ('../res/num_of_vehs_per_ap.ap', 'w+')
        
    def loc2ap_using_sq_cells (self, x, y):
        """
        Finding the AP covering the user's area, assuming that the number of APs is a power of 4.
        Input:  (x,y) location data
        Output: ap that covers this area
        """
        ap = 0
        x_offset, y_offset = x, y
        cur_edge = self.max_x / 2
        for p in range (self.max_power_of_4):
            ap += 4**(self.max_power_of_4-1-p)*int(2 * (y_offset // cur_edge) + x_offset // cur_edge)
            x_offset, y_offset = x_offset % cur_edge, y_offset % cur_edge   
            cur_edge /= 2
        return ap
    
    def print_usrs_ap (self):
        """
        Format-prints the users' AP, as caclulated earlier, to the .ap output file
        """
        
        new_usrs = list (filter (lambda usr: usr['new'], self.usrs))
        if (len (new_usrs) > 0):
            printf (self.ap_file, 'new_usrs: ')
            for usr in new_usrs: # for every new usr
                self.print_usr_ap (usr)

        old_usrs = list (filter (lambda usr: (usr['new']==False) and (usr['nxt ap'] != usr['cur ap']), self.usrs))
        if (len (old_usrs) > 0):
            printf (self.ap_file, '\nold_usrs: ')
            for usr in old_usrs: # for every existing usr
                self.print_usr_ap (usr)
                usr['cur ap'] = usr['nxt ap']
                
    def cnt_num_of_vehs_per_ap (self):
        """
        Count the number of vehicles associated with each AP during the simulation.
        """
        for ap in range(self.num_of_APs): 
            self.num_of_vehs_in_ap[ap].append (len (list (filter (lambda usr: usr['nxt ap'] == ap, self.usrs) )))
        
    def plot_num_of_vehs_per_ap (self):    
        """
        Plot for each ap the number of vehicles associated with it along the trace.
        """
        for plot_num in range (4**(self.max_power_of_4-1)):
            for ap in range (4*plot_num, 4*(plot_num+1)):
                printf (self.num_of_vehs_output_file, 'num_of_vehs_in_ap_{}: {}\n' .format (ap, self.num_of_vehs_in_ap[ap])) 
                plt.title ('Number of vehicles in each cell')
                plt.plot (range(len(self.num_of_vehs_in_ap[ap])), self.num_of_vehs_in_ap[ap], label='cell {}' .format(ap))
                plt.ylabel ('Number of vehicles')
            plt.legend()
            plt.savefig ('../res/num_of_vehs_per_cell_plot{}.jpg' .format(plot_num))
            plt.clf()
        
    def parse_file (self, usrs_loc_file_name):
        """
        - Read the input about the users locations.
        - Write the appropriate user-to-PoA connections to the file self.ap_file
        """
        self.usrs_loc_file_name = usrs_loc_file_name
        usrs_loc_file           = open ('../res/' + self.usrs_loc_file_name,  "r") 
        if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
            self.ap_file        = open ("../res/" + usrs_loc_file_name.split(".")[0] + ".ap",  "w+")  
            printf (self.ap_file, '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of user X at time t\n')
            printf (self.ap_file, 'num_of_APs = {}\n' .format (self.num_of_APs))
        
        for line in usrs_loc_file: 
    
            # remove the new-line character at the end (if any), and ignore comments lines 
            line = line.split ('\n')[0] 
            if (line.split ("//")[0] == ""):
                continue
    
            splitted_line = line.split (" ")
    
            if (splitted_line[0] == "t"):
                self.t = int(splitted_line[2])
                if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
                    self.print_usrs_ap() # First, print the APs of the users in the PREVIOUS cycles
                    printf(self.ap_file, '\n{}' .format (line)) # print the header of the current time: "t = ..."
                if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
                    self.cnt_num_of_vehs_per_ap ()
                for usr in self.usrs: # mark all existing usrs as old
                    usr['new'] = False
                    usr['cur ap'] = usr ['nxt ap']
                continue
    
            elif (splitted_line[0] == 'usrs_that_left:'):
                if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
                    printf(self.ap_file, '\n{}\n' .format (line))
                continue
    
            elif (splitted_line[0] != 'new_or_moved:'): 
                nxt_ap = self.loc2ap (float(splitted_line[2]), float(splitted_line[3]))
                usr_id = np.uint16(splitted_line[1])
                if (splitted_line[0] == 'n'): # new vehicle
                    self.usrs.append (
                        {'id' : usr_id, 'cur ap' : nxt_ap, 'nxt ap' : nxt_ap, 'new' : True}) # for a new usr, we mark the cur_ap same as nxt_ap 
                else: # existing user, who moved
                    list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, self.usrs))
                    if (len(list_of_usr) == 0):
                        print  ('Error at t={}: input file={}. Did not find old usr {}' .format (self.t, self.usrs_loc_file_name, splitted_line[1]))
                        exit ()
                    list_of_usr[0]['nxt ap'] = nxt_ap
    
    def post_processing (self):
        """
        Post processing after finished parse all the input file(s).
        The post processing may include:
        - Adding some lines to the output .ap file.
        - Plot the num_of_vehs 
        """
        if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
            printf(self.ap_file, "\n")   
        if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
            self.plot_num_of_vehs_per_ap ()
     
    def print_intermediate_res (self): 
        if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
            self.num_of_vehs_output_file = open ('../res/num_of_vehs_per_ap.ap', 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.num_of_vehs_output_file, '// after parsing the file {}\n' .format (self.usrs_loc_file_name))
            for ap in range (self.num_of_APs):
                printf (self.num_of_vehs_output_file, 'num_of_vehs_in_ap_{}: {}\n' .format (ap, self.num_of_vehs_in_ap[ap])) 
    
if __name__ == '__main__':
    max_power_of_4 = 3        
    my_loc2ap      = loc2ap_c (max_power_of_4 = max_power_of_4, use_sq_cells = True, verbose = VERBOSE_AP)
    
    for i in range (1): 
        usrs_loc_file_name = 'short_{}.loc' .format (i)
        my_loc2ap.parse_file             (usrs_loc_file_name)
        my_loc2ap.print_intermediate_res ()
        i += 1
    my_loc2ap.post_processing ()
