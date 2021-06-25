import networkx as nx
import numpy as np
import math
import itertools 
import time
import random
import heapq
import matplotlib.pyplot as plt

from usr_c import usr_c # class of the users
from decision_var_c import decision_var_c # class of the decision variables
from printf import printf
from scipy.optimize import linprog
from cmath import sqrt

class loc2ap_c (object):
    """
    Accept as input a .loc file (a file detailing the locations of all the new users / users who moved at each slot).
    Output a .ap file (a file detailing the Access Points of all the new users / users who moved at each slot).
    """

    
    # inline function for the files location-to-AP mapping
    # currently this function always maps assuming square cells.
    loc2ap = lambda self, x, y : self.loc2ap_using_sq_cells (x, y)
    
    # inline function for formatted-printing the AP of a single user
    print_usr_ap = lambda self, usr: printf(self.ap_file, "({},{})," .format (usr['id'], usr['nxt ap']))   

    def __init__(self, max_power_of_4=3):

        self.max_x, self.max_y = 12000, 12000 # size of the square cell of each AP, in meters. 
        
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

        old_usrs = list (filter (lambda usr: usr['new'] == False and usr['nxt ap'] != usr['cur ap'], self.usrs))
        if (len (old_usrs) > 0):
            printf (self.ap_file, 'old_usrs: ')
            for usr in old_usrs: # for every existing usr
                self.print_usr_ap (usr)
                usr['cur ap'] = usr['nxt ap']
                
    def cnt_num_of_vehs_per_ap (self):
        """
        Count the number of vehicles associated with each AP during the simulation.
        """
        for ap in self.num_of_APs: 
            self.num_of_vehs_in_ap[ap].append (len (list (filter (lambda usr: usr['nxt ap'] == ap, self.usrs) )))
        
        for usr in usrs:
            self.num_of_vehs_in_ap[usr['nxt ap']] += 1
    
    def parse_file (self, usrs_loc_file_name, use_sq_cells = True):
        """
        - Read the input about the users locations.
        - Write the appropriate user-to-PoA connections to the file self.ap_file
        """
        self.usrs_loc_file_name = usrs_loc_file_name
        self.ap_file  = open ("../res/" + self.usrs_loc_file_name.split(".")[0] + ".ap", "w+")  
        usrs_loc_file = open ("../res/" + self.usrs_loc_file_name,  "r") 
        printf (self.ap_file, '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of user X at time t\n')
        printf (self.ap_file, 'num_of_APs = {}' .format (self.num_of_APs))
        
        self.usrs = []
        for line in usrs_loc_file: 
    
            # remove the new-line character at the end (if any), and ignore comments lines 
            line = line.split ('\n')[0] 
            if (line.split ("//")[0] == ""):
                continue
    
            splitted_line = line.split (" ")
    
            if (splitted_line[0] == "t"):
                self.print_usrs_ap() # First, print the APs of the users in the PREVIOUS cycles
                self.cnt_num_of_vehs_per_ap ()
                printf(self.ap_file, '\n{}' .format (line)) # print the header of the current time: "t = ..."
                continue
    
            elif (splitted_line[0] == 'usrs_that_left:'):
                printf(self.ap_file, '\n{}\n' .format (line))
                continue
    
            elif (splitted_line[0] != 'new_or_moved:'): # new vehicle
                type   = splitted_line[0]
                nxt_ap = self.loc2ap (float(splitted_line[2]), float(splitted_line[3]))
                if (type == 'n'):
                    self.usrs.append (
                        {'id' : splitted_line[1], 'nxt ap' : nxt_ap, 'new' : True})
                else: # existing user, who moved
                    list_of_usr = list (filter (lambda usr: usr['id'] == splitted_line[1], self.usrs)) 
                    list_of_usr[0]['nxt ap'] == nxt_ap
    
        printf(self.ap_file, "\n")   
    
if __name__ == '__main__':
    # use_sq_cells                = True
    # if (use_sq_cells):
    #     max_power_of_4              = 2        
    #     self.num_of_APs        = 4**self.max_power_of_4
    #     my_loc2ap                   = loc2ap_c (max_power_of_4 = max_power_of_4)
    #     my_loc2ap.num_of_vehs_in_ap = np.empty (4**max_power_of_4, dtype = 'object')
    #     my_loc2ap.parse_file ('vehicles_1min.loc', use_sq_cells = use_sq_cells)

    X = [1,2,3,4]
    Y = [2,4,6,8]
    plt.plot (X, Y)
    plt.ylabel ('some numbers')
    plt.show()
        
        
    
    