import networkx as nx
import numpy as np
import math
import itertools 
import time
import random
import heapq

from usr_c import usr_c # class of the users
from decision_var_c import decision_var_c # class of the decision variables
from printf import printf
from scipy.optimize import linprog
from cmath import sqrt

# Levels of verbose (which output is generated)
VERBOSE_NO_OUTPUT             = 0
VERBOSE_ONLY_RES              = 1 # Write to a file the total cost and rsrc aug. upon every event
VERBOSE_RES_AND_LOG           = 2 # Write to a file the total cost and rsrc aug. upon every event + write a detailed ".log" file

class loc2ap_c (object):
    
    # inline function for the files location-to-AP mapping
    # currently this function always maps assuming square cells.
    loc2ap = lambda self, x, y : loc2ap_sq_power_of_4 (self, x, y)
    
    # inline function for formatted-printing the AP of a single user
    print_usr_ap = lambda self, usr: printf(self.ap_file, "({},{})," .format (usr['id'], usr['nxt ap']))   

    def loc2ap_sq_power_of_4 (self, x, y):
        """
        Finding the AP covering the user's area, assuming that the number of APs is a power of 4.
        Input:  (x,y) location data
        Output: ap that covers this area
        """
        ap = 0
        x_offset, y_offset = x, y
        cur_edge = self.max_x / 2
        for p in range (self.max_power_of_4):
            ap += 4**(1-p)*int(2 * (y_offset // cur_edge) + x_offset // cur_edge)
            x_offset, y_offset = x_offset % cur_edge, y_offset % cur_edge   
            cur_edge /= 2  
        return ap
    
    def __init__(self, num_of_APs=16, max_power_of_4=2):

        self.max_x, self.max_y = 12000, 12000 # size of the square cell of each AP, in meters. 
        self.max_power_of_4    = max_power_of_4    
        self.num_of_APs        = 4**self.max_power_of_4
        
        # # parameters to be used only for "line by line" cells' locations
        # self.num_of_APs         = num_of_APs
        # self.num_of_APs_in_row  = int (math.sqrt (self.num_of_APs)) #$$$ cast to int, floor  
        # self.cell_X_edge        = self.max_x / self.num_of_APs_in_row
        # self.cell_Y_edge        = self.cell_X_edge
    
    def print_usrs_ap (self):
        
        new_usrs = list (filter (usr['new']), self.usrs)
        if (len (new_usrs) > 0):
            printf (self.ap_file, 'new_usrs: ')
            for usr in new_usrs: # for every new usr
                self.print_usr_ap (usr)

        old_usrs = list (filter (usr['new'] == False and usr['nxt ap'] != usr['cur ap']), self.usrs)
        if (len (old_usrs) > 0):
            printf (self.ap_file, 'new_usrs: ')
            for usr in old_usrs: # for every existing usr
                self.print_usr_ap (usr)
                usr['cur ap'] = usr['cur ap']
    
    def parse_file (self, usrs_loc_file_name):
        """
        Read the input about the users locations, 
        and write the appropriate user-to-PoA connections to the file self.ap_file
        Assume that each AP covers a square area
        """
        self.usrs_loc_file_name = usrs_loc_file_name
        self.ap_file  = open ("../res/" + self.usrs_loc_file_name.split(".")[0] + ".ap", "w+")  
        usrs_loc_file = open ("../res/" + self.usrs_loc_file_name,  "r") 
        printf (self.ap_file, '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of user X at time t\n')
        printf (self.ap_file, 'num_of_APs = {}' .format (self.num_of_APs))
        
        # cur_ap_of_usr = [] # will hold pairs of (usr_id, cur_ap). 
        self.usrs = []
        for line in usrs_loc_file: 
    
            # remove the new-line character at the end (if any), and ignore comments lines 
            line = line.split ('\n')[0] 
            if (line.split ("//")[0] == ""):
                continue
    
            splitted_line = line.split (" ")
    
            if (splitted_line[0] == "t" or splitted_line[0] == 'usrs_that_left:'):
                self.print_usrs_ap()
                printf(self.ap_file, '\n{}' .format (line))
                # self.new_usrs, self.moved_usrs = [], []
                continue
    
            elif (splitted_line[0] == 'new_or_moved:'): # new vehicle
                # printf(self.ap_file, '\nnew_or_moved: ')
                self.new_or_moved = True
            
            else: # now we know that this line details a user that either joined, or moved.
                type   = splitted_line[0]
                nxt_ap = self.loc2ap (float(splitted_line[2]), float(splitted_line[3]))
                if (type == 'n'):
                    self.new_usrs.append (
                        {'id' : splitted_line[1], 'nxt ap' : nxt_ap, 'new' : True})
                    
                    # printf(self.ap_file, "({},{},{})," .format (type,usr_id, nxt_ap))                
                    # cur_ap_of_usr.append({'id' : usr_id, 'ap' : nxt_ap})
                else: # existing user, who moved
                    list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, self.old_usrs)) #cur_ap_of_usr))
                    if (len (list_of_usr)== 0):
                        print ('Inaal raback')
                        exit ()
                    list_of_usr[0]['nxt ap'] == nxt_ap
    
        printf(self.ap_file, "\n")   
    
if __name__ == '__main__':
    my_loc2ap = loc2ap_c (max_power_of_4 = 2)
    my_loc2ap.parse_file ('vehicles_1min.loc')
    
    