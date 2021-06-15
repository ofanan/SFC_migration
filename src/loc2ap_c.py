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



#############################################################################
# Inline functions
#############################################################################

class loc2ap_c (object):

    # # Currently unused.
    # # Returns the AP covering a given (x,y) location, assuming that the cells are identical fixed-size squares, and a line-by-line cells numbering
    # loc2ap_sq = lambda self, x, y: int (math.floor ((y / self.cell_Y_edge) ) * self.num_of_APs_in_row + math.floor ((x / self.cell_X_edge) )) 
    
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
        print ('ap = ', ap)
    
    def __init__(self, num_of_APs=16, max_power_of_4=2):

        self.max_x, self.max_y = 12000, 12000 # size of the square cell of each AP, in meters. 
        self.max_power_of_4    = max_power_of_4    
        self.num_of_APs        = 4**self.max_power_of_4
        
        # # parameters to be used only for "line by line" cells' locations
        # self.num_of_APs         = num_of_APs
        # self.num_of_APs_in_row  = int (math.sqrt (self.num_of_APs)) #$$$ cast to int, floor  
        # self.cell_X_edge        = self.max_x / self.num_of_APs_in_row
        # self.cell_Y_edge        = self.cell_X_edge
    
    def loc2ap (self, usrs_loc_file_name):
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
        
        cur_ap_of_usr = [] # will hold pairs of (usr_id, cur_ap). 
        for line in usrs_loc_file: 
    
            # remove the new-line character at the end (if any), and ignore comments lines 
            line = line.split ('\n')[0] 
            if (line.split ("//")[0] == ""):
                continue
    
            splitted_line = line.split (" ")
    
            if (splitted_line[0] == "t" or splitted_line[0] == 'usrs_that_left:'):
                printf(self.ap_file, '\n{}' .format (line))
                continue
    
            elif (splitted_line[0] == 'new_or_moved:'): # new vehicle
                printf(self.ap_file, '\nnew_or_moved: ')
            
            else: # now we know that this line details a user that either joined, or moved.
                type   = splitted_line[0] # type will be either 'n', or 'o' (new, old user, resp.).
                usr_id = splitted_line[1]
                nxt_ap = self.loc2ap_sq (float(splitted_line[2]), float(splitted_line[3]))
                if (type == 'n'): # new vehicle
                    printf(self.ap_file, "({},{},{})," .format (type,usr_id, nxt_ap))                
                    cur_ap_of_usr.append({'id' : usr_id, 'ap' : nxt_ap})
                else: # old vehicle
                    list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, cur_ap_of_usr))
                    if (len (list_of_usr)== 0):
                        print ('Inaal raback')
                        exit ()
                    if (list_of_usr[0]['ap'] == nxt_ap): # The user is moving within area covered by the cur AP
                        continue
                    printf(self.ap_file, "({},{},{})" .format (type, usr_id, nxt_ap))                
                    list_of_usr[0]['ap'] = nxt_ap       
                continue
    
        printf(self.ap_file, "\n")   
    
if __name__ == '__main__':
    my_loc2ap = loc2ap_c (max_power_of_4 = 2)
    my_loc2ap.loc2ap_sq_power_of_4 (2999, 6999)
    
    # max_x = 12000
    # x, y = 11000, 11000
    # ap = int(0)
    # x_offset, y_offset = x, y
    # cur_edge = max_x / 2
    # for p in range (2):
    #     ap += 4**(1-p)*int(2 * (y_offset // cur_edge) + x_offset // cur_edge)
    #     x_offset, y_offset = x_offset % cur_edge, y_offset % cur_edge
    #     print (x_offset, y_offset)   
    #     cur_edge /= 2
    
    # my_loc2ap.loc2ap (usrs_loc_file_name = 'vehicles_1min.loc')
    