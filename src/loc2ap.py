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

class loc2ap:

    # Returns the AP covering a given (x,y) location, assuming that the cells are identical fixed-size squares
    loc2ap_sq = lambda self, x, y: int (math.floor ((y / cell_Y_edge) ) * num_of_APs_in_row + math.floor ((x / cell_X_edge) )) 
    
    def loc2ap (self, usrs_loc_file_name):
        """
        Read the input about the users locations, 
        and write the appropriate user-to-PoA connections to the file ap_file
        Assume that each AP covers a square area
        """
        self.ap_file       = open ("../res/" + usrs_loc_file_name.split(".")[0] + ".ap", "w+")  
        self.usrs_loc_file = open ("../res/" + usrs_loc_file_name,  "r") 
        printf (self.ap_file, '// File format:\n//time = t: (1,a1),(2,a2), ...where\n') 
        printf (self.ap_file, '// n_or_a is \'n\' for new user, \'o\' else. aX is the Point-of-Access of user X at time t\n')
            
        max_X, max_Y = 1000, 1000 # size of the square cell of each AP, in meters. 
        num_of_APs_in_row = int (math.sqrt (num_of_leaves)) #$$$ cast to int, floor  
        cell_X_edge = max_X / num_of_APs_in_row
        cell_Y_edge = cell_X_edge
            
        for line in usrs_loc_file: 
        
            # Ignore comments lines
            if (line.split ("//")[0] == ""):
                continue
    
            splitted_line = line.split (" ")
    
            if (splitted_line[0] == "time"):
                printf(ap_file, "\ntime = {}: " .format (splitted_line[2].rstrip()))
                continue
        
            elif (splitted_line[0] == "u"):
                ap = loc2ap_sq (float(splitted_line[2]), float(splitted_line[3])) 
                printf(ap_file, "({}, {})," .format (line.split (" ")[1], ap))
                continue
            
        printf(ap_file, "\n")   
    
    if __name__ == '__main__':
        max_X, max_Y = 1000, 1000 # size of the square cell of each AP, in meters. 
        