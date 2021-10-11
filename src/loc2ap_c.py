import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
# from matplotlib.colors import LogNorm, Normalize
# from matplotlib.ticker import MaxNLocator
# import itertools 
# import time 

from printf import printf, printar, printmat
# from ntpath import split

GLOBAL_MAX_X_LUX, GLOBAL_MAX_Y_LUX = int(13622), int(11457)             # size of the city's area, in meters. 
MAX_X_LUX,        MAX_Y_LUX        = GLOBAL_MAX_X_LUX//2, GLOBAL_MAX_Y_LUX//2   # maximal allowed x,y values for the simulated area (which is possibly only a part of the full city area)  
LOWER_LEFT_CORNER          = np.array ([GLOBAL_MAX_X_LUX//4,   GLOBAL_MAX_Y_LUX//4], dtype='int16') # x,y indexes of the south-west corner of the simulated area

# Verbose levels, defining the outputs produced
VERBOSE_AP              = 1 # Generate ".ap" file, detailing the current cell of each vehicle during the sim.
VERBOSE_CNT             = 2 # Generate ".txt" file, detailing the number of vehicles at each cell during the sim.
VERBOSE_DEMOGRAPHY      = 3 # Collect data about the # of vehicles entering / leaving each cell, at each time slot`
VERBOSE_SPEED           = 4 # Collect data about the speed of vehicles in each cell, at each time slot`
VERBOSE_DEBUG           = 11

# Indices of the various field within the input '.loc' file 
type_idx   = 0 # type of the vehicle: either 'n' (new veh, which has just joined the sim), or 'o' (old veh, that moved). 
veh_id_idx = 1
x_pos_idx  = 2
y_pos_idx  = 3
speed_idx  = 4

# # types of vehicles' id
# new      = 0 # a new (unused before) vehicle's id
# old      = 1 # a vehicle that already exists in the sim
# recycled = 2 # a recycled id, namely, a new vehicle using an id of an old vehicle, who left.  

class loc2ap_c (object):
    """
    Accept as input a .loc file (a file detailing the locations of all users.
    Input: at each slot, all users who are either new (namely, just joined the simulated area); or "old" (users who already were in the simulated, but moved to another cell/AP).
    Optional input: a list of antennas locations. 
    Optional outputs: 
    - An .ap file (a file detailing the Access Points of all the new users / users who moved at each slot).
    - A cnt of the number of vehicles in each cell
    "AP" means: Access Point, with which a user is associated at each slot.
    "cell" means: the rectangular cell, to which a user is associated at each slot.
    When using real-world antennas locations, the user's cell is the cell of the AP with which the user is currently associated.
    Otherwise, "AP" and "cells" are identical.  
    """   
    # Map a given x,y position to an AP.
    # If there're real AP locations, use Voronoi distance (map each client to the nearest AP).
    # Else, use uniform-size rectangular cells, rather than finding the , replace the function below with the commented function below it.
    loc2ap = lambda self, x, y : self.loc2cell_using_rect_cells (x, y, max_power_of_4=self.max_power_of_4) if self.use_rect_AP_cells else self.nearest_ap (x,y) 

    # inline function for formatted-printing the AP of a single user
    print_usr_ap = lambda self, usr: printf(self.ap_file, "({},{})" .format (usr['id'], usr['nxt ap']))   
    
    # returns the distance between a given (x,y) position, and a given antenna
    sq_dist = lambda self, x, y, antenna : (x - antenna['x'])**2 + (y - antenna['y'])**2
    
    # Given a (x,y) position, returns the list of distances from it to all the APs
    list_of_sq_dists_from_APs = lambda self, x, y : np.array([self.sq_dist(x,y, AP) for AP in self.list_of_APs])
    
    # returns the id of the nearest antenna to the given (x,y) position
    # the func' ASSUMES THAT THE AP ID IS IDENTICAL TO THE INDEX OF THE AP IN THE LIST OF APS
    nearest_ap = lambda self, x, y : np.argmin (self.list_of_sq_dists_from_APs (x,y))
    
    # Returns the rectangular cell to which a given AP antenna belongs. 
    # If using only rectangular cells (not real antennas locations), the "cell" is merely identical to the "ap"
    ap2cell = lambda self, ap : ap if self.use_rect_AP_cells else self.list_of_APs[ap]['cell']  

    # An indication, expressing whether the mapping to cell used rectangular cells, or an antenna-locations (.antloc) input file
    cell_type_identifier = lambda self : '' if self.use_rect_AP_cells else '_ant'
    
    avg_num_of_vehs_per_cell = lambda self : np.array ([np.average(self.num_of_vehs_in_cell[cell]) for cell in range(self.num_of_cells)]) 
  
    gen_columns_for_heatmap = lambda self, lvl=0 : [str(i) for i in range(2**(self.max_power_of_4-lvl))]

    def __init__(self, max_power_of_4=3, verbose = VERBOSE_AP, antenna_loc_file_name=''):
        """
        Init a "loc2ap_c" object.
        A loc2ap_c is used can read ".loc" files (files detailing the location of each veh over time), and output ".ap" files (files detailing the AP assignment of each veh), and/or statistics 
        (e.g., number of vehs entering/levaing each cell, avg # of vehs in each cell, etc.).
        """

        self.verbose               = verbose      # verbose level - defining which outputs will be written
        self.debug                 = False 
        self.antenna_loc_file_name = antenna_loc_file_name
       
        self.max_x, self.max_y = MAX_X_LUX, MAX_Y_LUX # borders of the simulated area, in meters
        self.usrs              = []
        self.use_rect_AP_cells   = True if (self.antenna_loc_file_name=='') else False  

        self.max_power_of_4    = max_power_of_4
        self.num_of_cells      = 4**max_power_of_4
        self.list_of_APs = [] # List of the APs. Will be filled only if using antennas locations (and not synthetic rectangular cells).
        
        if (self.use_rect_AP_cells):
            self.num_of_APs        = self.num_of_cells 
        else:
            self.parse_antloc_file(self.antenna_loc_file_name, plot_ap_locs_heatmap=False)
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_in_ap = [[] for _ in range (self.num_of_APs)]
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.joined_ap          = [[] for _ in range(self.num_of_APs)] # self.joined_ap[i][j] will count the # of clients that joined AP i at slot j
            self.joined_cell        = [[] for _ in range(self.num_of_cells)] # self.joined_cell[i][j] will count the # of clients that joined cell i at slot j
            self.left_ap            = [[] for _ in range(self.num_of_APs)] # self.left_ap[i][j] will count the # of clients that left AP i at slot j
            self.left_cell          = [[] for _ in range(self.num_of_cells)] # self.left_cell[i][j] will count the # of clients that left cell i at slot j
            self.joined_ap_sim_via  = [[] for _ in range(self.num_of_APs)] # self.joined_ap_sim_via[i][j] will count the # of clients that left the sim at slot j, and whose last AP in the sim was AP i
            self.left_ap_sim_via    = [[] for _ in range(self.num_of_APs)] # self.left_ap_sim_via[i][j] will count the # of clients that left the sim at slot j, and whose last cell in the sim was cell i 
        if (VERBOSE_SPEED in self.verbose):
            self.speed_file = open ('../res/vehicles_speed.txt', 'w+')
            self.speed           = [{'speed' : 0, 'num of smpls' : 0} for _ in range(self.num_of_APs)]
        self.calc_tile2cell (lvl=0) # calc_tile2cell translates the number as a "tile" (XY grid) to the ID of the covering cell.
        
    def parse_antloc_file (self, antennas_loc_file_name, plot_ap_locs_heatmap=False):
        """
        Parse an .antloc file.
        An .antloc file is a file containing the list of antennas, with their IDs and (x,y) position within the simulated area
        """
        antennas_loc_file = open ('../res/antennas_locs/' + antennas_loc_file_name, 'r')
        
        for line in antennas_loc_file: 
        
            if (line == "\n" or line.split ("//")[0] == ""): # skip lines of comments and empty lines
                continue
            
            splitted_line = line.split (',')
            x = float(splitted_line[1])
            y = float(splitted_line[2])
            self.list_of_APs.append ({'id' : float(splitted_line[0]), 'x' : x, 'y' : y, 'cell' : self.loc2cell_using_rect_cells (x, y, max_power_of_4=self.max_power_of_4)}) # 'cell' is the rectangle of the simulated area in which this AP is found
            
        self.num_of_APs = len (self.list_of_APs)
        
        ap2cell_file = open ('../res/{}_{}cells.ap2cell' .format(antennas_loc_file_name, self.num_of_cells), 'w')
        printf (ap2cell_file, '// This file details the cell associated with each AP.\n// Format: a c\n// Where a is the ap number, and c is the number of cell associated with it.\n')
        
        for ap in range(self.num_of_APs):
            printf (ap2cell_file, '{} {}\n' .format (ap, self.ap2cell(ap)))
        
        self.calc_tile2cell (lvl=0)
        
        return
         
        # if (plot_ap_locs_heatmap):
            # num_of_aps_per_cell = self.calc_num_of_aps_per_cell()        

        # Plots a heatmap, showing the number of APs in each cell.
        # When using rectangular AP-cells, the heatmap should show fix 1 for all cells.
        # When using an '.antloc' file, the heatmap shows the number of antennas in each cell.
        # num_of_aps_per_cell = self.calc_num_of_aps_per_cell()
        # for lvl in range (self.max_power_of_4):
        #     plt.figure()       
        #     my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (num_of_aps_per_cell), columns = self.gen_columns_for_heatmap(lvl)), cmap="YlGnBu")#, norm=LogNorm())
        #     my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
        #     plt.title   ('number of APs per cell')
        #     plt.savefig ('../res/heatmap_num_APs_per_cell_{}_{}cells.jpg' .format (self.antenna_loc_file_name, int(self.num_of_cells/(4**lvl))))
        #
        #     if (lvl < self.max_power_of_4-1): # if this isn't the last iteration, need to adapt avg_num_of_vehs_per_cell for the next iteration
        #         num_of_aps_per_cell = self.aggregate_heatmap_cells (num_of_aps_per_cell)
        
        plt.figure()
        plt.plot([ap['x'] for ap in self.list_of_APs], [ap['y'] for ap in self.list_of_APs], 'o', color='black');
        plt.axis([0, MAX_X_LUX, 0, MAX_Y_LUX])
        plt.savefig('../res/{}_ap_points.jpg' .format (self.antenna_loc_file_name))
        plt.clf()
        
    def loc2cell_using_rect_cells (self, x, y, max_power_of_4):
        """
        Finding the AP covering the user's area, assuming that the number of APs is a power of 4, and rectangular cells.
        Input:  (x,y) location data
        Output: ap that covers this area
        """
        ap = np.int8(0)
        x_offset, y_offset = x, y
        x_edge, y_edge = 0.5*self.max_x, 0.5*self.max_y
        for p in range (max_power_of_4):
            ap += 4**(max_power_of_4-1-p)*int(2 * (y_offset // y_edge) + x_offset // x_edge) #Y: 5728/2864
            x_offset, y_offset = x_offset % x_edge, y_offset % y_edge   
            x_edge /= 2
            y_edge /= 2
        return ap
    
    def print_usrs_ap (self):
        """
        Format-prints the users' AP, as calculated earlier, to the .ap output file
        """
        usrs = list (filter (lambda usr: usr['new'], self.usrs))
        if (len (usrs) > 0):
            printf (self.ap_file, 'new_usrs: ')
            for usr in usrs: # for every new usr
                self.print_usr_ap (usr)

        usrs = list (filter (lambda usr: (usr['new'] == False) and (usr['nxt ap'] != usr['cur ap']), self.usrs))
        printf (self.ap_file, '\nold_usrs: ')
        for usr in usrs: 
            self.print_usr_ap (usr)
            usr['cur ap'] = usr['nxt ap']
                
    def cnt_num_of_vehs_per_ap (self):
        """
        Count the number of vehicles associated with each AP in the current time slot.
        """
        for ap in range(self.num_of_APs): 
            self.num_of_vehs_in_ap[ap].append (len (list (filter (lambda usr: usr['nxt ap'] == ap, self.usrs) )))
    
    def print_demography (self):
        """
        Prints the number of vehicles that joined/left each cell during the last simulated time slot.
        """
        for ap in range(self.num_of_APs):
            printf (self.demography_file, 'ap_{}: joined {}\nap_{}: joined_ap_sim_via{}\nap_{}: left {}\nap_{}: left_sim_via {}\n' .format (
                                                ap, self.joined_ap[ap], 
                                                ap, self.joined_ap_sim_via[ap], 
                                                ap, self.left_ap[ap], 
                                                ap, self.left_ap_sim_via[ap]))
        for cell in range (self.num_of_cells):
            printf (self.demography_file, 'cell: left {}\n' .format (self.left_cell[cell]))

    def print_speed (self):
        """
        Prints the speed of vehicles that joined/left each cell during the last simulated time slot.
        """
        printf (self.speed_file, '{}' .format ([self.speed[ap]['speed'] for ap in range(self.num_of_APs)]))                                        

    def plot_demography_heatmap (self):
        """
        Plot heatmaps, showing the avg number of vehicles that joined/left each cell during the simulated period.
        """
        
        # Trunc the data of the first entry, in which obviously no veh joined/left any AP, or cell
        # self.joined_ap         = [self.joined_ap        [ap][1:] for ap in range (self.num_of_APs)] 
        self.joined_cell       = [self.joined_cell      [ap][1:] for ap in range (self.num_of_cells)] 
        # self.joined_ap_sim_via = [self.joined_ap_sim_via[ap][1:] for ap in range (self.num_of_APs)]

        # print ('avg num of vehs that: joined AP={:.2f}  ' .format 
        #        (np.average ([np.average(self.joined_ap[ap]) for ap in range(self.num_of_APs)])))
        # print ('avg num of vehs that: joined cell={:.2f}  ' .format 
        #        (np.average ([np.average(self.joined_cell[cell]) for cell in range(self.num_of_cells)])))
        # print ('left an AP={:.2f}' .format 
        #        (np.average ([np.average(self.left_ap[ap]) for ap in range(self.num_of_APs)])))
        # print ('left a cell={:.2f}' .format 
        #        (np.average ([np.average(self.left_cell[cell]) for cell in range(self.num_of_cells)])))
        # print ('joined the simulated area from ap={:.2f}' .format 
        #        (np.average ([np.average(self.joined_ap_sim_via[ap]) for ap in range(self.num_of_APs)])))
        # print ('left the simulated area from ap={:.2f}' .format 
        #        (np.average ([np.average(self.left_ap_sim_via[ap]) for ap in range(self.num_of_APs)])))
        #
        # plt.figure()
        #
        # columns = self.gen_columns_for_heatmap()
        file_name_suffix = '{}_{}rects' .format (self.time_period_str, self.num_of_cells)
        #
        # my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (np.array ([np.average(self.joined_cell[cell])    for cell in range(self.num_of_cell)])), columns=columns), cmap="YlGnBu")
        # # plt.title ('avg num of vehs that joined cell every sec in {}' .format (self.time_period_str))
        # plt.savefig('../res/heatmap_vehs_joined_cell{}.jpg' .format (file_name_suffix))
        # return
        
        heatmap_txt_file = open ('../res/heatmap_vehs_left.txt', 'w') 
        plt.figure()
        avg_vehs_left_per_rect = np.array ([np.average(self.left_cell[cell]) for cell in range(self.num_of_cells)])
         
        for lvl in range (0, self.max_power_of_4):
            columns = self.gen_columns_for_heatmap (lvl=lvl)
            self.calc_tile2cell (lvl) # call a function that translates the number as "tile" to the ID of the covering AP.
            plt.figure()       
            heatmap_vals = self.vec2heatmap (avg_vehs_left_per_rect)
            my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (heatmap_vals, columns=columns), cmap="YlGnBu"))
            printf   (heatmap_txt_file, 'lvl={}\n' .format (lvl+1))
            printmat (heatmap_txt_file, heatmap_vals, my_precision=2)
            my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
            plt.savefig('../res/heatmap_vehs_left_rect{}_{}_{}rects.jpg' .format (self.antenna_loc_file_name, self.usrs_loc_file_name, int(self.num_of_cells/(4**lvl))))
            if (lvl < self.max_power_of_4-1): # if this isn't the last iteration, need to adapt avg_vehs_left_per_rect for the next iteration
                avg_vehs_left_per_rect = self.avg_heatmap_cells (avg_vehs_left_per_rect)
        
        return ()
        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (np.array ([np.average(self.joined_ap_sim_via[ap]) for ap in range(self.num_of_APs)])), columns=columns), cmap="YlGnBu")
        # plt.title ('avg num of vehs that joined the sim every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_vehs_joined_sim_via_{}.jpg' .format (file_name_suffix))
        
        plt.figure ()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (np.array ([np.average(self.left_ap_sim_via[ap])   for ap in range(self.num_of_APs)])), columns=columns), cmap="YlGnBu")
        # plt.title ('avg num of vehs that left the sim every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_vehs_left_sim_via_{}.jpg' .format (file_name_suffix))
        
        
    def invert_mat_bottom_up (self, mat):
        """
        Unfortunately, we write matrix starting from the smallest value at the top, while plotting maps letting the "y" (north) direction "begin" at bottom, and increase towards the top.
        Hence, need to swap the matrix upside-down
        """ 
        inverted_mat = np.empty (mat.shape)
        for i in range (mat.shape[0]):
            inverted_mat[i][:] = mat[mat.shape[0]-1-i][:]
        return inverted_mat        

    def vec2heatmap (self, vec):
        """
        Order the values in the given vec so that they appear as in the geographical map of cells.
        """
        n = int (math.sqrt(len(vec)))
        if (len(vec) != len(self.tile2cell)): # The current mapping of tile2cell doesn't fit the number of rectangles in the given vec --> calculate a tile2cell mapping fitting the required len
            self.calc_tile2cell (lvl=self.max_power_of_4 - int(math.log2(n)))
        heatmap_val = np.array ([vec[self.tile2cell[i]] for i in range (len(self.tile2cell))]).reshape ( [n, n])
        
        # Unfortunately, we write matrix starting from the smallest value at the top, while plotting maps letting the "y" (north) direction "begin" at bottom, and increase towards the top.
        # Hence, need to swap the matrix upside-down
        return self.invert_mat_bottom_up(heatmap_val)

    def set_usrs_loc_file_name (self, usrs_loc_file_name=''):
        if (hasattr (self, 'usrs_loc_file_name')): # The field self.usrs_loc_file_name is already define
            return
        if (usrs_loc_file_name==''):
            print ('Please specify an existing usr loc file name')
            exit ()
        self.usrs_loc_file_name = usrs_loc_file_name
    
    def plot_num_of_vehs_in_cell_heatmaps (self, usrs_loc_file_name=''):
        """
        Generate a Python heatmap, showing at each cell the average number of vehicles found at that cell, along the simulation.
        The heatmaps are plotted for all possible resolutions between 4 cells, and the maximal # of cells simulated.
        """        
        
        self.set_usrs_loc_file_name(usrs_loc_file_name)

        avg_num_of_vehs_per_cell = self.avg_num_of_vehs_per_cell() 
        tmp_file = open ('../res/tmp.txt', 'w')
        for lvl in range (0, self.max_power_of_4):
            columns = self.gen_columns_for_heatmap (lvl=lvl)
            self.calc_tile2cell (lvl) # call a function that translates the number as "tile" to the ID of the covering AP.
            plt.figure()       
            heatmap_vals = self.vec2heatmap (avg_num_of_vehs_per_cell)
            if (lvl==3):
                my_heatmap = sns.heatmap (pd.DataFrame (heatmap_vals,columns=columns), cmap="YlGnBu", vmin=600, vmax=800)#, norm=LogNorm())
            else:
                my_heatmap = sns.heatmap (pd.DataFrame (heatmap_vals,columns=columns), cmap="YlGnBu")#, norm=LogNorm())
            printf (tmp_file, 'lvl={}\n' .format (lvl+1))
            printmat (tmp_file, heatmap_vals, my_precision=0)
            my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
            # plt.title ('avg num of vehs per cell')
            plt.savefig('../res/heatmap_num_vehs_{}_{}_{}rects.jpg' .format (self.antenna_loc_file_name, self.usrs_loc_file_name, int(self.num_of_cells/(4**lvl))))
            if (lvl < self.max_power_of_4-1): # if this isn't the last iteration, need to adapt avg_num_of_vehs_per_cell for the next iteration
                avg_num_of_vehs_per_cell = self.aggregate_heatmap_cells (avg_num_of_vehs_per_cell)

    def aggregate_heatmap_cells (self, vec):
        """
        aggregate the values within a vector in a way that allows using a heatmap with x0.25 the number of rectangles
        """
        reshaped_heatmap = vec.reshape (int(len(vec)/4), 4) # prepare the averaging for the next iteration
        return np.array([np.sum(reshaped_heatmap[i][:])for i in range(reshaped_heatmap.shape[0])], dtype='int') #perform the averaging, to be used by the ext iteration.

    def avg_heatmap_cells (self, vec):
        """
        average the values within a vector in a way that allows using a heatmap with x0.25 the number of rectangles
        """
        reshaped_heatmap = vec.reshape (int(len(vec)/4), 4) # prepare the averaging for the next iteration
        return np.array([np.mean(reshaped_heatmap[i][:])for i in range(reshaped_heatmap.shape[0])], dtype='int') #perform the averaging, to be used by the ext iteration.

    def plot_num_of_vehs_per_AP (self, usrs_loc_file_name=''):
        """
        Generate a Python heatmap, showing for each cell the average number of vehicles found at that cell, over the number of antennas in this cell.
        The heatmaps are plotted for all possible resolutions between 4 cells, and the maximal # of cells simulated.
        """        
        
        self.set_usrs_loc_file_name(usrs_loc_file_name)
        self.calc_num_of_vehs_per_cell()
        avg_num_of_vehs_per_cell = self.avg_num_of_vehs_per_cell ()
        num_of_aps_per_cell      = self.calc_num_of_aps_per_cell()
        avg_num_of_vehs_per_AP = np.array([(0 if (num_of_aps_per_cell[c]==0) else avg_num_of_vehs_per_cell[c] / num_of_aps_per_cell[c]) for c in range(self.num_of_cells) ])
        for lvl in range (0, self.max_power_of_4):
            self.calc_tile2cell (lvl) # call a function that translates the number as "tile" to the ID of the covering AP.
            plt.figure()       
            my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (avg_num_of_vehs_per_AP),columns=self.gen_columns_for_heatmap(lvl=lvl)), cmap="YlGnBu")#, norm=LogNorm())
            my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
            plt.title ('avg num of vehs per AP')
            plt.savefig('../res/heatmap_num_vehs_per_AP_{}_{}_{}cells.jpg' .format (self.antenna_loc_file_name, self.usrs_loc_file_name, int(self.num_of_cells/(4**lvl))))
            reshaped_heatmap = avg_num_of_vehs_per_AP.reshape (int(len(avg_num_of_vehs_per_AP)/4), 4) # prepare the averaging for the next iteration
            if (lvl < self.max_power_of_4-1): # if this isn't the last iteration, need to adapt avg_num_of_vehs_per_AP for the next iteration
                avg_num_of_vehs_per_AP = np.array([np.sum(reshaped_heatmap[i][:])for i in range(reshaped_heatmap.shape[0])], dtype='int') #perform the averaging, to be used by the ext iteration.

    def calc_num_of_vehs_per_cell (self): 
        """
        Calculate the number of vehicles per cell at each slot during the simulated period, given the num of vehs per AP:
        self.num_of_vehs_in_cell[i][j] will hold the num of vehs in cell i in time slot j 
        """
        self.num_of_vehs_in_cell = np.zeros ( (self.num_of_cells, len(self.num_of_vehs_in_ap[0])), dtype='int16')  
        for ap in range (self.num_of_APs):
            self.num_of_vehs_in_cell[self.ap2cell(ap)] += np.array (self.num_of_vehs_in_ap[ap]) # Add the # of vehs in this AP to the (avg) number of vehs in the cell to which this AP belongs    
        
    def plot_speed_heatmap (self):
        """
        Plot a heatmap, showing the average speed of vehicles at each cell.
        """
        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap ([self.speed[ap]['speed'] for ap in range(self.num_of_APs)]), 
                                                columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('avg speed in each cell')
        plt.savefig('../res/heatmap_speed.jpg')
        
    def calc_tile2cell (self, lvl):
        """
        prepare a translation of the "Tile" (line-by-line regular index given to cells) to the cell id.
        """
        
        # To calclate the tile, we calculate positions within each cell in the simulated area, and then call loc2cell_using_rect_cells() to calculate the cell associated with this position. 
        max_power_of_4   = self.max_power_of_4 - lvl
        n                = int(math.sqrt (self.num_of_cells/4**lvl))
        self.tile2cell   = np.empty (n**2, dtype = 'uint8')
        offset_x         = self.max_x // (2*n)        
        offset_y         = self.max_y // (2*n)        
        rect             = 0
        for y in range (offset_x, self.max_y, self.max_y // n): 
            for x in range (offset_y, self.max_x, self.max_x // n): 
                self.tile2cell[rect] = self.loc2cell_using_rect_cells(x, y, max_power_of_4=max_power_of_4) 
                rect+=1 

    def plot_num_of_vehs_per_ap_graph (self):    
        """
        Plot for each ap the number of vehicles associated with it along the trace.
        """
        for plot_num in range (4**(self.max_power_of_4-1)):
            for cell in range (4*plot_num, 4*(plot_num+1)):
                if (VERBOSE_CNT in self.verbose):
                    printf (self.num_of_vehs_output_file, 'num_of_vehs_in_cell{}: {}\n' .format (cell, self.num_of_vehs_in_cell[cell])) 
            #     plt.title ('Number of vehicles in each cell')
            #     plt.plot (range(len(self.num_of_vehs_in_ap[ap])), self.num_of_vehs_in_ap[ap], label='cell {}' .format(ap))
            #     plt.ylabel ('Number of vehicles')
            #     plt.xlabel ('time [minutes, starting at 07:30]')
            # plt.legend()
            # plt.savefig ('../res/num_of_vehs_per_cell_plot{}.jpg' .format(plot_num))
            # plt.clf()
            
 
    def parse_file (self):
        """
        - Read and parse input ".loc" file, detailing the users locations 
        - Write the appropriate user-to-PoA connections to the file self.ap_file, or to files summing the number of vehicles per cell.
        """
        for line in self.usrs_loc_file: 
    
            # remove the new-line character at the end (if any), and ignore comments lines 
            line = line.split ('\n')[0] 
            if (line.split ("//")[0] == ""):
                continue
    
            splitted_line = line.split (" ")
            
            if (splitted_line[0] == "t"): # reached the next simulation time slot
                if (VERBOSE_AP in self.verbose):
                    printf(self.ap_file, '\n{}\n' .format (line)) # print the header of the current time: "t = ..."
                self.t = int(splitted_line[2])
                if (VERBOSE_DEMOGRAPHY in self.verbose):
                    # for ap in range (self.num_of_APs): 
                    #     self.joined_ap        [ap]  .append(np.int16(0))
                    #     self.left_ap          [ap]  .append(np.int16(0))
                    #     self.joined_ap_sim_via[ap]  .append(np.int16(0))
                    #     self.left_ap_sim_via  [ap]  .append(np.int16(0))
                    for cell in range (self.num_of_cells): 
                        self.joined_cell      [cell].append(np.int16(0))
                        self.left_cell        [cell].append(np.int16(0))
                continue

            elif (splitted_line[0] == 'usrs_that_left:'):
                if (VERBOSE_AP in self.verbose):
                    printf(self.ap_file, '{}\n' .format (line))
                ids_of_usrs_that_left_ap = [int(id) for id in splitted_line[1:] if id!= '']                
                # if (VERBOSE_DEMOGRAPHY in self.verbose):
                #     for usr in list (filter (lambda usr: usr['id'] in ids_of_usrs_that_left_ap, self.usrs)): 
                #         self.left_ap_sim_via[usr['cur ap']][-1] += 1 # inc the # of vehicles left the sim' via this cell
                self.usrs = list (filter (lambda usr : (usr['id'] not in ids_of_usrs_that_left_ap), self.usrs))
                continue
    
            elif (splitted_line[0] == 'new_or_moved:'): 
                splitted_line = splitted_line[1:] # the rest of this line details the locations of users that are either new, or old (existing) users who moved during the last time slot
                if (splitted_line !=['']): # Is there a non-empty list of vehicles that are old / new / recycled?  

                    splitted_line = splitted_line[0].split (')') # split the line into the data given for each distinct usr
                    for tuple in splitted_line:  
                        if (len(tuple) <= 1): # no more new vehicles in this list. #$$$$$$$$$$$$ or <1?>
                            break
                        tuple = tuple.split("(")
                        tuple   = tuple[1].split (',')

                        usr_id   = np.uint16(tuple[veh_id_idx])
                        nxt_ap   = self.loc2ap (float(tuple[x_pos_idx]), float(tuple[y_pos_idx]))
                        nxt_cell = self.ap2cell (nxt_ap)
                        if (VERBOSE_DEBUG in self.verbose and nxt_ap not in range(self.num_of_APs)):
                            print ('Error: t = {} usr={}, nxt_ap={}, pos=({},{}), MAX_X={}, MAX_Y={}. num_of_aps={} ' .format (self.t, usr_id, nxt_ap, tuple[x_pos_idx], tuple[y_pos_idx], MAX_X_LUX, MAX_Y_LUX, self.num_of_APs))
                            print ('Calling loc2ap again for deubgging')
                            nxt_ap = self.loc2ap (float(tuple[x_pos_idx]), float(tuple[y_pos_idx]))
                            nxt_cell = self.ap2cell (nxt_ap)
                            exit ()

                        if (VERBOSE_DEBUG in self.verbose and nxt_ap >= self.num_of_APs):
                            print ('Error at t={}: got ap num {}. usr={}, x={:.0f},y={:.0f}' .format (self.t, nxt_ap, usr_id, tuple[x_pos_idx], tuple[y_pos_idx]))
                            exit ()
                        if (tuple[type_idx] == 'n'): # new vehicle
                            self.usrs.append ({'id' : usr_id, 'cur ap' : nxt_ap, 'nxt ap' : nxt_ap, 'nxt cell' : nxt_cell, 'new' : True}) # for a new usr, we mark the cur_ap same as nxt_ap 
                            if (VERBOSE_DEMOGRAPHY in self.verbose): 
                            #     self.joined_ap        [nxt_ap][-1]   += 1 # inc the # of usrs that joined this AP at this slot
                                self.joined_cell      [nxt_cell][-1] += 1 # inc the # of usrs that joined this cell at this slot
                            #     self.joined_ap_sim_via[nxt_ap][-1] += 1 # inc the # of usrs that joined the sim' via this cell
                        elif (tuple[type_idx] == 'o'): # recycled vehicle's id, or an existing user, who moved
                            list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, self.usrs))
                            if (len(list_of_usr) == 0):
                                print  ('Error at t={}: input file={}. Did not find old / recycled usr {}' .format (self.t, self.usrs_loc_file_name, usr_id))
                                exit ()
                            list_of_usr[0]['nxt ap']   = nxt_ap
                            list_of_usr[0]['nxt cell'] = nxt_cell
                            if (VERBOSE_DEMOGRAPHY in self.verbose and nxt_ap!= list_of_usr[0]['cur ap']): #this user moved to another cell  
                                # self.joined_ap[nxt_ap][-1]                   += 1 # inc the # of usrs that joined this AP
                                self.joined_cell[nxt_cell][-1]               += 1 # inc the # of usrs that joined this cell
                                # self.left_ap  [list_of_usr[0]['cur ap']][-1] += 1 # inc the # of usrs that left the previous cell of that usr
                        else:
                            print ('Wrong type of usr.')
                            exit () 
                        if (VERBOSE_SPEED in self.verbose):
                            self.speed[nxt_ap] = {'speed' : (float(tuple[speed_idx]) + self.speed[nxt_ap]['num of smpls'] * self.speed[nxt_ap]['speed'])/(self.speed[nxt_ap]['num of smpls'] + 1), 
                                                  'num of smpls' : self.speed[nxt_ap]['num of smpls'] + 1}
                            
                # At this point we finished handling all the usrs (left / new / moved) reported by the input ".loc" file at this slot. So now, output the data to ".ap" file, and/or to a file, counting the vehicles at each cell
                if (VERBOSE_AP in self.verbose):
                    self.print_usrs_ap() # Print the APs of the users 
                if (VERBOSE_CNT in self.verbose):
                    self.cnt_num_of_vehs_per_ap ()
                for usr in self.usrs: # mark all existing usrs as old
                    usr['new']      = False
                    usr['cur ap']   = usr ['nxt ap']
                    usr['cur cell'] = usr['nxt cell']
    
    def post_processing (self):
        """
        Post processing after finished parsing all the input file(s).
        The post processing may include:
        - Adding some lines to the output .ap file.
        - Plot the num_of_vehs 
        """
        if (VERBOSE_AP in self.verbose):
            printf(self.ap_file, "\n")   
        if (VERBOSE_CNT in self.verbose):
            # self.plot_num_of_vehs_per_ap_graph ()
            self.plot_num_of_vehs_in_cell_heatmaps()
            self.plot_num_of_vehs_per_AP()
        # if (VERBOSE_DEMOGRAPHY in self.verbose):
        #     self.plot_demography_heatmap()
        if (VERBOSE_SPEED in self.verbose):
            
            # first, fix the speed, as we assumed a first veh with speed '0'.
            for ap in [ap for ap in range (self.num_of_APs) if (self.speed[ap]['num of smpls'] > 0)]:
                self.speed[ap]['speed'] = self.speed[ap]['speed'] * (self.speed[ap]['num of smpls'] +1) / self.speed[ap]['num of smpls']
            self.print_speed()
            self.plot_speed_heatmap()
     
    def print_intermediate_res (self): 
        """
        Print the current aggregate results; used for having intermediate results when running long simulation.
        """
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_file_name, 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.num_of_vehs_output_file, '// after parsing the file {}\n' .format (self.usrs_loc_file_name))
            for ap in range (self.num_of_APs):
                printf  (self.num_of_vehs_output_file, 'num_of_vehs_in_ap_{}: ' .format (ap))
                printar (self.num_of_vehs_output_file, self.num_of_vehs_in_ap[ap])
            self.calc_num_of_vehs_per_cell()
            for cell in range (self.num_of_cells):
                printf  (self.num_of_vehs_output_file, 'num_of_vehs_in_cell_{}:' .format (cell))
                printar (self.num_of_vehs_output_file, self.num_of_vehs_in_cell[cell])
        if (VERBOSE_DEMOGRAPHY in self.verbose): 
            printf (self.demography_file, '// after parsing {}\n' .format (self.usrs_loc_file_name))                
            self.print_demography()
        if (VERBOSE_SPEED in self.verbose): 
            self.speed_file   = open ('../res/vehicles_speed.txt', 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.speed_file, '// after parsing {}\n' .format (self.usrs_loc_file_name))                
            self.print_speed()
    
    def parse_loc_files (self, loc_file_names):
        """
        Parse one or more ".loc" files, named "files_prefix_i.loc", where i = 0, 1, ... num_of_files-1
        E.g. if files_prefix = vehicles and num_of_files = 2,
        this function will parse the files vehicles_0.loc, vehicles_1.loc
        for each of the parsed files, the function will:
        1. output the number of vehicles at each ap. AND/OR
        2. output the APs of all new/left/moved users at each time slot.
        The exact behavior is by the value of self.verbose
        """
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_file_name = '../res/num_of_vehs_{}_{}aps.txt' .format (self.antenna_loc_file_name, self.num_of_APs)
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_file_name, 'w+')
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.demography_file = open ('../res/demography_{}_{}.txt' .format(loc_file_names[0].split('.')[0], 4**self.max_power_of_4), 'w+')
        if (VERBOSE_AP in self.verbose):
            self.ap_file_name = loc_file_names[0].split('.')[0] + '_' + self.cell_type_identifier() + str(self.num_of_APs) + 'aps' +".ap"
            self.ap_file        = open ("../res/" + self.ap_file_name, "w+")  
            if (self.use_rect_AP_cells):
                printf (self.ap_file, '// Using rectangle cells\n')
            else:
                printf (self.ap_file, '// .antloc file={}\n' .format (self.antenna_loc_file_name))

            printf (self.ap_file, '// File format:\n//for each time slot:\n')
            printf (self.ap_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (self.ap_file, '//"new_usrs" is a list of the new usrs, and their APs, e.g.: (0, 2)(1,3) means that new usr 0 is in cell 2, and new usr 1 is in cell 3.\n')
            printf (self.ap_file, '//"old_usrs" is a list of the usrs who moved to another cell in the last time slot, and their current APs, e.g.: (0, 2)(1,3) means that old usr 0 is now in cell 2, and old usr 1 is now in cell 3.\n')
        
        for file_name in loc_file_names: 
            self.usrs_loc_file_name = file_name
            self.usrs_loc_file      = open ('../res/loc_files/' + self.usrs_loc_file_name,  "r") 
            self.parse_file             ()
            self.print_intermediate_res ()
        self.post_processing ()

    def print_as_sq_mat (self, output_file, mat):
        """
        Receive a vector ("mat"), reshape it and format-print it to the given output file as a square mat.
        """
        n = int (math.sqrt(mat.shape[0] * mat.shape[1]))
        mat = mat.reshape (n, n)
        for x in range (n):
            for y in range (n):
                printf (output_file, '{:.0f}\t' .format (mat[x][y]))
            printf (output_file, '\n')
    
    def print_num_of_vehs_per_server (self, output_file_name):
        """
        Print the number of vehicles in the sub-tree below each server, assuming that the simulated area is iteratively partitioned to rectangular cells,
        so that the number of cells is a power of 4. 
        """
        output_file = open ('../res/' + output_file_name, 'w')
        printf (output_file, 'avg num of vehs per server\n')
        avg_num_of_vehs_per_ap = np.array ([np.average(self.num_of_vehs_in_ap[ap]) for ap in range(self.num_of_APs)]) 
        for lvl in range (self.max_power_of_4):
            printf (output_file, '\nlvl {}\n********************************\n' .format(lvl))
            self.calc_tile2cell (lvl) # call a function that translates the number as "tile" to the ID of the covering AP.
            self.print_as_sq_mat (output_file, self.vec2heatmap (avg_num_of_vehs_per_ap))
            reshaped_heatmap = avg_num_of_vehs_per_ap.reshape (int(len(avg_num_of_vehs_per_ap)/4), 4) # prepare the averaging for the next iteration
            avg_num_of_vehs_per_ap = np.array([np.sum(reshaped_heatmap[i][:])for i in range(reshaped_heatmap.shape[0])], dtype='int') #perform the averaging, to be used by the ext iteration.

    def calc_num_of_aps_per_cell (self):
        """
        Returns the number of aps in each cell
        """
        num_of_aps_in_cell = np.zeros (self.num_of_cells)
        
        for ap in range(self.num_of_APs):
            num_of_aps_in_cell[self.ap2cell(ap)] += 1
        
        return num_of_aps_in_cell

    def rd_num_of_vehs_per_ap_n_cell (self, input_file_name):
        """
        Read the number of vehicles at each ap, and each cell, as written in the input files. 
        """
        input_file  = open ('../res/' + input_file_name, "r")  
    
        self.num_of_vehs_in_ap   = [] #[[] for _ in range (self.num_of_APs)]   # self.num_of_vehs_in_ap[i][j] will hold the # of vehs in ap i in time slot j 
        self.num_of_vehs_in_cell = [] #[[] for _ in range (self.num_of_cells)] # self.num_of_vehs_in_cell[i][j] will hold the # of vehs in cell i in time slot j
        for line in input_file:
    
            if (line == "\n" or line.split ("//")[0] == ""):
                continue
    
            line = line.split ("\n")[0]
            splitted_line = line.split (":")
            vec_name = splitted_line[0].split('_') 
            vec_data = splitted_line[1].split('[')[1].split(']')[0].split()
            num_of_vehs = []
            for num_of_vehs_in_this_time_slot in vec_data:
                num_of_vehs.append (int(num_of_vehs_in_this_time_slot))
            if (vec_name[4] == 'ap'): # the vector's name begins by "num_of_vehs_in_ap"
                self.num_of_vehs_in_ap.append(num_of_vehs)
            else: # Now we know that the vector's name begins by "num_of_vehs_in_cell"
                self.num_of_vehs_in_cell.append(num_of_vehs)


    def plot_voronoi_diagram (self):
        """
        Plot a Voronoi diagram of the PoAs in self.list_of_APs
        """
        points = np.array ([[ap['x'], ap['y']] for ap in self.list_of_APs])
        
        # vor = Voronoi(points)
        fig = voronoi_plot_2d(Voronoi(points), show_vertices=False)
        plt.xlim(0, MAX_X_LUX); plt.ylim(0, MAX_Y_LUX)
        plt.show()

if __name__ == '__main__':

    List = [ [1, 2], [2,5], [5,1] ]

    max_power_of_4 = 4
    my_loc2ap      = loc2ap_c (max_power_of_4 = max_power_of_4, verbose = [VERBOSE_DEMOGRAPHY], antenna_loc_file_name = '') #'Lux.center.post.antloc')
    # my_loc2ap.plot_voronoi_diagram()
    
    # Processing
    my_loc2ap.parse_loc_files (['0829_0830_8secs.loc']) #'0730_0830_8secs.loc']) #(['0829_0830_8secs.loc' '0730_0830_8secs.loc']) #'0730_0830_8secs.loc'  (['0730.loc', '0740.loc', '0750.loc', '0800.loc', '0810.loc', '0820.loc'])
    # # my_loc2ap.plot_num_of_vehs_per_AP (usrs_loc_file_name='0829_0830_8secs.loc')
    
    # # Post=processing
    # my_loc2ap.rd_num_of_vehs_per_ap_n_cell ('num_of_vehs_Lux.center.post.antloc_1524aps.txt')# ('num_of_vehs_per_ap_256aps_ant.txt')
    # # my_loc2ap.plot_num_of_vehs_per_AP (usrs_loc_file_name='0730_0830_8secs.loc')
    # my_loc2ap.plot_num_of_vehs_in_cell_heatmaps (usrs_loc_file_name='0829_0830_8secs.loc')
