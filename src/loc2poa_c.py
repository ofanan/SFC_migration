import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
# from matplotlib.co//lors import LogNorm, Normalize
# from matplotlib.ticker import MaxNLocator
import itertools 
# import time 
# from ntpath import split

# My own format print functions 
from printf import printf, printar, printmat

# Verbose levels, defining the outputs produced
VERBOSE_POA              = 1 # Generate ".poa" file, detailing the current Point of Access of each user during the sim.
VERBOSE_CNT              = 2 # Generate ".txt" file, detailing the number of vehicles at each cell during the sim, and generate from it plots / heatmaps
VERBOSE_DEMOGRAPHY       = 3 # Collect data about the # of vehicles entering / leaving each cell, at each time slot, and generate from it diagrams / heatmaps
VERBOSE_SPEED            = 4 # Collect data about the speed of vehicles in each cell, at each time slot`
VERBOSE_DEBUG            = 5
VERBOSE_FIND_AREA        = 6 # Find the min and max positions of all simulated vehs along the simulation.

# size of the city's area, in meters.
GLOBAL_MAX_X = {'Lux' : int(13622), 'Monaco' : 9976}
GLOBAL_MAX_Y = {'Lux' : int(11457), 'Monaco' : 6356} # Monaco: min_x_pos_found= 0.0 max_x_pos_found= 9976.0 min_y_pos_found= 143.0 max_y_pos_found= 6356.0            

# maximal allowed x,y values for the simulated area (which is possibly only a part of the full city area)
MAX_X = {'Lux' :  GLOBAL_MAX_X['Lux']//2, 'Monaco' : GLOBAL_MAX_X['Monaco']}
MAX_Y = {'Lux' :  GLOBAL_MAX_Y['Lux']//2, 'Monaco' : GLOBAL_MAX_Y['Monaco']}

# x,y indexes of the south-west corner of the simulated area
LOWER_LEFT_CORNER = {'Lux'   : np.array ([GLOBAL_MAX_X['Lux']//4,   GLOBAL_MAX_Y['Lux']//4], dtype='int16'), 
                    'Monaco' : np.zeros (2, dtype='int16')} 

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

directions = ['s', 'n', 'e', 'w', 'se', 'sw', 'ne', 'nw', 'out']

class loc2poa_c (object):
    """
    This class processes the locations of users (vehicles/pedestrians) and of antennas; calculates and plots statistics (e.g., about usrs' mobility); and calculates / plots the assignment of usrs to antennas, and of antennas to rectangular cells.
    Inputs: 
    Typical input is a .loc file (a file detailing the locations of all users.
    at each slot, all users who are either new (namely, just joined the simulated area); or "old" (users who already were in the simulated, but moved to another cell/PoA).
    Optional input: a list of antennas locations. 
    Optional outputs: 
    - An .poa file (a file detailing the Access Points of all the new users / users who moved at each slot).
    - A cnt of the number of vehicles in each cell
    - various plots (e.g., heatmaps of the number of vehs within each rectangles, or the number of vehs joining/levaving each rec).
    "PoA" means: Access Point, with which a user is associated at each slot.
    "cell" means: the rectangular cell, to which a user is associated at each slot.
    When using real-world antennas locations, the user's cell is the cell of the PoA with which the user is currently associated.
    Otherwise, "AP" and "cells" are identical.  
    """   

    # Given a length of a vec, return the square root of its length
    sqrt_len = lambda self, vec : int (math.sqrt(len(vec)))

    # Given n and index, returns the position of the index in a nXn mat.
    vec2mat_idx = lambda  self, n, idx : [idx//n, idx%n]
    
    # reshape a vector of length n^2 as a n X n mat   
    vec2sq_mat = lambda self, vec : vec.reshape ( [self.sqrt_len(vec), self.sqrt_len(vec)])
    
    # Map a given x,y position to an PoA (Point of Access).
    # If there input include real PoA locations, the mapping is by Voronoi distance, namely, each client is mapped to the nearest poa.
    # Else, use a partition of the area into uniform-size rectangular cells.
    loc2poa = lambda self, x, y : self.loc2cell_using_rect_cells (x, y, max_power_of_4=self.max_power_of_4) if self.use_rect_PoA_cells else self.nearest_poa (x,y) 

    # inline function for formatted-printing the PoA of a single user
    print_usr_poa = lambda self, usr: printf(self.poa_file, "({},{})" .format (usr['id'], usr['nxt poa']))   
    
    # returns the distance between a given (x,y) position, and a given antenna
    sq_dist = lambda self, x, y, antenna : (x - antenna['x'])**2 + (y - antenna['y'])**2
    
    # Given a (x,y) position, returns the list of distances from it to all the PoAs (Point of Access)
    list_of_sq_dists_from_PoAs = lambda self, x, y : np.array([self.sq_dist(x,y, poa) for poa in self.list_of_PoAs])
    
    # returns the id of the nearest antenna to the given (x,y) position
    # the func' ASSUMES THAT THE PoA ID IS IDENTICAL TO THE INDEX OF THE PoA IN THE LIST OF PoAS
    nearest_poa = lambda self, x, y : np.argmin (self.list_of_sq_dists_from_PoAs (x,y))
    
    # Returns the rectangular cell to which a given PoA antenna belongs. 
    # If using only rectangular cells (not real antennas locations), the "cell" is merely identical to the "poa"
    poa2cell = lambda self, poa : np.int16 (poa if self.use_rect_PoA_cells else self.list_of_PoAs[poa]['cell'])  

    # generate a string, which indicates the input files. Used for generating meaningful indicative output files names.    
    input_files_str = lambda self, loc_file_name : loc_file_name.split('.')[0] + '_{}' .format ('' if self.use_rect_PoA_cells else self.antloc_file_name.split('.')[1]) 
    
    # Calculate the avg number of vehs in each cell, along the whole sim
    avg_num_of_vehs_per_cell = lambda self : np.array ([np.average(self.num_of_vehs_in_cell[cell]) for cell in range(self.num_of_cells)]) 
  
    # Generate the 'columns' required for generating a heatmap
    gen_columns_for_heatmap = lambda self, lvl=0 : [str(i) for i in range(2**(self.max_power_of_4-lvl))]

    # return True iff the given idx is between 0 and self.num_of_cells
    is_in_range_of_cells = lambda self, idx : (idx >=0 and idx <= self.num_of_cells)

    def __init__(self, max_power_of_4=3, verbose = VERBOSE_POA, antloc_file_name='', city=''):
        """
        Init a "loc2poa_c" object.
        A loc2poa_c is used can read ".loc" files (files detailing the location of each veh over time), and output ".poa" files (files detailing the PoA assignment of each veh), and/or statistics 
        (e.g., number of vehs entering/levaing each cell, avg # of vehs in each cell, etc.).
        """

        self.verbose            = verbose      # verbose level - defining which outputs will be written
        self.debug              = False 
        self.antloc_file_name   = antloc_file_name
        self.city                  = (antloc_file_name.split('.')[0] if antloc_file_name!='' else city)
       
        self.max_x, self.max_y = MAX_X[self.city], MAX_Y[self.city] # borders of the simulated area, in meters
        self.usrs              = []
        self.use_rect_PoA_cells   = True if (self.antloc_file_name=='') else False

        self.max_power_of_4    = max_power_of_4
        self.num_of_cells      = 4**max_power_of_4
        self.num_of_tiles      = self.num_of_cells
        self.sqrt_num_of_cells = int (math.sqrt (self.num_of_cells))
        self.list_of_PoAs = [] # List of the PoAs. Will be filled only if using antennas locations (and not synthetic rectangular cells).
        
        if (self.use_rect_PoA_cells):
            self.num_of_PoAs        = self.num_of_cells 
        else:
            self.parse_antloc_file(self.antloc_file_name, plot_poa_locs_heatmap=False)
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_in_poa = [[] for _ in range (self.num_of_PoAs)]
        if (VERBOSE_SPEED in self.verbose):
            self.speed_file = open ('../res/vehicles_speed.txt', 'w+')
            self.speed           = [{'speed' : 0, 'num of smpls' : 0} for _ in range(self.num_of_PoAs)]
        self.calc_cell2tile (lvl=0) # calc_cell2tile translates the number as a "cell" to the ID in a vector, covering the same area as a tile
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.joined_poa          = [[] for _ in range(self.num_of_PoAs)] # self.joined_poa[i][j] will count the # of clients that joined PoA i at slot j
            self.joined_cell        = [[] for _ in range(self.num_of_cells)] # self.joined_cell[i][j] will count the # of clients that joined cell i at slot j
            self.left_poa            = [[] for _ in range(self.num_of_PoAs)] # self.left_poa[i][j] will count the # of clients that left PoA i at slot j
            self.left_cell          = [[] for _ in range(self.num_of_cells)] # self.left_cell[i][j] will count the # of clients that left cell i at slot j
            self.joined_poa_sim_via  = [[] for _ in range(self.num_of_PoAs)] # self.joined_poa_sim_via[i][j] will count the # of clients that left the sim at slot j, and whose last PoA in the sim was PoA i
            
            self.left_cell_to = []
            for _ in range(self.num_of_cells):
                self.left_cell_to.append ({'s' : 0, 'n' : 0, 'e' : 0, 'w' : 0, 'se' : 0, 'sw' : 0, 'ne' : 0, 'nw' : 0, 'out' : 0})
        # self.tmp_file = open ('../res/tile2cell.txt', 'a')
        # printf (self.tmp_file, 'cell2tile=\n')        
        # self.print_as_sq_mat (self.tmp_file, self.vec2sq_mat (self.cell2tile))
        # printf (self.tmp_file, '\ntile2cell=\n')
        # self.print_as_sq_mat (self.tmp_file, self.vec2sq_mat (self.tile2cell))
        # old_x, old_y = 272,2113
        # new_x, new_y = 234,3454
        # print ('old cell={}. new cell={}' .format (self.loc2poa(old_x, old_y), self.loc2poa(new_x, new_y))) 
        # exit ()
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.calc_ngbr_rects ()
    
    def calc_ngbr_rects (self):
        """
        Used for debugging only  
        Find the 4 (north, south, east, west) neighbours of each cell (rect). If it's an edge cell, the tile number of the neighbor will be -1.
        """
        self.ngbrs_of_cell = [None] * self.num_of_cells
    
        n = self.sqrt_num_of_cells
        for cell in range(self.num_of_cells):
            self.ngbrs_of_cell[self.cell2tile[cell]] = {'w'  : -1 if (cell%n==0)    else self.cell2tile [cell-1], # west neighbor 
                                        'e'  : -1 if (cell%n==n-1)  else self.cell2tile [cell+1], # east neighbor
                                        'n'  : -1 if (cell//n==0)   else self.cell2tile [cell-n], # north neighbor
                                        's'  : -1 if (cell//n==n-1) else self.cell2tile [cell+n],  # south neighbor
                                        'nw' : -1 if (cell%n==0   or cell//n==0)   else self.cell2tile [cell-n-1], # north-west neighbor
                                        'ne' : -1 if (cell%n==n-1 or cell//n==0)   else self.cell2tile [cell-n+1], # north-west neighbor
                                        'sw' : -1 if (cell%n==0   or cell//n==n-1) else self.cell2tile [cell+n-1], # north-west neighbor
                                        'se' : -1 if (cell%n==n-1 or cell//n==n-1) else self.cell2tile [cell+n+1] # north-west neighbor
                                        }
    
        # printf (self.tmp_file, '\nneighbours from west=\n')
        # for cell in range(self.num_of_cells):
        #     printf (self.tmp_file, '{}\t' .format (self.ngbrs_of_cell[cell]['w']))
        #     if (cell % n == n-1):
        #         printf (self.tmp_file, '\n') 
                        
    def direction_of_mv (self, src, dst):
        """
        returns the direction of the move from cell src to cell dst
        """
        for direction in ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw']:
            if (self.ngbrs_of_cell[src][direction] == dst):
                return direction
        return -1 # Error code - didn't find the relation between src and dst 
    
        
    def parse_antloc_file (self, antennas_loc_file_name, plot_poa_locs_heatmap=False):
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
            self.list_of_PoAs.append ({'id' : float(splitted_line[0]), 'x' : x, 'y' : y, 'cell' : self.loc2cell_using_rect_cells (x, y, max_power_of_4=self.max_power_of_4)}) # 'cell' is the rectangle of the simulated area in which this PoA is found
            
        self.num_of_PoAs = len (self.list_of_PoAs)
        
        poa2cell_file = open ('../res/{}_{}cells.poa2cell' .format(antennas_loc_file_name, self.num_of_cells), 'w')
        printf (poa2cell_file, '// This file details the cell associated with each PoA.\n// Format: a c\n// Where a is the poa number, and c is the number of cell associated with it.\n')
        
        for poa in range(self.num_of_PoAs):
            printf (poa2cell_file, '{} {}\n' .format (poa, self.poa2cell(poa)))
        
        self.calc_cell2tile (lvl=0)
        
        return
         
        # if (plot_poa_locs_heatmap):
            # num_of_poas_per_cell = self.calc_num_of_poas_per_cell()        

        # Plots a heatmap, showing the number of PoAs in each cell.
        # When using rectangular PoA-cells, the heatmap should show fix 1 for all cells.
        # When using an '.antloc' file, the heatmap shows the number of antennas in each cell.
        # num_of_poas_per_cell = self.calc_num_of_poas_per_cell()
        # for lvl in range (self.max_power_of_4):
        #     plt.figure()       
        #     my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (num_of_poas_per_cell), columns = self.gen_columns_for_heatmap(lvl)), cmap="YlGnBu")#, norm=LogNorm())
        #     my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
        #     plt.title   ('number of PoAs per cell')
        #     plt.savefig ('../res/heatmap_num_PoAs_per_cell_{}_{}cells.jpg' .format (self.antloc_file_name, int(self.num_of_cells/(4**lvl))))
        #
        #     if (lvl < self.max_power_of_4-1): # if this isn't the last iteration, need to adapt avg_num_of_vehs_per_cell for the next iteration
        #         num_of_poas_per_cell = self.aggregate_heatmap_cells (num_of_poas_per_cell)
        
        plt.figure()
        plt.plot([poa['x'] for poa in self.list_of_PoAs], [poa['y'] for poa in self.list_of_PoAs], 'o', color='black');
        plt.axis([0, MAX_X[self.city], 0, MAX_Y[self.city]])
        plt.savefig('../res/{}_poa_points.jpg' .format (self.antloc_file_name))
        plt.clf()
        
    def loc2cell_using_rect_cells (self, x, y, max_power_of_4):
        """
        Finding the PoA covering the user's area, assuming that the number of PoAs is a power of 4, and rectangular cells.
        Input:  (x,y) location data
        Output: poa that covers this area
        """
        poa = np.int8(0)
        x_offset, y_offset = x, y
        x_edge, y_edge = 0.5*self.max_x, 0.5*self.max_y
        for p in range (max_power_of_4):
            poa += 4**(max_power_of_4-1-p)*int(2 * (y_offset // y_edge) + x_offset // x_edge) 
            x_offset, y_offset = x_offset % x_edge, y_offset % y_edge   
            x_edge /= 2
            y_edge /= 2
        return poa
    
    def print_usrs_poa (self):
        """
        Format-prints the users' PoA, as calculated earlier, to the .poa output file
        """
        usrs = list (filter (lambda usr: usr['new'], self.usrs))
        if (len (usrs) > 0):
            printf (self.poa_file, 'new_usrs: ')
            for usr in usrs: # for every new usr
                self.print_usr_poa (usr)

        usrs = list (filter (lambda usr: (usr['new'] == False) and (usr['nxt poa'] != usr['cur poa']), self.usrs))
        printf (self.poa_file, '\nold_usrs: ')
        for usr in usrs: 
            self.print_usr_poa (usr)
            usr['cur poa'] = usr['nxt poa']
                
    def cnt_num_of_vehs_per_poa (self):
        """
        Count the number of vehicles associated with each PoA in the current time slot.
        """
        for poa in range(self.num_of_PoAs): 
            self.num_of_vehs_in_poa[poa].append (np.int16 (len (list (filter (lambda usr: usr['nxt poa'] == poa, self.usrs) ))))
    
    def print_demography (self):
        """
        Prints the number of vehicles that joined/left each cell during the last simulated time slot.
        """
        # for poa in range(self.num_of_PoAs):
        #     printf (self.demography_file, 'poa_{}: joined {}\npoa_{}: joined_poa_sim_via{}\npoa_{}: left {}\npoa_{}: \n' .format (
        #                                         poa, self.joined_poa[poa], 
        #                                         poa, self.joined_poa_sim_via[poa], 
        #                                         poa, self.left_poa[poa]))
        for cell in range (self.num_of_cells):
            printf (self.demography_file, 'cell_{}: left {}\n' .format (cell, self.left_cell[cell]))

    def print_speed (self):
        """
        Prints the speed of vehicles that joined/left each cell during the last simulated time slot.
        """
        printf (self.speed_file, '{}' .format ([self.speed[poa]['speed'] for poa in range(self.num_of_PoAs)]))                                        

    def plot_demography_heatmap (self):
        """
        Plot heatmaps, showing the avg number of vehicles that joined/left each cell during the simulated period.
        """
        
        # Trunc the data of the first entry, in which obviously no veh joined/left any PoA, or cell
        # self.joined_poa         = [self.joined_poa        [poa][1:] for poa in range (self.num_of_PoAs)] 
        self.joined_cell       = [self.joined_cell      [poa][1:] for poa in range (self.num_of_cells)] 
        print ('left a cell={:.2f}' .format 
               (np.average ([np.average(self.left_cell[cell]) for cell in range(self.num_of_cells)])))
        # print ('joined the simulated area from poa={:.2f}' .format 
        #        (np.average ([np.average(self.joined_poa_sim_via[poa]) for poa in range(self.num_of_PoAs)])))
        # print ('left the simulated area from poa={:.2f}' .format 
        #        (np.average ([np.average(self.left_poa_sim_via[poa]) for poa in range(self.num_of_PoAs)])))
        #
        # plt.figure()
        #
        # columns = self.gen_columns_for_heatmap()
        file_name_suffix = '{}rects' .format (self.num_of_cells)       
        heatmap_txt_file = open ('../res/heatmap_vehs_left.txt', 'a') 
        plt.figure()
        avg_vehs_left_per_rect = np.array ([np.average(self.left_cell[cell]) for cell in range(self.num_of_cells)])
         
        columns = self.gen_columns_for_heatmap (lvl=0)
        self.calc_cell2tile (lvl=0) # call a function that translates the number as "tile" to the ID of the covering PoA.
        plt.figure()       
        heatmap_vals = self.vec2heatmap (avg_vehs_left_per_rect)
        my_heatmap = sns.heatmap (pd.DataFrame (heatmap_vals, columns=columns), cmap="YlGnBu")
        printf   (heatmap_txt_file, 'num of rects = {}\n' .format ( self.num_of_cells))
        printmat (heatmap_txt_file, heatmap_vals, my_precision=2)
        my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
        plt.savefig('../res/heatmap_vehs_left_rect{}_{}_{}rects.jpg' .format (self.antloc_file_name, self.usrs_loc_file_name, int(self.num_of_cells)))

        return 
        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (np.array ([np.average(self.joined_poa_sim_via[poa]) for poa in range(self.num_of_PoAs)])), columns=columns), cmap="YlGnBu")
        # plt.title ('avg num of vehs that joined the sim every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_vehs_joined_sim_via_{}.jpg' .format (file_name_suffix))
        
        plt.figure ()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (np.array ([np.average(self.left_poa_sim_via[poa])   for poa in range(self.num_of_PoAs)])), columns=columns), cmap="YlGnBu")
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
        n = self.sqrt_len(vec)
        if (len(vec) != len(self.cell2tile)): # The current mapping of cell2tile doesn't fit the number of rectangles in the given vec --> calculate a cell2tile mapping fitting the required len
            self.calc_cell2tile (lvl=self.max_power_of_4 - int(math.log2(n)))
        heatmap_val = np.array ([vec[self.cell2tile[i]] for i in range (len(self.cell2tile))]).reshape ( [n, n])
        
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
            self.calc_cell2tile (lvl) # call a function that translates the number as "tile" to the ID of the covering PoA.
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
            plt.savefig('../res/num_vehs_{}_{}_{}rects.jpg' .format (self.antloc_file_name, self.usrs_loc_file_name, int(self.num_of_cells/(4**lvl))))
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

    def plot_num_of_vehs_per_PoA (self, usrs_loc_file_name=''):
        """
        Generate a Python heatmap, showing for each cell the average number of vehicles found at that cell, over the number of antennas in this cell.
        The heatmaps are plotted for all possible resolutions between 4 cells, and the maximal # of cells simulated.
        """        
        
        self.set_usrs_loc_file_name(usrs_loc_file_name)
        self.calc_num_of_vehs_per_cell()
        avg_num_of_vehs_per_cell = self.avg_num_of_vehs_per_cell ()
        num_of_poas_per_cell      = self.calc_num_of_poas_per_cell()
        avg_num_of_vehs_per_PoA = np.array([(0 if (num_of_poas_per_cell[c]==0) else avg_num_of_vehs_per_cell[c] / num_of_poas_per_cell[c]) for c in range(self.num_of_cells) ])
        for lvl in range (0, self.max_power_of_4):
            self.calc_cell2tile (lvl) # call a function that translates the number as "tile" to the ID of the covering PoA.
            plt.figure()       
            my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap (avg_num_of_vehs_per_PoA),columns=self.gen_columns_for_heatmap(lvl=lvl)), cmap="YlGnBu")#, norm=LogNorm())
            my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
            # plt.title ('avg num of vehs per PoA')
            plt.savefig('../res/heatmap_num_vehs_per_PoA_{}_{}_{}cells.jpg' .format (self.antloc_file_name, self.usrs_loc_file_name, int(self.num_of_cells/(4**lvl))))
            reshaped_heatmap = avg_num_of_vehs_per_PoA.reshape (int(len(avg_num_of_vehs_per_PoA)/4), 4) # prepare the averaging for the next iteration
            if (lvl < self.max_power_of_4-1): # if this isn't the last iteration, need to adapt avg_num_of_vehs_per_PoA for the next iteration
                avg_num_of_vehs_per_PoA = np.array([np.sum(reshaped_heatmap[i][:])for i in range(reshaped_heatmap.shape[0])], dtype='int') #perform the averaging, to be used by the ext iteration.

    def calc_num_of_vehs_per_cell (self): 
        """
        Calculate the number of vehicles per cell at each slot during the simulated period, given the num of vehs per PoA:
        self.num_of_vehs_in_cell[i][j] will hold the num of vehs in cell i in time slot j 
        """
        self.num_of_vehs_in_cell = np.zeros ( (self.num_of_cells, len(self.num_of_vehs_in_poa[0])), dtype='int16')  
        for poa in range (self.num_of_PoAs):
            self.num_of_vehs_in_cell[self.poa2cell(poa)] += np.array (self.num_of_vehs_in_poa[poa], dtype='int16') # Add the # of vehs in this PoA to the (avg) number of vehs in the cell to which this PoA belongs    
        
    def plot_speed_heatmap (self):
        """
        Plot a heatmap, showing the average speed of vehicles at each cell.
        """
        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec2heatmap ([self.speed[poa]['speed'] for poa in range(self.num_of_PoAs)]), 
                                                columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('avg speed in each cell')
        plt.savefig('../res/heatmap_speed.jpg')
        
    def calc_cell2tile (self, lvl):
        """
        prepare a translation of the "Tile" (line-by-line regular index given to cells) to the cell id.
        """
        
        # To calclate the tile, we calculate positions within each cell in the simulated area, and then call loc2cell_using_rect_cells() to calculate the cell associated with this position. 
        max_power_of_4   = self.max_power_of_4 - lvl
        n                = int(math.sqrt (self.num_of_cells/4**lvl))
        self.cell2tile   = np.empty (n**2, dtype = 'uint8')
        rect             = 0
        for y in range (self.max_y // (2*n), self.max_y, self.max_y // n): 
            for x in range (self.max_x // (2*n), self.max_x, self.max_x // n): 
                self.cell2tile[rect] = self.loc2cell_using_rect_cells(x, y, max_power_of_4=max_power_of_4) 
                rect+=1 
        
        # for demography verbose, we need also the other direction, which maps a given cell to its physical location in the tile. 
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.tile2cell = np.empty (self.num_of_tiles, dtype = 'uint8')
            for cell in range (self.num_of_cells):
                self.tile2cell[self.cell2tile[cell]] = cell

    def plot_num_of_vehs_per_poa_graph (self):    
        """
        Plot for each poa the number of vehicles associated with it along the trace.
        """
        for plot_num in range (4**(self.max_power_of_4-1)):
            for cell in range (4*plot_num, 4*(plot_num+1)):
                if (VERBOSE_CNT in self.verbose):
                    printf (self.num_of_vehs_output_file, 'num_of_vehs_in_cell{}: {}\n' .format (cell, self.num_of_vehs_in_cell[cell])) 
            #     plt.title ('Number of vehicles in each cell')
            #     plt.plot (range(len(self.num_of_vehs_in_poa[poa])), self.num_of_vehs_in_poa[poa], label='cell {}' .format(poa))
            #     plt.ylabel ('Number of vehicles')
            #     plt.xlabel ('time [minutes, starting at 07:30]')
            # plt.legend()
            # plt.savefig ('../res/num_of_vehs_per_cell_plot{}.jpg' .format(plot_num))
            # plt.clf()
            
 
    def parse_file (self):
        """
        - Read and parse input ".loc" file, detailing the users locations 
        - Write the appropriate user-to-PoA connections to the file self.poa_file, or to files summing the number of vehicles per cell.
        """
        if (VERBOSE_FIND_AREA in self.verbose):
            min_x_pos_found, min_y_pos_found = float ('inf'), float ('inf')   
            max_x_pos_found, max_y_pos_found = 0,0
              
        for line in self.usrs_loc_file: 
    
            # remove the new-line character at the end (if any), and ignore comments lines 
            line = line.split ('\n')[0] 
            if (line.split ("//")[0] == ""):
                continue
    
            splitted_line = line.split (" ")
            
            if (splitted_line[0] == "t"): # reached the next simulation time slot
                if (VERBOSE_POA in self.verbose):
                    printf(self.poa_file, '\n{}\n' .format (line)) # print the header of the current time: "t = ..."
                self.t = int(splitted_line[2])
                if (self.is_first_slot):
                    self.first_t = self.t
                if (VERBOSE_DEMOGRAPHY in self.verbose):
                    # for poa in range (self.num_of_PoAs): 
                    #     self.joined_poa        [poa]  .append(np.int16(0))
                    #     self.left_poa          [poa]  .append(np.int16(0))
                    #     self.joined_poa_sim_via[poa]  .append(np.int16(0))
                    for cell in range (self.num_of_cells): 
                        self.joined_cell [cell].append(np.int16(0))
                        self.left_cell   [cell].append(np.int16(0))
                continue

            elif (splitted_line[0] == 'usrs_that_left:'):
                if (VERBOSE_POA in self.verbose):
                    printf(self.poa_file, '{}\n' .format (line))
                ids_of_usrs_that_left_poa = [int(usr_id) for usr_id in splitted_line[1:] if usr_id!= '']                
                if (VERBOSE_DEMOGRAPHY in self.verbose):
                    for usr in list (filter (lambda usr: usr['id'] in ids_of_usrs_that_left_poa, self.usrs)): # for each usr that left
                        self.left_cell    [usr['cur cell']][-1]    += 1 # increase the cntr of the usrs that left from the cur cell of that usr at this cycle
                        self.left_cell_to [usr['cur cell']]['out'] += 1 # inc the cntr of the # of veh left this cell to outside the sim (counting along the whole sim, not per cycle)
                self.usrs = list (filter (lambda usr : (usr['id'] not in ids_of_usrs_that_left_poa), self.usrs))
                continue
    
            elif (splitted_line[0] == 'new_or_moved:'): 
                splitted_line = splitted_line[1:] # the rest of this line details the locations of users that are either new, or old (existing) users who moved during the last time slot
                if (splitted_line !=['']): # Is there a non-empty list of vehicles that are old / new / recycled?  

                    splitted_line = splitted_line[0].split (')') # split the line into the data given for each distinct usr
                    for my_tuple in splitted_line:  
                        if (len(my_tuple) <= 1): # no more new vehicles in this list. 
                            break
                        my_tuple = my_tuple.split("(")
                        my_tuple   = my_tuple[1].split (',')
                        usr_id   = np.uint16(my_tuple[veh_id_idx])
                        x, y = float(my_tuple[x_pos_idx]), float(my_tuple[y_pos_idx])
                        if (VERBOSE_FIND_AREA in self.verbose):
                            min_x_pos_found = min (x, min_x_pos_found)
                            min_y_pos_found = min (y, min_y_pos_found)
                            max_x_pos_found = max (x, max_x_pos_found)
                            max_y_pos_found = max (y, max_y_pos_found)
                        nxt_poa   = self.loc2poa (x,y)
                        nxt_cell = self.poa2cell (nxt_poa)
                        if (VERBOSE_DEBUG in self.verbose):
                            if (nxt_poa not in range(self.num_of_PoAs)):
                                print ('Error: t = {} usr={}, nxt_poa={}, pos=({},{}), MAX_X={}, MAX_Y={}. num_of_poas={} ' .format (self.t, usr_id, nxt_poa, x, y, MAX_X['Lux'], MAX_Y['Lux'], self.num_of_PoAs))
                                print ('Calling loc2poa again for deubgging')
                                nxt_poa = self.loc2poa (float(x), float(y))
                                nxt_cell = self.poa2cell (nxt_poa)
                                exit ()
                        
                        if (my_tuple[type_idx] == 'n'): # new vehicle
                            self.usrs.append ({'id' : usr_id, 'cur poa' : nxt_poa, 'nxt poa' : nxt_poa, 'nxt cell' : nxt_cell, 'new' : True}) # for a new usr, we mark the cur_poa same as nxt_poa 
                            if (VERBOSE_DEMOGRAPHY in self.verbose): 
                                # self.joined_poa_sim_via[nxt_poa][-1] += 1 # inc the # of usrs that joined the sim' via this cell
                                # self.joined_poa        [nxt_poa][-1]   += 1 # inc the # of usrs that joined this PoA at this slot
                                self.joined_cell      [nxt_cell][-1] += 1 # inc the # of usrs that joined this cell at this slot

                        elif (my_tuple[type_idx] == 'o'): # recycled vehicle's id, or an existing user, who moved
                            list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, self.usrs))
                            if (len(list_of_usr) == 0):
                                print  ('Error at t={}: input file={}. Did not find old / recycled usr {}' .format (self.t, self.usrs_loc_file_name, usr_id))
                                exit ()
                            list_of_usr[0]['nxt poa']   = nxt_poa # list_of_usr[0] is the old usr who moved.
                            list_of_usr[0]['nxt cell'] = nxt_cell
                            cur_cell                   = list_of_usr[0]['cur cell']
                            if (VERBOSE_DEMOGRAPHY in self.verbose and nxt_poa!= cur_cell): #this user moved to another cell  
                                # self.joined_poa[nxt_poa][-1]                   += 1 # inc the # of usrs that joined this PoA
                                # self.left_poa  [list_of_usr[0]['cur poa']][-1] += 1 # inc the # of usrs that left the previous cell of that usr
                                self.joined_cell  [nxt_cell][-1]               += 1 # inc the # of usrs who joined this cell
                                self.left_cell    [cur_cell] [-1]              += 1 # inc the # of usrs who left this cell at this cycle
                                
                                
                                direction = self.direction_of_mv (cur_cell, nxt_cell)
                                if (direction == -1): # error 
                                    printf (self.demography_file, '\\\ Error at t={}. usr_id={}. x={:.0f}, y={:.0f}, cur_cell={}, nxt_poa={}, nxt_cell={}\n' .format(self.t, usr_id, x, y, cur_cell, nxt_poa, nxt_cell))
                                else:
                                    self.left_cell_to [cur_cell][direction] += 1 # increase the cntr of usrs who left this cell to the relevant direction
                        else:
                            print ('Wrong type of usr:{}' .format (my_tuple[type_idx]))
                            exit () 
                        if (VERBOSE_SPEED in self.verbose):
                            self.speed[nxt_poa] = {'speed' : (float(my_tuple[speed_idx]) + self.speed[nxt_poa]['num of smpls'] * self.speed[nxt_poa]['speed'])/(self.speed[nxt_poa]['num of smpls'] + 1), 
                                                  'num of smpls' : self.speed[nxt_poa]['num of smpls'] + 1}
                
                # At this point we finished handling all the usrs (left / new / moved) reported by the input ".loc" file at this slot. So now, output the data to "..poa" file, and/or to a file, counting the vehicles at each cell
                if (VERBOSE_POA in self.verbose):
                    self.print_usrs_poa() # Print the PoAs of the users 
                if (VERBOSE_CNT in self.verbose):
                    self.cnt_num_of_vehs_per_poa ()
                for usr in self.usrs: # mark all existing usrs as old
                    usr['new']      = False
                    usr['cur poa']  = usr ['nxt poa']
                    usr['cur cell'] = usr ['nxt cell']
                self.is_first_slot = False
        # print ('min_x_pos_found=', min_x_pos_found, 'max_x_pos_found=', max_x_pos_found, 'min_y_pos_found=', min_y_pos_found, 'max_y_pos_found=', max_y_pos_found)
    
    def print_demography_diagram (self):
        """
        print a detailed demographic diagram, detailing the number of vehs left from each cell to each direction.
        The directions to which a veh can move are 'n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se', and 'out'.
        'out' indicates that a car left the simulated area.
        """
        printf (self.demography_file, '\\\ Demography diagrams\n')
        printf (self.demography_file, '\\\ Showing for each rectangle the average number of cars left to each direction, at each slot\n')
        printf (self.demography_file, '\\\ The number in the middle show the average number of cars left the simulated area from this rectangle\n\n\n')
        
        
        diagram_val = np.array ([self.left_cell_to[self.cell2tile[i]] for i in range (len(self.cell2tile))]).reshape ( [self.sqrt_num_of_cells, self.sqrt_num_of_cells])
        
        for row in reversed(range(len(diagram_val))): # use reversed order, as we'd like to begin with the rectangle corresponding to the northest areas, which have the largest row indices.
            for col in range(len(diagram_val[0])):
                printf (self.demography_file, '  (row,col)=({},{})\t\t\t\t' .format (row, col))
            printf (self.demography_file, '\n')
            for col in range(len(diagram_val[0])):
                printf (self.demography_file, '{:.2f}\t{:.2f}\t{:.2f}\t\t\t' .format (
                        diagram_val[row][col]['sw'] / self.sim_len,
                        diagram_val[row][col]['s']  / self.sim_len,
                        diagram_val[row][col]['se'] / self.sim_len))
            printf (self.demography_file, '\n')
            for col in range(len(diagram_val[0])):
                printf (self.demography_file, '{:.2f}\t{:.2f}\t{:.2f}\t\t\t' .format (
                        diagram_val[row][col]['w']   / self.sim_len,
                        diagram_val[row][col]['out'] / self.sim_len,
                        diagram_val[row][col]['e']   / self.sim_len))
            printf (self.demography_file, '\n')
            for col in range(len(diagram_val[0])):
                printf (self.demography_file, '{:.2f}\t{:.2f}\t{:.2f}\t\t\t' .format (
                        diagram_val[row][col]['nw'] / self.sim_len,
                        diagram_val[row][col]['n']  / self.sim_len,
                        diagram_val[row][col]['ne'] / self.sim_len))
            printf (self.demography_file, '\n')
            for col in range(len(diagram_val[0])):
                printf (self.demography_file, '  total\t{:.2f}\t\t\t\t\t' .format (
                        sum ([diagram_val[row][col][direction] for direction in directions]) / self.sim_len)),
            printf (self.demography_file, '\n\n')
    
    
    def post_processing (self):
        """
        Post processing after finished parsing all the input file(s).
        The post processing may include:
        - Adding some lines to the output .poa file.
        - Plot the num_of_vehs 
        """
        self.sim_len = self.t - self.first_t + 1
        if (VERBOSE_POA in self.verbose):
            printf(self.poa_file, "\n")   
        if (VERBOSE_CNT in self.verbose):
            # self.plot_num_of_vehs_per_poa_graph ()
            self.plot_num_of_vehs_in_cell_heatmaps()
            # self.plot_num_of_vehs_per_PoA()
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.print_demography_diagram ()
            self.plot_demography_heatmap()
        if (VERBOSE_SPEED in self.verbose):
            
            # first, fix the speed, as we assumed a first veh with speed '0'.
            for poa in [poa for poa in range (self.num_of_PoAs) if (self.speed[poa]['num of smpls'] > 0)]:
                self.speed[poa]['speed'] = self.speed[poa]['speed'] * (self.speed[poa]['num of smpls'] +1) / self.speed[poa]['num of smpls']
            self.print_speed()
            self.plot_speed_heatmap()
     
    def print_intermediate_res (self): 
        """
        Print the current aggregate results; used for having intermediate results when running long simulation.
        """
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_file_name, 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.num_of_vehs_output_file, '// after parsing the file {}\n' .format (self.usrs_loc_file_name))
            for poa in range (self.num_of_PoAs):
                printf  (self.num_of_vehs_output_file, 'num_of_vehs_in_poa_{}: ' .format (poa))
                # print (self.num_of_vehs_in_poa[poa])
                print ('{}' .format (self.num_of_vehs_in_poa[poa]), file = self.num_of_vehs_output_file, flush = True)
            self.calc_num_of_vehs_per_cell()
            for cell in range (self.num_of_cells):
                printf  (self.num_of_vehs_output_file, 'num_of_vehs_in_cell_{}:' .format (cell))
                printar (self.num_of_vehs_output_file, self.num_of_vehs_in_cell[cell])
        # if (VERBOSE_DEMOGRAPHY in self.verbose): 
        #     printf (self.demography_file, '// after parsing {}\n' .format (self.usrs_loc_file_name))
        #     self.print_demography()                
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
        2. output the PoAs of all new/left/moved users at each time slot.
        The exact behavior is by the value of self.verbose
        """
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_file_name = '../res/num_of_vehs_{}_{}.txt' .format (loc_file_names[0], self.antloc_file_name)
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_file_name, 'w+')
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.demography_file = open ('../res/demography_{}_{}.txt' .format(loc_file_names[0].split('.')[0], self.num_of_cells), 'w+')
        if (VERBOSE_POA in self.verbose):
            self.poa_file_name = self.input_files_str (loc_file_names[0]) + '.poa'
            self.poa_file      = open ("../res/ap_files/" + self.poa_file_name, "w+")  
            if (self.use_rect_PoA_cells):
                printf (self.poa_file, '// Using rectangle cells\n')
            else:
                printf (self.poa_file, '// .antloc file={}\n' .format (self.antloc_file_name))

            printf (self.poa_file, '// File format:\n//for each time slot:\n')
            printf (self.poa_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (self.poa_file, '//"new_usrs" is a list of the new usrs, and their PoAs, e.g.: (0, 2)(1,3) means that new usr 0 is in cell 2, and new usr 1 is in cell 3.\n')
            printf (self.poa_file, '//"old_usrs" is a list of the usrs who moved to another cell in the last time slot, and their current PoAs, e.g.: (0, 2)(1,3) means that old usr 0 is now in cell 2, and old usr 1 is now in cell 3.\n')
        
        print ('Begin parsing files with max_power_of_4={}' .format (self.max_power_of_4))
        self.is_first_slot = True
        for file_name in loc_file_names: 
            self.usrs_loc_file_name = file_name
            self.usrs_loc_file      = open ('../res/loc_files/' + self.usrs_loc_file_name,  "r")
            print ('parsing file {}' .format (self.usrs_loc_file_name)) 
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
        avg_num_of_vehs_per_poa = np.array ([np.average(self.num_of_vehs_in_poa[poa]) for poa in range(self.num_of_PoAs)]) 
        for lvl in range (self.max_power_of_4):
            printf (output_file, '\nlvl {}\n********************************\n' .format(lvl))
            self.calc_cell2tile (lvl) # call a function that translates the number as "tile" to the ID of the covering PoA.
            self.print_as_sq_mat (output_file, self.vec2heatmap (avg_num_of_vehs_per_poa))
            reshaped_heatmap = avg_num_of_vehs_per_poa.reshape (int(len(avg_num_of_vehs_per_poa)/4), 4) # prepare the averaging for the next iteration
            avg_num_of_vehs_per_poa = np.array([np.sum(reshaped_heatmap[i][:])for i in range(reshaped_heatmap.shape[0])], dtype='int') #perform the averaging, to be used by the ext iteration.

    def calc_num_of_poas_per_cell (self):
        """
        Returns the number of poas in each cell
        """
        num_of_poas_in_cell = np.zeros (self.num_of_cells)
        
        for poa in range(self.num_of_PoAs):
            num_of_poas_in_cell[self.poa2cell(poa)] += 1
        
        return num_of_poas_in_cell

    def rd_num_of_vehs_per_poa_n_cell (self, input_file_name):
        """
        Read the number of vehicles at each poa, and each cell, as written in the input files. 
        """
        input_file  = open ('../res/' + input_file_name, "r")  
    
        self.num_of_vehs_in_poa   = [] #[[] for _ in range (self.num_of_PoAs)]   # self.num_of_vehs_in_poa[i][j] will hold the # of vehs in poa i in time slot j 
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
            if (vec_name[4] == 'poa'): # the vector's name begins by "num_of_vehs_in_poa"
                self.num_of_vehs_in_poa.append(num_of_vehs)
            else: # Now we know that the vector's name begins by "num_of_vehs_in_cell"
                self.num_of_vehs_in_cell.append(num_of_vehs)


    def plot_voronoi_diagram (self):
        """
        Plot a Voronoi diagram of the PoAs in self.list_of_PoAs
        """
        points = np.array ([[poa['x'], poa['y']] for poa in self.list_of_PoAs])
        
        voronoi_plot_2d(Voronoi(points), show_vertices=False)
        plt.xlim(0, MAX_X[self.city]); plt.ylim(0, MAX_Y[self.city])
        plt.show()

if __name__ == '__main__':

    max_power_of_4 = 4
    my_loc2poa      = loc2poa_c (max_power_of_4 = max_power_of_4, verbose = [VERBOSE_CNT], antloc_file_name = '', city='Monaco') #Monaco.Monaco_Telecom.antloc', city='Monaco') #'Lux.center.post.antloc')
    # my_loc2poa.plot_voronoi_diagram()
    
    # Processing
    my_loc2poa.parse_loc_files (['Monaco_0730_0830_60secs.loc']) #(['Monaco_0730_0830_60secs.loc']) #(['Lux_0730_0740_1secs.loc', 'Lux_0740_0750_1secs.loc', 'Lux_0750_0800_1secs.loc', 'Lux_0800_0810_1secs.loc', 'Lux_0810_0820_1secs.loc', 'Lux_0820_0830_1secs.loc']) #'0730_0830_8secs.loc']) #(['0829_0830_8secs.loc' '0730_0830_8secs.loc']) #'0730_0830_8secs.loc'  (['0730.loc', '0740.loc', '0750.loc', '0800.loc', '0810.loc', '0820.loc'])  #['Lux_0829_0830_1secs.loc']
    # my_loc2poa.plot_num_of_vehs_in_cell_heatmaps( )
    
    # # Post-processing
    # my_loc2poa.rd_num_of_vehs_per_poa_n_cell ('num_of_vehs_Lux.center.post.antloc_1524poas.txt')# ('num_of_vehs_per_poa_256aps_ant.txt')
    # # my_loc2poa.plot_num_of_vehs_per_PoA (usrs_loc_file_name='0730_0830_8secs.loc')
    # my_loc2poa.plot_num_of_vehs_in_cell_heatmaps (usrs_loc_file_name='0829_0830_8secs.loc')