import math, pickle
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from printf import printf, printmat, invert_mat_bottom_up, printFigToPdf

# from matplotlib.co//lors import LogNorm, Normalize
# from matplotlib.ticker import MaxNLocator
# import itertools 

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

# x,y indexes of the south-west corner of the simulated area
LOWER_LEFT_CORNER = {'Lux'   : np.array ([GLOBAL_MAX_X['Lux']//4,   GLOBAL_MAX_Y['Lux']//4], dtype='int16'),   # 3405, 2864.25
                    'Monaco' : np.array ([2350, 2050], dtype='int16')}                  

# x,y indexes of the north-east corner of the simulated area
UPPER_RIGHT_CORNER = {'Lux'   : np.array ([GLOBAL_MAX_X['Lux']*3//4, GLOBAL_MAX_Y['Lux']*3//4], dtype='int16'), # 10216.5, 8593  
                    'Monaco' : np.array ([5450, 3050], dtype='int16')} 

# The 4 corners of the absolute (non-rotated) position of the simulated area. 
# The order is: upper-left, upper-rigth, lower-right, lower-left
SIMULATED_AREA_RECT = {'Lux': [ (GLOBAL_MAX_X['Lux']//4, GLOBAL_MAX_Y['Lux']*3//4), # upper-left  corner
                       UPPER_RIGHT_CORNER['Lux'],                                   # upper-right corner
                       (GLOBAL_MAX_X['Lux']*3//4, GLOBAL_MAX_Y['Lux']//4),          # lower-right corner
                       LOWER_LEFT_CORNER['Lux']],                                   # lower-left  corner
                       'Monaco' : [ (3541, 967),  # upper-left  corner
                                    (5363, 3475), # upper-right  corner
                                    (6171, 2887), # lower-right  corner                                     
                                    (4349, 380)   # lower-left  corner
                                     ]}

upper_left_idx, upper_right_idx, lower_right_idx, lower_left_idx = 0,1,2,3 # indices for upper-left, upper-right, lower-right, lower-left corners in the simulated area

# vectors for the X, Y edges of the simulated area.
X_EDGE = {'Lux'    : np.array(SIMULATED_AREA_RECT['Lux']   [lower_right_idx]) - np.array(SIMULATED_AREA_RECT['Lux']   [lower_left_idx]), 
          'Monaco' : np.array(SIMULATED_AREA_RECT['Monaco'][lower_right_idx]) - np.array(SIMULATED_AREA_RECT['Monaco'][lower_left_idx])}
Y_EDGE = {'Lux'    : np.array(SIMULATED_AREA_RECT['Lux']   [upper_left_idx])  - np.array(SIMULATED_AREA_RECT['Lux']   [lower_left_idx]), 
          'Monaco' : np.array(SIMULATED_AREA_RECT['Monaco'][upper_left_idx])  - np.array(SIMULATED_AREA_RECT['Monaco'][lower_left_idx])}
                                                                                     
# maximal allowed x,y values for the simulated area (which is possibly only a part of the full city area)
MIN_X = {'Lux' : 0, 'Monaco' : 0}
MIN_Y = {'Lux' : 0, 'Monaco' : 0}
MAX_X = {'Lux'    : UPPER_RIGHT_CORNER['Lux']   [0] - LOWER_LEFT_CORNER['Lux']   [0],
         'Monaco' : UPPER_RIGHT_CORNER['Monaco'][0] - LOWER_LEFT_CORNER['Monaco'][0]}
MAX_Y = {'Lux'    : UPPER_RIGHT_CORNER['Lux']   [1] - LOWER_LEFT_CORNER['Lux']   [1],
         'Monaco' : UPPER_RIGHT_CORNER['Monaco'][1] - LOWER_LEFT_CORNER['Monaco'][1]}

HEATMAP_NUM_VEHS_VMAX  = {'Lux' : 800, 'Monaco' : 3000}
HEATMAP_VEHS_LEFT_VMAX = {'Lux' : 5.5, 'Monaco' : 2.5}

# Indices of the various field within the input '.loc' file 
type_idx   = int(0) # type of the vehicle: either 'n' (new veh, which has just joined the sim), or 'o' (old veh, that moved). 
veh_id_idx = int(1)
x_pos_idx  = int(2)
y_pos_idx  = int(3)
speed_idx  = int(4)

# In Monaco we partition the city area into several horizontal almost-square rectangles before further iteratively partitioning it into squares
NUM_OF_TOP_LVL_SQS = {'Lux' : 1, 'Monaco' : 3}

HEATMAP_FONT_SIZE = 17 

# # types of vehicles' id
# new      = 0 # a new (unused before) vehicle's id
# old      = 1 # a vehicle that already exists in the sim
# recycled = 2 # a recycled id, namely, a new vehicle using an id of an old vehicle, who left.  

directions = ['s', 'n', 'e', 'w', 'se', 'sw', 'ne', 'nw', 'out']


# Checks whether a given position is within the simulated area of a given city
# Input: city, and [x,y] position
# Output: True iff this vehicle is within the simulated area of this city.
is_in_simulated_area = lambda city, position : False if (position[0] <= MIN_X[city] or position[0] >= MAX_X[city] or position[1] <= MIN_Y[city] or position[1] >= MAX_Y[city]) else True

# Checks whether the given (x,y) position is within the global area of the given city.
# Input: city, and [x,y] position
# Output: True iff this position is within the global area of that city.
is_in_global_area = lambda city, position : False if (position[0] <= 0 or position[0] >= GLOBAL_MAX_X[city] or position[1] <= 0 or position[1] >= loc2poa_c.GLOBAL_MAX_Y[city]) else True

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
    
    # Map a given x,y position to a PoA (Point of Access).
    # If the input includes real PoA locations, the mapping is by Voronoi distance, namely, each client is mapped to the nearest poa.
    # Else, use a partition of the area into uniform-size rectangular cells.
    loc2poa = lambda self, x, y : self.loc2cell_using_rect_cells (x, y, max_power_of_4=self.max_power_of_4) if self.use_rect_PoA_cells else self.nearest_poa (x,y)
     
    # calculate the respective cell when using rectangular cell.
    # This is done by calling loc2cell_using_squarlets with the relative position w.r.t. the concrete squarlet's boarder, and adding the number of cells in more western squarlets. 
    loc2cell_using_rect_cells = lambda self, x, y, max_power_of_4 : self.loc2cell_using_squarlets (x = x % self.x_edge_of_squarlet, y = y, max_x = self.x_edge_of_squarlet, max_y=self.max_y) + (x // self.x_edge_of_squarlet)*4**self.max_power_of_4 
    
    # inline function for formatted-printing the PoA of a single user
    print_usr_poa = lambda self, usr: printf(self.poa_file, "({},{:.0f})" .format (usr['id'], usr['nxt poa']))   
    
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

    # rotate a given point by self.angle radians counter-clockwise around self.pivot
    rotate_point = lambda self, point : [int (self.pivot[0] + math.cos(self.angle) * (point[0] - self.pivot[0]) - math.sin(self.angle) * (point[1] - self.pivot[1])),
                                         int (self.pivot[1] + math.sin(self.angle) * (point[0] - self.pivot[0]) + math.cos(self.angle) * (point[1] - self.pivot[1]))]
     
    # Given the x,y position, return the x,y position within the simulated area (city center) 
    pos_to_relative_pos = lambda self, pos: np.array(pos, dtype='int16') - LOWER_LEFT_CORNER [self.city]

    edges_of_smallest_rect = lambda self : [(self.max_x/NUM_OF_TOP_LVL_SQS[self.city]) / (2**self.max_power_of_4), self.max_y / (2**self.max_power_of_4)]

    # Parse the number of rectangles, given some settings.
    find_num_of_rects = lambda self, string : int(string.split('rects')[0].split('_')[-1])
    
    def gen_heatmap (self, df, vmax=None, pcl_input_file_name=None, plot_colorbar=True): 
        """
        Generate a well-customized sns heatmap from the given data frame.
        Input: df - the data frame for the heatmap.
               However, when given a pickle input file name, read the df from this input file, rather than from the "df" input
               vmax - maximum value in the heatmap (optional input) 
        Output: the heatmap
        """
        
        if (pcl_input_file_name != None):
            df = pd.read_pickle(r'../res/{}' .format (pcl_input_file_name))
        plt.rcParams.update({'font.size': HEATMAP_FONT_SIZE})
        if (vmax==None):
            if (plot_colorbar): 
                if (self.city=='Lux'): 
                    my_heatmap = sns.heatmap(df, vmin=0, cmap="YlGnBu", linewidths=0.1, xticklabels=False, yticklabels=False, cbar=True)
                else:
                    my_heatmap = sns.heatmap(df, vmin=0, cmap="YlGnBu", linewidths=0.1, xticklabels=False, yticklabels=False, cbar=True, 
                                             cbar_kws={"shrink": 0.35, 'aspect' : 10, 'ticks' : [0, 40, 80, 120]}) # vmax=df.values.max()
            else: 
                my_heatmap = sns.heatmap(df, vmin=0, cmap="YlGnBu", linewidths=0.1, xticklabels=False, yticklabels=False, cbar=False) 
        else:
            if (plot_colorbar): 
                if (self.city=='Lux'): 
                    my_heatmap = sns.heatmap(df, vmin=0, vmax=vmax, cmap="YlGnBu", linewidths=0.1, xticklabels=False, yticklabels=False, cbar=True) 
                else: 
                    my_heatmap = sns.heatmap(df, vmin=0, cmap="YlGnBu", linewidths=0.1, xticklabels=False, yticklabels=False, cbar=True, 
                                             cbar_kws={"shrink": 0.35, 'aspect' : 10, 'ticks' : [0, 1, 2]}) # vmax=df.values.max()
            else: 
                my_heatmap = sns.heatmap(df, vmin=0, vmax=vmax, cmap="YlGnBu", linewidths=0.1, xticklabels=False, yticklabels=False, cbar=False) 
        # my_heatmap.figure.axes[-1].yaxis.label.set_size(30)

        my_heatmap.set_aspect("equal") # Keep each rectangle as square 
        my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top

        return my_heatmap

    # def gen_heatmap_series (self, pcl_input_file_names, pcl_lanes_len_input_file_name=None): 
    #     """
    #     Generate a series of 4 heatmaps, using the given pcl input files for the data.
    #     Inputs: 
    #     pcl_input_file_names - a vector of input pickle file names.
    #     pcl_lanes_len_input_file_name - when given, should divide each value read from pcl_input_file_names by the respective value in pcl_lanes_len_input_file_name.
    #     """
    #
    #     if (len (pcl_input_file_names) != 4):
    #         print ('Error: gen_heatmap_series needs 4 input pickle file names in pcl_input_file_names')
    #         return
    #
    #     # Read the lenghts of lanes in each rectangle from the relevant input .pcl file name, if exists.
    #     if (pcl_lanes_len_input_file_name!=None):
    #         tot_len_of_lanes = pd.read_pickle (r'../res/{}' .format(pcl_lanes_len_input_file_name))
    #
    #     fig, axn = plt.subplots(2, 2, sharex=False, sharey=False)
    #
    #     cbar_ax = fig.add_axes([.91, .3, .03, .4])
    #
    #     for i, ax in enumerate(axn.flat):
    #         if (pcl_lanes_len_input_file_name==None): # no need to normalize by the lanes' lengths
    #             df = pd.read_pickle(r'../res/{}' .format (pcl_input_file_names[i]))
    #         else:
    #             vec = tot_len_of_lanes[i+1]
    #             print (vec) 
    #             df = pd.read_pickle(r'../res/{}' .format (pcl_input_file_names[i]))
    #         my_heatmap = sns.heatmap(df, ax=ax, cmap="YlGnBu", linewidths=0.1, xticklabels=False, yticklabels=False, vmin=0, vmax=HEATMAP_NUM_VEHS_VMAX[self.city], 
    #                     cbar=i == 0, cbar_ax=None if i else cbar_ax)
    #         my_heatmap.set_aspect("equal") # Keep each rectangle as square 
    #
    #     fig.tight_layout(rect=[0, 0, .9, 1])        
    #
    #     # plt.rcParams.update({'font.size': HEATMAP_FONT_SIZE})
    #     # my_heatmap = sns.heatmap(df, vmin=0, vmax=HEATMAP_NUM_VEHS_VMAX[self.city], cmap="YlGnBu", linewidths=0.1, xticklabels=False, yticklabels=False, cbar=True, heatmap=False) # vmax=df.values.max()
    #     # my_heatmap.figure.axes[-1].yaxis.label.set_size(30)
    #     #
    #     # my_heatmap.set_aspect("equal") # Keep each rectangle 
    #     # my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
    #     #
    #     # return my_heatmap

    def __init__(self, max_power_of_4=3, verbose=[], antloc_file_name='', city=''):
        """
        Init a "loc2poa_c" object.
        A loc2poa_c is used can read ".loc" files (files detailing the location of each veh over time), and output ".poa" files (files detailing the PoA assignment of each veh), and/or statistics 
        (e.g., number of vehs entering/levaing each cell, avg # of vehs in each cell, etc.).
        """

        self.verbose            = verbose      # verbose level - defining which outputs will be written
        self.debug              = False 
        self.antloc_file_name   = antloc_file_name
        
        if (antloc_file_name !='' ): 
            self.city = antloc_file_name.split('.')[0] # extract the city from the antennas' location file 
        elif (city != ''):
            self.city = city
        else:
            print ('Error: nor antenna location file, neither city was specified.')
            exit ()
            
        if (antloc_file_name=='' and VERBOSE_POA in self.verbose):
            print ('Error: VERBOSE_POA was requested, but no antloc file was specified')
            exit ()
            
        if (self.city=='Monaco' and max_power_of_4 > 3):
            print ('Error: you chose to run Monaco with max_power_of_4={}. We currently run Monaco with max_power_of_4 <= 3' .format (max_power_of_4))
            exit ()
                  
        self.max_x = MAX_X[self.city]
        self.max_y = MAX_Y[self.city] # borders of the simulated area, in meters
        self.x_edge_of_squarlet = self.max_x / NUM_OF_TOP_LVL_SQS[self.city]   
        self.usrs              = []
        self.use_rect_PoA_cells   = True if (self.antloc_file_name=='') else False

        self.max_power_of_4    = max_power_of_4
        self.num_of_cells      = 4**max_power_of_4 * NUM_OF_TOP_LVL_SQS[self.city]
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
            self.speed_file    = open ('../res/{}_speed.txt' .format (self.city), 'a')
            self.speed         = [{'speed' : 0, 'num of smpls' : 0} for _ in range(self.num_of_PoAs)] # self.speed[poa] will hold the avg. speed (averaging along the whole sim' time) in Point-of-Access poa 
            self.speed_in_slot = [] # self.speed[t] will hold the average speed (averaing over all the vehicles within the simulated area) at time slot t 
        self.calc_cell2tile (lvl=0) # calc_cell2tile translates the number as a "cell" to the ID in a vector, covering the same area as a tile
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.joined_poa         = [[] for _ in range(self.num_of_PoAs)] # self.joined_poa[i][j] will count the # of clients that joined PoA i at slot j
            # self.joined_cell        = [[] for _ in range(self.num_of_cells)] # self.joined_cell[i][j] will count the # of clients that joined cell i at slot j
            self.left_poa           = [[] for _ in range(self.num_of_PoAs)] # self.left_poa[i][j] will count the # of clients that left PoA i at slot j
            self.left_cell          = [[] for _ in range(self.num_of_cells)] # self.left_cell[i][j] will count the # of clients that left cell i at slot j
            self.joined_poa_sim_via = [[] for _ in range(self.num_of_PoAs)] # self.joined_poa_sim_via[i][j] will count the # of clients that left the sim at slot j, and whose last PoA in the sim was PoA i
            
            if (NUM_OF_TOP_LVL_SQS[self.city]==1): # Finding neighbors is currently supported only when there's a single squarlet
                self.calc_ngbr_rects ()
                self.left_cell_to = []
                for _ in range(self.num_of_cells):
                    self.left_cell_to.append ({'s' : 0, 'n' : 0, 'e' : 0, 'w' : 0, 'se' : 0, 'sw' : 0, 'ne' : 0, 'nw' : 0, 'out' : 0})
        
        edges_of_smallest_rect = self.edges_of_smallest_rect ()

        print ('Smallest rectangles is: {:.1f} x {:.1f}' .format (edges_of_smallest_rect[0], edges_of_smallest_rect[1]))
    
    def calc_ngbr_rects (self):
        """
        Used for debugging only  
        Find the 4 (north, south, east, west) neighbours of each cell (rect). If it's an edge cell, the tile number of the neighbor will be -1.
        Assumes that the simulated area includes exactly one squarlet, iteratively pratitioned to 4 quadrants.
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
    
        
    def parse_antloc_file (self, antloc_file_name, plot_poa_locs_heatmap=False):
        """
        - Parse an .antloc file (an .antloc file is a file containing the list of antennas, with their IDs and (x,y) position within the simulated area).
        - Generate an output .poa2cell file. A .poa2cell is a file showing which cell (rectangle area) each antenna belongs to. 
        """
        antloc_file = open ('../res/antennas_locs/' + antloc_file_name, 'r')
        
        self.use_rect_PoA_cells = False
        for line in antloc_file: 
        
            if (line == "\n" or line.split ("//")[0] == ""): # skip lines of comments and empty lines
                continue
            
            splitted_line = line.split (',')
            x = float(splitted_line[1])
            y = float(splitted_line[2])
            self.list_of_PoAs.append ({'id' : float(splitted_line[0]), 'x' : x, 'y' : y, 'cell' : self.loc2cell_using_rect_cells (x, y, max_power_of_4=self.max_power_of_4)}) # 'cell' is the rectangle of the simulated area in which this PoA is found
            
        self.num_of_PoAs = len (self.list_of_PoAs)
        
        poa2cell_file = open ('../res/poa2cell_files/{}_{}cells.poa2cell' .format(antloc_file_name, self.num_of_cells), 'w')
        printf (poa2cell_file, '// This file details the cell associated with each PoA.\n// Format: p c\n// Where p is the poa number, and c is the number of cell associated with it.\n')
        
        
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
        
    def loc2cell_using_squarlets (self, x, y, max_x=None, max_y=None, max_power_of_4=None):
        """
        Finding the PoA covering the user's area, assuming that the number of PoAs is a power of 4, and the area is iteratively partitioned into 4 identical squares.
        Input:  (x,y) location data; (max_x, max_y) - nort-east corner for the area to partition; the south-west corner is (0,0).
        Output: poa that covers this area
        """        
        max_x          = self.max_x           if (max_x == None)          else max_x
        max_y          = self.max_y           if (max_y == None)          else max_y
        max_power_of_4 = self.max_power_of_4  if (max_power_of_4 == None) else max_power_of_4

        poa = np.int8(0)
        x_offset, y_offset = x, y
        x_edge, y_edge = 0.5*max_x, 0.5*max_y
        for p in range (max_power_of_4):
            poa += 4**(max_power_of_4-1-p)*int(2 * (y_offset // y_edge) + x_offset // x_edge) 
            x_offset, y_offset = x_offset % x_edge, y_offset % y_edge   
            x_edge /= 2
            y_edge /= 2
        return int(poa)
    
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
        # printf (self.speed_file, '{}' .format ([self.speed[poa]['speed'] for poa in range(self.num_of_PoAs)]))
        del (self.speed_in_slot[0]) # Remove the first element, which may represent a non-stable warm-up                                         
        printf (self.speed_file, 'avg_speed_in_slot={}\navg_speed={}\n' .format (self.speed_in_slot, np.average(self.speed_in_slot)))
        plt.plot ([t*60 for t in range (len(self.speed_in_slot))], self.speed_in_slot)
        plt.xlim (0, 3600)
        printFigToPdf ('{}.spd' .format (self.usrs_loc_file_name))
        plt.show()
        
    def plot_demography_heatmap (self, input_demography_file_name=None, usrs_loc_file_name=None, plot_colorbar=False):
        """
        Plot heatmaps, showing the avg number of vehicles that joined/left each cell during the simulated period.
        The heatmap will be saved to a file. 
        The name of the heatmap file will contain the name given in the input usrs_loc_file_name (if the input is explicitized), or according to of self.usrs_loc_file_name (else).
        If input_demography_file_name==None, the demography data should already exist in self.left_cell. 
        Else, the demography data will be extracted by parsing the file named input_demography_file_name.  
        """
        
        file_name_suffix = '{}rects' .format (self.num_of_cells)       
        plt.figure()
        if (input_demography_file_name != None):
            self.parse_demongraphy_file(input_demography_file_name)

        avg_vehs_left_per_rect = np.array ([np.average(self.left_cell[cell]) for cell in range(self.num_of_cells)])
        
        # printf (open ('../res/demography.txt', 'a'), '{}_{}rects. avg vehs left per rect={:.2f}. max vehs left per rect={:.2f}\n' .format(usrs_loc_file_name, self.num_of_cells, np.average(avg_vehs_left_per_rect), np.max(avg_vehs_left_per_rect)))
         
        self.calc_cell2tile (lvl=0) # call a function that translates the number as "tile" to the ID of the covering PoA.
        heatmap_vals = self.vec2heatmap (avg_vehs_left_per_rect)
        self.gen_heatmap (df=pd.DataFrame (heatmap_vals), vmax=HEATMAP_VEHS_LEFT_VMAX[self.city], plot_colorbar=plot_colorbar) 
        usrs_loc_file_name = usrs_loc_file_name if (usrs_loc_file_name!=None) else self.usrs_loc_file_name 
        plt.savefig('../res/heatmap_vehs_left_{}rects.pdf' .format (usrs_loc_file_name.split('.txt')[0]), bbox_inches='tight')

        return 
        
        heatmap_txt_file = open ('../res/heatmap_vehs_left.txt', 'a') 
        printf   (heatmap_txt_file, 'num of rects = {}\n' .format ( self.num_of_cells))
        printmat (heatmap_txt_file, heatmap_vals, my_precision=2)
        # my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top

        columns = self.gen_columns_for_heatmap (lvl=0)
        plt.figure()
        sns.heatmap (pd.DataFrame (self.vec2heatmap (np.array ([np.average(self.joined_poa_sim_via[poa]) for poa in range(self.num_of_PoAs)])), columns=columns), cmap="YlGnBu")
        # plt.title ('avg num of vehs that joined the sim every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_vehs_joined_sim_via_{}.jpg' .format (file_name_suffix))
        
        plt.figure ()
        sns.heatmap (pd.DataFrame (self.vec2heatmap (np.array ([np.average(self.left_poa_sim_via[poa])   for poa in range(self.num_of_PoAs)])), columns=columns), cmap="YlGnBu")
        # plt.title ('avg num of vehs that left the sim every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_vehs_left_sim_via_{}.jpg' .format (file_name_suffix))
        
        
    def vec2heatmap (self, vec):
        """
        Order the values in the given vec so that they appear as in the geographical map of cells.
        """
        if (NUM_OF_TOP_LVL_SQS[self.city]==1): # When there's only one squarlet, all the mapping is a single square mat
            nrows = self.sqrt_len(vec)
            ncols = nrows
            if (len(vec) != len(self.cell2tile)): # The current mapping of cell2tile doesn't fit the number of rectangles in the given vec --> calculate a cell2tile mapping fitting the required len; currently generating heatmaps of other levels is supported only for lvl=0
                self.calc_cell2tile (lvl=self.max_power_of_4 - int(math.log2(nrows)))
        else:
            nrows = int (2**self.max_power_of_4)
            ncols = nrows * NUM_OF_TOP_LVL_SQS[self.city]
            if (nrows*ncols != len (self.cell2tile)):
                print ('Error in calculation. max_p_of_4={}, nrows={}, ncols={}, len(cell2tile)={}' .format(self.max_power_of_4, nrows, ncols, len(self.cell2tile)))
                exit ()
        heatmap_val = np.array ([vec[self.cell2tile[i]] for i in range (len(self.cell2tile))]).reshape ( [nrows, ncols])
        
        # Unfortunately, we write matrix starting from the smallest value at the top, while plotting maps letting the "y" (north) direction "begin" at bottom, and increase towards the top.
        # Hence, need to swap the matrix upside-down
        return invert_mat_bottom_up(heatmap_val)

    def set_usrs_loc_file_name (self, usrs_loc_file_name=''):
        if (hasattr (self, 'usrs_loc_file_name')): # The field self.usrs_loc_file_name is already defined
            return
        if (usrs_loc_file_name==''):
            print ('Please specify an existing usr loc file name')
            exit ()
        self.usrs_loc_file_name = usrs_loc_file_name
    
    def calc_num_of_vehs_in_cell_heatmap (self):
        """
        Generate a heatmap, detailing the avg. number of vehicles in each cell (averaging along the simulation period).
        The heatmap is organized so that northern entries will have a SMALLER Y values, so that it can be plotted 
        using the common geographic convention, where north appears above. 
        The heatmap values are dumpted to a .pcl file, and also printed to the output file self.num_of_vehs_output_file.   
        """

        heatmap_vals = self.vec2heatmap(self.avg_num_of_vehs_per_cell())
        # Dump the values of the heatmap into a .pcl file
        with open ('../res/' + self.num_of_vehs_output_file_name.split('.res')[0] + '.pcl', 'wb') as pcl_output_file:
            pickle.dump (heatmap_vals, pcl_output_file)
        printmat (self.num_of_vehs_output_file, heatmap_vals, my_precision=0)
        printf   (self.num_of_vehs_output_file, 'sum of vals in the heatmap={:.0f}, max_val in the heatmap={:.0f}' .format (np.sum(heatmap_vals), np.max(heatmap_vals)))
            
    
    def plot_num_of_vehs_in_cell_heatmap (self, pcl_num_vehs_input_file_name, pcl_lanes_len_input_file_name=None, plot_colorbar=True): 
        """
        Generate a Python heatmap, showing at each cell the average number of vehicles found at that cell, along the simulation.
        Inputs:
        pcl_num_vehs_input_file_name - 
            a list with names of .pcl files, in which there exist the data about the number of vehicles in each rectangle.
        pcl_lanes_len_input_file_name - 
            optional input: a file with the data of the total length of lanes in each cell. When this input is given,
            the output heatmap shows linear density, namely, the total num of vehs in each rectangle, divided by the total length of rectangles in that cell.
            If this input is not given, the heatmap will show merely the number of vehs in each rectangle.
        plot_colorbar - 
            When True, a colorbar legend will be printed in the right of the heatmap.
        Output: the heatmap figure is saved in a file, whose name is based on the already-set self.usrs_loc_file_name (if exists), or on the argument usrs_locs_file_name (otherwise).
        """        
        
        # Read the lenghts of lanes in each rectangle from the relevant input .pcl file name, if exists.
        if (pcl_lanes_len_input_file_name!=None):
            if (pcl_lanes_len_input_file_name.split ('_')[0] != self.city):
                print ('pcl_lanes_len_input_file_name={}, self.city={}' .format(pcl_lanes_len_input_file_name, self.city))
            tot_len_of_lanes = pd.read_pickle (r'../res/{}' .format(pcl_lanes_len_input_file_name))
        
        plt.figure()       
        
        # Generate a heatmap, and save it into a file
        num_of_rects = self.find_num_of_rects (pcl_num_vehs_input_file_name) # Extract the # of rectangles from the input file name 
        if (pcl_lanes_len_input_file_name==None): # No lanes len input file was given --> no need to normalize the hea
            self.gen_heatmap (df=pd.read_pickle(r'../res/{}' .format (pcl_num_vehs_input_file_name)), vmax=HEATMAP_NUM_VEHS_VMAX[self.city], plot_colorbar=plot_colorbar)
            plt.savefig('../res/{}_num_vehs_{}rects.pdf' .format (self.city, num_of_rects), bbox_inches='tight') 
        else: 
            list_of_entry = list (filter (lambda item : item['num_of_rects']==num_of_rects, tot_len_of_lanes))
            tot_len_of_lanes_at_this_lvl = np.array(list_of_entry[0]['tot_len_of_lanes_in_rect'])
            df = pd.read_pickle(r'../res/{}' .format (pcl_num_vehs_input_file_name)) #np.empty ( (len(tot_len_of_lanes_at_this_lvl), len(tot_len_of_lanes_at_this_lvl[0])) )
            for row in range (len (tot_len_of_lanes_at_this_lvl)):
                for col in range (len (tot_len_of_lanes_at_this_lvl[0])):
                    df[row][col] = df[row][col] / tot_len_of_lanes_at_this_lvl[row][col] if (tot_len_of_lanes_at_this_lvl[row][col]>=0.1) else None 
            self.gen_heatmap (df=df, plot_colorbar=plot_colorbar)
            plt.savefig('../res/{}_lin_density_{}rects.pdf' .format (self.city, num_of_rects), bbox_inches='tight') 
         
    def calc_num_vehs_to_slot (self, city):
        """
        Calculate for each number of vehicles a list of the time slots having this exact number of vehicles.
        Input: 
        city - either 'Lux', or 'Monaco'
        Output:
        a .pcl file, holding a list of dictionaries, where for each number of vehicles (key) there exist a list of the time slots having this exact number of vehicles (value). 
        """
        
        num_vehs_in_slot = np.array (pd.read_pickle(r'../res/{}_0730_0830_1secs_num_of_vehs.pcl' .format (city)), dtype='int16')

        num_vehs_in_slot_set = set (num_vehs_in_slot)

        for num_veh in num_vehs_in_slot_set: # for each distinct value of the number of vehicles
            indices = [i for i, x in enumerate(num_vehs_in_slot) if x == num_veh] 
            print ('slot of {} are' .format (num_veh))
            print (indices) 
         
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

    def plot_num_of_vehs_per_PoA (self, usrs_loc_file_name='', plot_colorbar=False):
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
            _ = self.gen_heatmap (pd.DataFrame (self.vec2heatmap (avg_num_of_vehs_per_PoA)), plot_colorbar=plot_colorbar)
            # my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
            # plt.title ('avg num of vehs per PoA')
            plt.savefig('../res/num_vehs_per_PoA_{}_{}_{}cells.jpg' .format (self.antloc_file_name, self.usrs_loc_file_name, int(self.num_of_cells/(4**lvl))))
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
        
    # def plot_speed_heatmap (self):
    #     """
    #     Plot a heatmap, showing the average speed of vehicles at each cell.
    #     """
    #     plt.figure()
    #     sns.heatmap = self.gen_heatmap (pd.DataFrame (self.vec2heatmap ([self.speed[poa]['speed'] for poa in range(self.num_of_PoAs)])))
    #     plt.title ('avg speed in each cell')
    #     plt.savefig('../res/heatmap_speed.jpg')
        
    def calc_cell2tile (self, lvl=0): 
 
        cell2tile_in_squarlet = self.calc_cell2tile_in_squarlet(lvl=0)
        # print ('cell2tile_in_squarlet=\n{}' .format (cell2tile_in_squarlet))
        self.cell2tile = cell2tile_in_squarlet 
        for sq_num in range (NUM_OF_TOP_LVL_SQS[self.city]-1): 
            self.cell2tile = np.concatenate ((self.cell2tile, (sq_num+1)*4**self.max_power_of_4+cell2tile_in_squarlet), axis=1)

        self.cell2tile = np.asarray(self.cell2tile).reshape(-1) 

        # for demography verbose, we need also the other direction, which maps a given cell to its physical location in the tile. 
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.tile2cell = np.empty (self.num_of_tiles, dtype = 'uint8')
            for cell in range (self.num_of_cells):
                self.tile2cell[self.cell2tile[cell]] = cell
                
    def calc_cell2tile_in_squarlet (self, lvl=0):
        """
        prepare a translation of the cell id's to the tile (horizontal and vertical SHTI VA'EREV) rectangles, within a single squarlet.
        A squarlet is an almost-square area, which is iteratively paritioned into identically-sized squares.
        We assume that the squarlets are located horizontally, along the x axis.
        """
        max_x = self.max_x // NUM_OF_TOP_LVL_SQS[self.city] 
        max_y = self.max_y 
        
        # To calclate the tile, we calculate positions within each cell in the simulated area, and then call self.loc2cell_using_squarlets() to calculate the cell associated with this position. 
        max_power_of_4        = self.max_power_of_4 - lvl
        n                     = int(math.sqrt ( (self.num_of_cells/NUM_OF_TOP_LVL_SQS[self.city])/4**lvl))
        cell2tile_in_squarlet = np.empty (n**2, dtype = 'uint8')
        rect             = 0
        for y in range (max_y // (2*n), max_y, max_y // n): 
            for x in range (max_x // (2*n), max_x, max_x // n): 
                cell2tile_in_squarlet[rect] = self.loc2cell_using_squarlets(x, y, max_x, max_y, max_power_of_4) 
                rect+=1 
        return self.vec2sq_mat(cell2tile_in_squarlet)

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
            
 
    # def cnt_tot_vehs_per_slot (self, input_loc_file_name):
    #     """
    #     A fast run, which only count the number of vehicles in the whole simualated area in each slot, and its average.
    #     """
    #
    #     tot_num_of_vehs_in_slot = [0]
    #     for line in self.usrs_loc_file: 
    #
    #         # remove the new-line character at the end (if any), and ignore comments lines 
    #         line = line.split ('\n')[0] 
    #         if (line.split ("//")[0] == ""):
    #             continue
    #
    #         splitted_line = line.split (" ")
    #
    #         if (splitted_line[0] == "t"): # reached the next simulation time slot
    #
    #         elif (splitted_line[0] == 'usrs_that_left:'):
    #
    #
    #         elif (splitted_line[0] == 'new_or_moved:'): 
    #
    #             if (my_tuple[type_idx] == 'n'): # new vehicle

    
    def parse_file (self):
        """
        - Read and parse input ".loc" file, detailing the users locations 
        - Write the appropriate user-to-PoA connections to the file self.poa_file, or to files summing the number of vehicles per cell.
        """
        if (VERBOSE_FIND_AREA in self.verbose): # If this run is used for finding the maximal area in which car moves, init variables capturing the min,max of x,y found during the trace.
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
                if (VERBOSE_SPEED in self.verbose):
                    num_of_smpls_in_cur_slot = 0
                    avg_speed_in_cur_slot    = 0

                if (self.is_first_slot):
                    self.first_t = self.t
                if (VERBOSE_DEMOGRAPHY in self.verbose):
                    # for poa in range (self.num_of_PoAs): 
                    #     self.joined_poa        [poa]  .append(np.int16(0))
                    #     self.left_poa          [poa]  .append(np.int16(0))
                    #     self.joined_poa_sim_via[poa]  .append(np.int16(0))
                    for cell in range (self.num_of_cells): 
                        # self.joined_cell [cell].append(np.int16(0))
                        self.left_cell[cell].append(np.int16(0))
                continue
            
            elif (splitted_line[0] == 'usrs_that_left:'):
                if (VERBOSE_POA in self.verbose):
                    printf(self.poa_file, '{}\n' .format (line))
                ids_of_usrs_that_left_poa = [int(usr_id) for usr_id in splitted_line[1:] if usr_id!= '']                
                if (VERBOSE_DEMOGRAPHY in self.verbose):
                    for usr in list (filter (lambda usr: usr['id'] in ids_of_usrs_that_left_poa, self.usrs)): # for each usr that left
                        # if (usr['cur cell'] not in range (self.num_of_cells)): #$$$
                        #     print ('Err: cur cell=-1. t={}, usr.id={}' .format (self.t, usr['id'], ))
                        #     exit ()
                        # print ('usr cur cell={}' .format (usr['cur cell']))
                        # if (len (self.left_cell[usr['cur cell']])==0): #$$$
                        #     print ('warning: len=0. t={}, usr.id={}, usr.cur_cell={}' .format (self.t, usr['id'], usr['cur cell']))
                        #     continue
                        self.left_cell [usr['cur cell']][-1]    += 1 # increase the cntr of the usrs that left from the cur cell of that usr at this cycle
                        if (NUM_OF_TOP_LVL_SQS[self.city]==1): # Finding neighbors is currently supported only when there's a single squarlet
                            self.left_cell_to [usr['cur cell']]['out'] += 1 # inc the cntr of the # of veh left this cell to outside the sim (counting along the whole sim, not per cycle)
                self.usrs = list (filter (lambda usr : (usr['id'] not in ids_of_usrs_that_left_poa), self.usrs)) # Filter-out the users who left from the list of usrs
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
                        if (nxt_cell not in range (self.num_of_cells)): #$$$
                            print ('Error: t={}, usr_id={}, (x,y)=({},{}), nxt_cell={}' .format (self.t, usr_id, x, y, nxt_cell))
                            exit ()
                        if (VERBOSE_DEBUG in self.verbose):
                            if (nxt_poa not in range(self.num_of_PoAs)):
                                print ('Error: t = {} usr={}, nxt_poa={}, pos=({},{}), MAX_X={}, MAX_Y={}. num_of_poas={} ' .format (self.t, usr_id, nxt_poa, x, y, MAX_X['Lux'], MAX_Y['Lux'], self.num_of_PoAs))
                                print ('Calling loc2poa again for deubgging')
                                nxt_poa = self.loc2poa (float(x), float(y))
                                nxt_cell = self.poa2cell (nxt_poa)
                                exit ()
                        
                        if (VERBOSE_SPEED in self.verbose):
                            num_of_smpls_in_cur_slot += 1
                            avg_speed_in_cur_slot     = (avg_speed_in_cur_slot*(num_of_smpls_in_cur_slot-1) + float(my_tuple[speed_idx]))/num_of_smpls_in_cur_slot # Reservoir sampling for the speed in current slot  
                            # nxt_poa = int (nxt_poa)
                            # self.speed[nxt_poa] = {'speed' : (float(my_tuple[speed_idx]) + self.speed[nxt_poa]['num of smpls'] * self.speed[nxt_poa]['speed'])/(self.speed[nxt_poa]['num of smpls'] + 1), # implement a Reservoir sampling for the avg. speed among all vehicles in this PoA. 
                            #                       'num of smpls' : self.speed[nxt_poa]['num of smpls'] + 1}
                            
                        if (my_tuple[type_idx] == 'n'): # new vehicle
                            self.usrs.append ({'id' : usr_id, 'cur poa' : nxt_poa, 'nxt poa' : nxt_poa, 'nxt cell' : nxt_cell, 'new' : True}) # for a new usr, we mark the cur_poa same as nxt_poa 
                            # if (VERBOSE_DEMOGRAPHY in self.verbose): 
                                # self.joined_poa_sim_via[nxt_poa][-1] += 1 # inc the # of usrs that joined the sim' via this cell
                                # self.joined_poa        [nxt_poa][-1]   += 1 # inc the # of usrs that joined this PoA at this slot
                                # self.joined_cell      [nxt_cell][-1] += 1 # inc the # of usrs that joined this cell at this slot

                        elif (my_tuple[type_idx] == 'o'): # recycled vehicle's id, or an existing user, who moved
                            list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, self.usrs))
                            if (len(list_of_usr) == 0): # Didn't find this old usr in the list of currently running usrs
                                if (self.input_loc_file_is_rotated): # This indeed can happen in rotated files, where the simulated area changes
                                    self.usrs.append ({'id' : usr_id, 'cur poa' : nxt_poa, 'nxt poa' : nxt_poa, 'nxt cell' : nxt_cell, 'new' : True}) # for a new usr, we mark the cur_poa same as nxt_poa 
                                    # if (VERBOSE_DEMOGRAPHY in self.verbose): 
                                        # self.joined_poa_sim_via[nxt_poa][-1] += 1 # inc the # of usrs that joined the sim' via this cell
                                        # self.joined_poa        [nxt_poa][-1]   += 1 # inc the # of usrs that joined this PoA at this slot
                                        # self.joined_cell      [nxt_cell][-1] += 1 # inc the # of usrs that joined this cell at this slot
                                else: 
                                    print  ('Error at t={}: input file={}. Did not find old / recycled usr {}' .format (self.t, self.usrs_loc_file_name, usr_id))
                                    exit ()
                            else: # Found this usr in the list of existing usrs. So update its location and movement at the databases.
                                list_of_usr[0]['nxt poa']  = nxt_poa # list_of_usr[0] is the old usr who moved.
                                list_of_usr[0]['nxt cell'] = nxt_cell
                                cur_cell                   = list_of_usr[0]['cur cell']
                                if (VERBOSE_DEMOGRAPHY in self.verbose and nxt_poa!= cur_cell): #this user moved to another cell  
                                    # self.joined_poa[nxt_poa][-1]                   += 1 # inc the # of usrs that joined this PoA
                                    # self.left_poa  [list_of_usr[0]['cur poa']][-1] += 1 # inc the # of usrs that left the previous cell of that usr
                                    # self.joined_cell  [nxt_cell][-1] += 1 # inc the # of usrs who joined this cell
                                    self.left_cell    [cur_cell][-1] += 1 # inc the # of usrs who left this cell at this cycleW
                                
                                    if (NUM_OF_TOP_LVL_SQS[self.city]==1): # printing demography map when num_of_top_lvl_sqs>! is currently unsupported
                                        direction = self.direction_of_mv (cur_cell, nxt_cell)
                                        if (direction == -1): # error 
                                            printf (self.demography_file, '\\\ Error at t={}. usr_id={}. x={:.0f}, y={:.0f}, cur_cell={}, nxt_poa={}, nxt_cell={}\n' .format(self.t, usr_id, x, y, cur_cell, nxt_poa, nxt_cell))
                                        else:
                                            self.left_cell_to [cur_cell][direction] += 1 # increase the cntr of usrs who left this cell to the relevant direction

                        else:
                            print ('Wrong type of usr:{}' .format (my_tuple[type_idx]))
                            exit () 
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
                if (VERBOSE_SPEED in self.verbose):
                    self.speed_in_slot.append (avg_speed_in_cur_slot) 
                
        # print ('min_x_pos_found=', min_x_pos_found, 'max_x_pos_found=', max_x_pos_found, 'min_y_pos_found=', min_y_pos_found, 'max_y_pos_found=', max_y_pos_found)
    
    def print_demography_diagram (self):
        """
        print a detailed demographic diagram, detailing the number of vehs left from each cell to each direction.
        The directions to which a veh can move are 'n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se', and 'out'.
        'out' indicates that a car left the simulated area.
        """
        if (NUM_OF_TOP_LVL_SQS[self.city]>1): # Finding neighbors is currently supported only when there's a single squarlet
            print ('Sorry. Currently demography diagram are supported only when num_of_top_lvl_sqs==1')
            return
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
            self.plot_num_of_vehs_in_cell_heatmap()
            # self.plot_num_of_vehs_per_PoA()
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.print_demography()
            self.print_demography_diagram ()
            self.plot_demography_heatmap()
        if (VERBOSE_SPEED in self.verbose): 
            printf (self.speed_file, '// after parsing {}\n' .format (self.usrs_loc_file_name))                
            self.print_speed()
            # self.plot_speed_heatmap()
     
    def print_intermediate_res (self): 
        """
        Print the current aggregate results; used for having intermediate results when running long simulation.
        """
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_output_file_name, 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.num_of_vehs_output_file, '// after parsing the file {}\n' .format (self.usrs_loc_file_name))
            # for poa in range (self.num_of_PoAs):
            #     printf  (self.num_of_vehs_output_file, 'num_of_vehs_in_poa_{}: ' .format (poa))
            #     print ('{}' .format (self.num_of_vehs_in_poa[poa]), file = self.num_of_vehs_output_file, flush = True)
            self.calc_num_of_vehs_per_cell()
            # for cell in range (self.num_of_cells):
            #     printf  (self.num_of_vehs_output_file, 'num_of_vehs_in_cell_{}:' .format (cell))
            #     printar (self.num_of_vehs_output_file, self.num_of_vehs_in_cell[cell])
            peak_num_of_vehs_in_any_cell = max ([self.num_of_vehs_in_cell[cell][t] for cell in range(self.num_of_cells) for t in range(len(self.num_of_vehs_in_cell[0]))]  )
            edges_of_smallest_rect = self.edges_of_smallest_rect () 
            printf (self.num_of_vehs_output_file, '\n// maximal num of vehs in a rect at any time={}, rect size={:.1f} x {:.1f}, maximal car density={} \n' .format (
                    peak_num_of_vehs_in_any_cell, edges_of_smallest_rect[0], edges_of_smallest_rect[1], peak_num_of_vehs_in_any_cell/(edges_of_smallest_rect[0]*edges_of_smallest_rect[1])))
        # if (VERBOSE_DEMOGRAPHY in self.verbose): 
        #     printf (self.demography_file, '// after parsing {}\n' .format (self.usrs_loc_file_name))
        #     self.print_demography()                
    
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
        city_token_in_usr_loc_file_name = loc_file_names[0].split('_')[0]
        if (city_token_in_usr_loc_file_name != self.city):
            print ('Error: loc2poa_c initialized with city={}, but the input .loc file name is {}' .format (self.city, loc_file_names[0]))
            exit ()
        self.slot_len = int (loc_file_names[0].split('secs')[0].split('_')[-1])
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_output_file_name = '../res/{}_num_of_vehs_{}rects.txt' .format (loc_file_names[0], self.num_of_cells)
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_output_file_name, 'w+')
            
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            if (self.slot_len > 1): # Demography (usrs' mobility) should be measured only using the smallest time scale, namely, when the length of a simulation slot is 1. 
                print ('Cancelling demography verbose because slot len>1. slot_len={}' .format (self.slot_len)) 
                self.verbose.remove(VERBOSE_DEMOGRAPHY)
            self.demography_file = open ('../res/demography_{}_{}.txt' .format(loc_file_names[0].split('.')[0], self.num_of_cells), 'w+')
        
        if (VERBOSE_POA in self.verbose):
            self.poa_file_name = self.input_files_str (loc_file_names[0]) + '.poa'
            self.poa_file      = open ("../res/poa_files/" + self.poa_file_name, "w")  
            if (self.use_rect_PoA_cells):
                printf (self.poa_file, '// Using rectangle cells\n')
            else:
                printf (self.poa_file, '// .antloc file={}\n' .format (self.antloc_file_name))

            printf (self.poa_file, '// File format:\n//for each time slot:\n')
            printf (self.poa_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (self.poa_file, '//"new_usrs" is a list of the new usrs, and their PoAs, e.g.: (0, 2)(1,3) means that new usr 0 is in cell 2, and new usr 1 is in cell 3.\n')
            printf (self.poa_file, '//"old_usrs" is a list of the usrs who moved to another cell in the last time slot, and their current PoAs, e.g.: (0, 2)(1,3) means that old usr 0 is now in cell 2, and old usr 1 is now in cell 3.\n')
        
        print ('Parsing .loc files. verbose={}, max_power_of_4={}. antloc_file={}' .format (self.verbose, self.max_power_of_4, self.antloc_file_name))
        self.is_first_slot = True
        for file_name in loc_file_names: 
            self.usrs_loc_file_name = file_name
            self.usrs_loc_file      = open ('../res/loc_files/' + self.usrs_loc_file_name,  "r")
            print ('parsing file {}' .format (self.usrs_loc_file_name)) 
            self.input_loc_file_is_rotated = True if (len (file_name.split('rttd'))==2) else False
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

    def parse_density_file (self, input_file_name):
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

    def parse_demongraphy_file (self, input_file_name):
        """
        Read the number of vehicles at each poa, and each cell, as written in the input files. 
        """
        input_file  = open ('../res/' + input_file_name, "r")  
    
        self.left_cell = [] 
        
        for line in input_file:
    
            if (line == "\n" or line.split ("//")[0] == ""):
                continue
    
            self.left_cell.append([])
            line = line.split ("\n")[0]
            splitted_line = line.split (":")
            cell_num = int(splitted_line[0].split('_')[1]) 
            vec_data = splitted_line[1].split('[')[1].split(']')[0].split(', ')
            for num_of_vehs_in_this_time_slot in vec_data:
                self.left_cell[cell_num].append (int(num_of_vehs_in_this_time_slot))


    def plot_voronoi_diagram (self):
        """
        Plot a Voronoi diagram of the previously parsed .antloc (antennas-locations) files.
        The Voronoi diagram is saved as a .pdf into ../res/
        """
        points = np.array ([[poa['x'], poa['y']] for poa in self.list_of_PoAs])
        
        voronoi_plot_2d(Voronoi(points), show_vertices=False, show_points=False) # For showing the points, set show_points. May fix the points' size by point_size
        plt.xlim(0, MAX_X[self.city]); plt.ylim(0, MAX_Y[self.city])
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        printFigToPdf ('{}_Voronoi' .format (self.city))        
    # def rotate_loc_file (self, loc_file_names, angle=54):
    #     """
    #     Given an input .loc file, generate an output .loc file, where each location is rotated angle degrees clockwise w.r.t. the center of the simulated area.
    #     """
    #     if (self.city != 'Monaco'):
    #         print ('Error: currently, we rotate only Monaco')
    #     self.pivot = [GLOBAL_MAX_X[self.city]/2, GLOBAL_MAX_Y[self.city]/2] # pivot point, around which the rotating is done
    #     self.angle = -math.radians(angle) # convert the requested angle degrees of clcokwise rotating to the radians value of rotating counter-clockwise used by rotate_point.
    #
    #     for loc_file_name in loc_file_names: 
    #         self.input_loc_file   = open ('../res/loc_files/' + loc_file_name,  "r")
    #         loc_output_file = open ('../res/loc_files/' + loc_file_name.split('.poa')[0] + '_rttd{}.loc' .format (angle),  "w")
    #
    #         printf (loc_output_file, '// Generated by loc2poa_c.py\n')
    #
    #         for line in self.input_loc_file: 
    #
    #             # copy empty and comments lines as is to the output  
    #             line = line.split ('\n')[0] 
    #             if (line.split ("//")[0] == ""):
    #                 printf (loc_output_file, '{}\n' .format (line))
    #                 continue
    #
    #             splitted_line = line.split (" ")
    #
    #             if (splitted_line[0] == "t" or splitted_line[0] == 'usrs_that_left:'): # copy each such line as is to the output
    #                 printf (loc_output_file, '{}' .format (line))
    #
    #             elif (splitted_line[0] == 'new_or_moved:'): 
    #                 printf (loc_output_file, 'new_or_moved: ')
    #                 splitted_line = splitted_line[1:] # the rest of this line details the locations of users that are either new, or old (existing) users who moved during the last time slot
    #                 if (splitted_line !=['']): # Is there a non-empty list of vehicles that are old / new / recycled?  
    #
    #                     splitted_line = splitted_line[0].split (')') # split the line into the data given for each distinct usr
    #                     for my_tuple in splitted_line:  
    #                         if (len(my_tuple) <= 1): # no more new vehicles in this list. 
    #                             break
    #                         my_tuple = my_tuple.split("(")
    #                         my_tuple   = my_tuple[1].split (',')
    #                         [x,y] = self.rotate_point ([float(my_tuple[x_pos_idx]), float(my_tuple[y_pos_idx])])
    #                         if (not (is_in_simulated_area ('Monaco', [x,y]))):
    #                             continue
    #                         printf (loc_output_file, '({},{},{},{})' .format (my_tuple[type_idx], my_tuple[veh_id_idx], x, y))
    #                 else: # no usrs in the list - merely write the token 'new_or_moved:' to the output file
    #                     printf (loc_output_file, '{}' .format (line))
    #             printf (loc_output_file, '\n' .format (line))

    def plot_tot_num_of_vehs_over_t (self):
        
        # num_of_vehs_in_cell_0 = np.array ([2076,2093,2115,2131,2138,2157,2161,2193,2209,2217,2220,2242,2270,2284,2308,2327,2331,2332,2339,2346,2381,2411,2413,2442,2432,2471,2509,2499,2512,2538,2536,2554,2579,2588,2584,2608,2623,2649,2632,2668,2660,2679,2708,2727,2743,2745,2762,2771,2805,2813,2807,2837,2837,2828,2829,2864,2866,2877,2887,2929], dtype='int16')
        # num_of_vehs_in_cell_1 = np.array ([2335,2342,2355,2385,2410,2421,2435,2435,2450,2487,2517,2518,2524,2514,2539,2567,2591,2622,2632,2639,2636,2650,2692,2716,2737,2759,2764,2787,2820,2829,2845,2859,2881,2886,2914,2924,2943,2969,2997,2998,3031,3050,3070,3099,3123,3155,3148,3177,3185,3205,3220,3251,3256,3306,3305,3295,3331,3368,3390,3392], dtype='int16')
        # num_of_vehs_in_cell_2 = np.array ([1424,1438,1446,1453,1482,1480,1513,1533,1543,1535,1541,1551,1558,1591,1593,1603,1620,1622,1635,1643,1660,1661,1659,1666,1687,1696,1703,1716,1724,1740,1744,1765,1765,1784,1796,1809,1820,1816,1809,1824,1834,1849,1871,1876,1875,1880,1907,1906,1910,1907,1926,1931,1945,1964,1981,2000,2009,1987,1992,2028], dtype='int16')
        # tot_num_of_vehs_in_slot = num_of_vehs_in_cell_0 + num_of_vehs_in_cell_1 + num_of_vehs_in_cell_2
        tot_num_of_vehs_in_slot = [2678,2706,2734,2781,2758,2764,2786,2783,2799,2725,2739,2704,2706,2722,2728,2725,2734,2765,2767,2756,2794,2809,2815,2830,2901,2940,2965,2973,2958,2934,2878,2952,2884,2866,2884,2923,2951,2988,2952,2967,2957,2956,2959,2889,2894,2922,2951,2983,2925,2898,2901,2870,2932,2972,2920,2950,2906,2912,2946,2931]

        plt.plot ([t for t in range (len(tot_num_of_vehs_in_slot))], tot_num_of_vehs_in_slot)
        plt.show ()

def plot_demography_heatmap (city, max_power_of_4):
    """
    Plot a demography heatmap, namely, a heatmap showing the avg. # of cars left each rectangle during the simulation. 
    Inputs: 
    - city 
    - max_power_of_4 - details the number of iterative partition of the simulated area into rectangles.
    Outputs:
    A heatmap, saved in a .pdf file.
    """

    city = 'Monaco'
    max_power_of_4             = 0
    if (max_power_of_4==0 and city!='Monaco'):
        print ('Error: max_power_of_4==0 is allowed only for Monaco')
        exit ()
    if (max_power_of_4==4 and city=='Monaco'):
        print ('Error: max_power_of_4==4 is not used for Monaco as this generates too small rectangles')
        exit ()
    my_loc2poa                 = loc2poa_c (max_power_of_4 = max_power_of_4, city=city) #Monaco.Telecom.antloc', city='Monaco') #'Lux.post.antloc')
    input_demography_file_name ='{}_demography_0730_0830_1secs_{}.txt' .format (city, 3 * 4**max_power_of_4 if city=='Monaco' else 4**max_power_of_4)  
    my_loc2poa.plot_demography_heatmap (usrs_loc_file_name=input_demography_file_name, input_demography_file_name=input_demography_file_name, plot_colorbar=True)

if __name__ == '__main__':

    city = 'Monaco'
    my_loc2poa = loc2poa_c (max_power_of_4 = 0, city=city, verbose=[]) 
    my_loc2poa.calc_num_vehs_to_slot (city)

    # plot_demography_heatmap (city='Lux', max_power_of_4=1)

    # my_loc2poa     = loc2poa_c (max_power_of_4 = max_power_of_4, city=city) #Monaco.Telecom.antloc', city='Monaco') #'Lux.post.antloc')
    # pcl_num_vehs_input_file_name='num_of_vehs_{}_0730_0830_1secs.loc__{}rects.pcl' .format (city, 4**max_power_of_4 * (3 if city=='Monaco' else 1))
    # my_loc2poa.plot_num_of_vehs_in_cell_heatmap (pcl_num_vehs_input_file_name='num_of_vehs_{}_0730_0830_1secs.loc__{}rects.pcl' .format (city, 4**max_power_of_4 * (3 if city=='Monaco' else 1)), 
    #                                             pcl_lanes_len_input_file_name='{}_lanes_len.pcl' .format (city))
    
    # my_loc2poa     = loc2poa_c (max_power_of_4 = max_power_of_4, verbose = [], antloc_file_name = '', city='Lux') #Monaco.Telecom.antloc', city='Monaco') #'Lux.post.antloc')
    # pcl_input_file_name = 'num_of_vehs_Lux_0730_0830_1secs.loc__4rects.pcl'
    # my_loc2poa.gen_heatmap (df=None, pcl_input_file_name=pcl_input_file_name)
    # plt.savefig('../res/' + pcl_input_file_name.split('.pcl')[0] + '.pdf', bbox_inches='tight')
    
    # my_loc2poa.plot_tot_num_of_vehs_over_t ()
    
    # Processing
    # my_loc2poa.parse_loc_files (['Lux_0730_0830_16secs.loc']) #(['Monaco_0730_0800_1secs_rttd54.loc 'Lux_0829_0830_8secs.loc']) #(['Lux_0730_0740_1secs.loc', 'Lux_0740_0750_1secs.loc', 'Lux_0750_0800_1secs.loc', 'Lux_0800_0810_1secs.loc', 'Lux_0810_0820_1secs.loc', 'Lux_0820_0830_1secs.loc'])

    # my_loc2poa     = loc2poa_c (max_power_of_4 = max_power_of_4, verbose = [VERBOSE_CNT, VERBOSE_SPEED], antloc_file_name = '', city='Lux') #Monaco.Telecom.antloc', city='Monaco') #'Lux.post.antloc')
    
    # # Post-processing
    # my_loc2poa     = loc2poa_c (max_power_of_4 = max_power_of_4, verbose = [], antloc_file_name = 'Monaco.Telecom.antloc', city='Monaco') #Monaco.Telecom.antloc', city='Monaco') #'Lux.post.antloc')
    # my_loc2poa.plot_voronoi_diagram()


    # df= [[2481.923888888889, 2954.5575, 1684.8925]]
    # with open ('../res/num_of_vehs_Monaco_0730_0830_1secs.loc__3rects.pcl' , 'wb') as pcl_output_file:
    #     pickle.dump (df, pcl_output_file)

