import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import itertools 
import time 

from usr_c import usr_c # class of the users
from printf import printf

VERBOSE_POST_PROCESSING = 0 # Don't read ".loc" file. Read ".ap" or ".txt" files, and analyze them - e.g., count the number of cars in each cell. 
VERBOSE_AP              = 1 # Generate ".ap" file, detailing the current cell of each vehicle during the sim.
VERBOSE_CNT             = 2 # Generate ".txt" file, detailing the number of vehicles at each cell during the sim.
VERBOSE_DEMOGRAPHY      = 3 # Collect data about the # of vehicles entering / leaving each cell, at each time slot`
VERBOSE_SPEED           = 4 # Collect data about the speed of vehicles in each cell, at each time slot`

type_idx   = 0
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
    Accept as input a .loc file (a file detailing the locations of all the new users / users who moved at each slot).
    Optional outputs: 
    - An .ap file (a file detailing the Access Points of all the new users / users who moved at each slot).
    - A cnt of the number of vehicles in each cell
    """   
    # inline function for the files location-to-AP mapping
    # currently this function always maps assuming square cells.
    loc2ap = lambda self, x, y : self.loc2ap_using_rect_cells (x, y)
    
    # inline function for formatted-printing the AP of a single user
    print_usr_ap = lambda self, usr: printf(self.ap_file, "({},{})" .format (usr['id'], usr['nxt ap']))   

    def __init__(self, use_sq_cells = True, max_power_of_4=3, verbose = VERBOSE_AP):

        self.verbose           = verbose      # verbose level - defining which outputs will be written
        self.debug             = False 
        
        self.max_x, self.max_y = 13622, 11457 # size of the total area, in meters
        self.usrs              = []
        self.use_sq_cells      = use_sq_cells
        if (self.use_sq_cells):
            self.max_power_of_4    = max_power_of_4
            self.num_of_APs        = 4**max_power_of_4
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_in_ap = [[] for ap in range(self.num_of_APs)]
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.demography_file = open ('../res/vehicles_demography.txt', 'w+')
            self.joined          = [[] for ap in range(self.num_of_APs)]
            self.left            = [[] for ap in range(self.num_of_APs)]
            self.joined_sim_via  = [[] for ap in range(self.num_of_APs)]
            self.left_sim_via    = [[] for ap in range(self.num_of_APs)]
        if (VERBOSE_SPEED in self.verbose):
            self.speed_file = open ('../res/vehicles_speed.txt', 'w+')
            self.speed           = [{'speed' : 0, 'num of smpls' : 0} for ap in range(self.num_of_APs)]
        if (VERBOSE_CNT in self.verbose):
            self.num_of_vehs_file_name = '../res/num_of_vehs_per_ap_{}aps.txt' .format (4**self.max_power_of_4)
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_file_name, 'w+')
        self.tile2ap (lvl=0) # tile2ap translates the number as a "tile" (XY grid) to the ID of the covering AP.
        
    def loc2ap_using_rect_cells (self, x, y):
        """
        Finding the AP covering the user's area, assuming that the number of APs is a power of 4, and rectangular cells.
        Input:  (x,y) location data
        Output: ap that covers this area
        """
        ap = np.int8(0)
        x_offset, y_offset = x, y
        x_edge, y_edge = 0.5*self.max_x, 0.5*self.max_y
        for p in range (self.max_power_of_4):
            ap += 4**(self.max_power_of_4-1-p)*int(2 * (y_offset // y_edge) + x_offset // x_edge)
            x_offset, y_offset = x_offset % x_edge, y_offset % y_edge   
            x_edge /= 2
            y_edge /= 2
        return ap
    
    def print_usrs_ap (self):
        """
        Format-prints the users' AP, as caclculated earlier, to the .ap output file
        """
        usrs = list (filter (lambda usr: usr['new'], self.usrs))
        if (len (usrs) > 0):
            printf (self.ap_file, 'usrs: ')
            for usr in usrs: # for every new usr
                self.print_usr_ap (usr)

        usrs = list (filter (lambda usr: (usr['new'] == False) and (usr['nxt ap'] != usr['cur ap']), self.usrs))
        printf (self.ap_file, '\nold_usrs: ')
        for usr in usrs: 
            self.print_usr_ap (usr)
            usr['cur ap'] = usr['nxt ap']
                
    def cnt_num_of_vehs_per_ap (self):
        """
        Count the number of vehicles associated with each AP at the current parsed simulation step.
        """
        for ap in range(self.num_of_APs): 
            self.num_of_vehs_in_ap[ap].append (len (list (filter (lambda usr: usr['nxt ap'] == ap, self.usrs) )))

    
    def print_demography (self):
        """
        Prints the number of vehicles that joined/left each cell during the last simulated time slot.
        """
        for ap in range(self.num_of_APs):
            printf (self.demography_file, 'ap_{}: joined {}\nap_{}: joined_sim_via{}\nap_{}: left {}\nap_{}: left_sim_via {}\n' .format (
                                                ap, self.joined[ap], 
                                                ap, self.joined_sim_via[ap], 
                                                ap, self.left[ap], 
                                                ap, self.left_sim_via[ap]))
        printf (self.demography_file, '\n')                                        

    def print_speed (self):
        """
        Prints the speed of vehicles that joined/left each cell during the last simulated time slot.
        """
        printf (self.speed_file, '{}' .format ([self.speed[ap]['speed'] for ap in range(self.num_of_APs)]))                                        

    def plot_demography_heatmap (self):
        """
        Plot heatmaps, showing the avg number of vehicles that joined/left each cell during the simulated period.
        """
        self.joined         = [self.joined[ap][1:]         for ap in range (self.num_of_APs)]
        self.joined_sim_via = [self.joined_sim_via[ap][1:] for ap in range (self.num_of_APs)]

        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec_to_heatmap (np.array ([np.average(self.joined[ap]) for ap in range(self.num_of_APs)])), 
                                                columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('avg num of cars that joined cell every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_cars_joined.jpg')
        
        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec_to_heatmap (np.array ([np.average(self.left[ap]) for ap in range(self.num_of_APs)])), 
                                                columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('avg num of cars that left cell every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_cars_left.jpg')
        
        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec_to_heatmap (np.array ([np.average(self.joined_sim_via[ap]) for ap in range(self.num_of_APs)])), 
                                                columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('avg num of cars that joined the sim every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_cars_joined_sim_via.jpg')
        
        plt.figure ()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec_to_heatmap (np.array ([np.average(self.left_sim_via[ap]) for ap in range(self.num_of_APs)])), 
                                                columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('avg num of cars that left the sim every sec in {}' .format (self.time_period_str))
        plt.savefig('../res/heatmap_cars_left_sim_via.jpg')

    def rd_num_of_vehs_per_ap (self, input_file_name):
        """
        Read the number of vehicels at each cell, as written in the input files. 
        """
        input_file  = open ('../res/' + input_file_name, "r")  
        
        self.num_of_vehs_in_ap = []
        for line in input_file:
            
            if (line == "\n" or line.split ("//")[0] == ""):
                continue
        
            num_of_vehs_in_cur_ap = []
            line = line.split ("\n")[0]
            splitted_line = line.split (":")
            # ap_num = splitted_line[0].split("_")[-1]
            splitted_line = splitted_line[1].split('[')[1].split(']')[0].split(', ')
            for cur_num_of_vehs_in_this_ap in splitted_line:
                num_of_vehs_in_cur_ap.append (int(cur_num_of_vehs_in_this_ap))
            
            self.num_of_vehs_in_ap.append (num_of_vehs_in_cur_ap)            
        
    def invert_mat_bottom_up (self, mat):
        """
        Unfortunately, we write matrix starting from the smallest value at the top, while plotting maps letting the "y" (north) direction "begin" at bottom, and increase towards the top.
        Hence, need to swap the matrix upside-down
        """ 
        inverted_mat = np.empty (mat.shape)
        for i in range (mat.shape[0]):
            inverted_mat[i][:] = mat[mat.shape[0]-1-i][:]
        return inverted_mat        

    def vec_to_heatmap (self, vec):
        """
        Order the values in the given vec so that they appear as in the geographical map of cells.
        """
        n = int (math.sqrt(len(vec)))
        heatmap_val = np.array ([vec[self.tile_to_ap[i]] for i in range (len(self.tile_to_ap))]).reshape ( [n, n])
        # Unfortunately, we write matrix starting from the smallest value at the top, while plotting maps letting the "y" (north) direction "begin" at bottom, and increase towards the top.
        # Hence, need to swap the matrix upside-down
        return self.invert_mat_bottom_up(heatmap_val)

    def plot_num_of_vehs_per_ap_heatmap (self):
        """
        Plot a heatmap, showing at each cell the average number of vehicles found at that cell, along the simulation.
        """
        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec_to_heatmap (np.array ([np.average(self.num_of_vehs_in_ap[ap]) for ap in range(self.num_of_APs)])), 
                                                columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('avg num of cars per cell')
        plt.savefig('../res/heatmap_num_vehs.jpg')
        
    def plot_speed_heatmap (self):
        """
        Plot a heatmap, showing the average speed of vehicles at each cell.
        """
        plt.figure()
        my_heatmap = sns.heatmap (pd.DataFrame (self.vec_to_heatmap ([self.speed[ap]['speed'] for ap in range(self.num_of_APs)]), 
                                                columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('speed in each cell')
        plt.savefig('../res/heatmap_speed.jpg')
        
    def tile2ap (self, lvl):
        """
        prepare a translation of the "Tile" (line-by-line regular index given to cells) to the number of AP.
        """
        if (lvl == 0):
            power = lvl
            n = int(math.sqrt (self.num_of_APs/4**power))
            self.tile_to_ap       = np.empty (n**2, dtype = 'uint8')
            offset_x          = self.max_x // (2*n)        
            offset_y          = self.max_y // (2*n)        
            ap                = 0
            for y in range (offset_x, self.max_y, self.max_y // n): 
                for x in range (offset_y, self.max_x, self.max_x // n): 
                    self.tile_to_ap[ap] = self.loc2ap(x, y)
                    ap+=1 
        elif (lvl == 1):
            self.tile_to_ap = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15] 
        elif (lvl == 2):
            self.tile_to_ap = [0,1,2,3]
        elif (lvl == 3):
            self.tile_to_ap = [0]
        else:
            print ('Sorry, the level of 4 you chose is still unsupported by tile2ap')
            exit ()

    def plot_num_of_vehs_per_ap (self):    
        """
        Plot for each ap the number of vehicles associated with it along the trace.
        """
        for plot_num in range (4**(self.max_power_of_4-1)):
            for ap in range (4*plot_num, 4*(plot_num+1)):
                if (VERBOSE_CNT in self.verbose):
                    printf (self.num_of_vehs_output_file, 'num_of_vehs_in_ap_{}: {}\n' .format (ap, self.num_of_vehs_in_ap[ap])) 
                plt.title ('Number of vehicles in each cell')
                plt.plot (range(len(self.num_of_vehs_in_ap[ap])), self.num_of_vehs_in_ap[ap], label='cell {}' .format(ap))
                plt.ylabel ('Number of vehicles')
                plt.xlabel ('time [minutes, starting at 07:30]')
            plt.legend()
            plt.savefig ('../res/num_of_vehs_per_cell_plot{}.jpg' .format(plot_num))
            plt.clf()
            
    # def find_max_X_max_Y (self):
    #     max_x, max_y = 0, 0
    #     for i in range (9):
    #         usrs_loc_file_name = 'vehicles_{}.loc' .format (i)
    #         print ('checking file {}' .format (usrs_loc_file_name)) 
    #         usrs_loc_file           = open ('../res/' + usrs_loc_file_name,  "r") 
    #         for line in usrs_loc_file: 
    #             line = line.split ('\n')[0] 
    #             if (line.split ("//")[0] == ""):
    #                 continue
    #
    #             splitted_line = line.split (" ")
    #
    #             Code isn't complete here. Need to be revised
    #     print ('max_x = {}, max_y = {}' .format (max_x, max_y))

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
                    for ap in range (self.num_of_APs): 
                        self.joined        [ap].append(0)
                        self.left          [ap].append(0)
                        self.joined_sim_via[ap].append(0)
                        self.left_sim_via  [ap].append(0)
                continue
    
            elif (splitted_line[0] == 'usrs_that_left:'):
                if (VERBOSE_AP in self.verbose):
                    printf(self.ap_file, '{}\n' .format (line))
                ids_of_usrs_that_left = [int(id) for id in splitted_line[1:] if id!= '']                
                if (VERBOSE_DEMOGRAPHY in self.verbose):
                    for usr in list (filter (lambda usr: usr['id'] in ids_of_usrs_that_left, self.usrs)): 
                        self.left_sim_via[usr['cur ap']][-1] += 1 # inc the # of vehicles left the sim' via this cell
                        # print (self.left_sim_via) #$$$
                self.usrs = list (filter (lambda usr : (usr['id'] not in ids_of_usrs_that_left), self.usrs))
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

                        usr_id = np.uint16(tuple[veh_id_idx])
                        nxt_ap = self.loc2ap (float(tuple[x_pos_idx]), float(tuple[y_pos_idx]))
                        if (tuple[type_idx] == 'n'): # new vehicle
                            self.usrs.append ({'id' : usr_id, 'cur ap' : nxt_ap, 'nxt ap' : nxt_ap, 'new' : True}) # for a new usr, we mark the cur_ap same as nxt_ap 
                            if (VERBOSE_DEMOGRAPHY in self.verbose): 
                                self.joined        [nxt_ap][-1] += 1 # inc the # of usrs that joined this cell at this cycle
                                self.joined_sim_via[nxt_ap][-1] += 1 # inc the # of usrs that joined the sim' via this cell
                        elif (tuple[type_idx] == 'o'): # recycled vehicle's id, or an existing user, who moved
                            list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, self.usrs))
                            if (len(list_of_usr) == 0):
                                print  ('Error at t={}: input file={}. Did not find old / recycled usr {}' .format (self.t, self.usrs_loc_file_name, splitted_line[1]))
                                exit ()
                            list_of_usr[0]['nxt ap'] = nxt_ap
                            if (VERBOSE_DEMOGRAPHY in self.verbose and nxt_ap!= list_of_usr[0]['cur ap']): #this user moved to another cell  
                                self.joined[nxt_ap][-1]                   += 1 # inc the # of usrs that joined this cell
                                self.left  [list_of_usr[0]['cur ap']][-1] += 1 # inc the # of usrs that left the previous cell of that usr
                        else:
                            print ('Wrong type of usr.')
                            exit () 
                        if (VERBOSE_SPEED in self.verbose):
                            self.speed[nxt_ap] = {'speed' : (float(tuple[speed_idx]) + self.speed[nxt_ap]['num of smpls'] * self.speed[nxt_ap]['speed'])/(self.speed[nxt_ap]['num of smpls'] + 1), 
                                                  'num of smpls' : self.speed[nxt_ap]['num of smpls'] + 1}
                            
                            # print ('self.speed[{}]={}' .format(nxt_ap, self.speed[nxt_ap])) #$$$$
                                    
                # At this point we finished handling all the usrs (left / new / moved) reported by the input ".loc" file at this slot. So now, output the data to ".ap" file, and/or to a file, counting the vehicles at each cell
                if (VERBOSE_AP in self.verbose):
                    self.print_usrs_ap() # First, print the APs of the users in the PREVIOUS cycles
                if (VERBOSE_CNT in self.verbose):
                    self.cnt_num_of_vehs_per_ap ()
                for usr in self.usrs: # mark all existing usrs as old
                    usr['new'] = False
                    usr['cur ap'] = usr ['nxt ap']
    
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
            self.plot_num_of_vehs_per_ap ()
        if (VERBOSE_DEMOGRAPHY in self.verbose):
            self.plot_demography_heatmap()
        if (VERBOSE_SPEED in self.verbose):
            
            # first, fix the speed, as we assumed a first car with speed '0'.
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
                printf (self.num_of_vehs_output_file, 'num_of_vehs_in_ap_{}: {}\n' .format (ap, self.num_of_vehs_in_ap[ap]))
        if (VERBOSE_DEMOGRAPHY in self.verbose): 
            self.demography_file   = open ('../res/vehicles_demography.txt', 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.demography_file, '// after parsing {}\n' .format (self.usrs_loc_file_name))                
            self.print_demography()
        if (VERBOSE_SPEED in self.verbose): 
            self.speed_file   = open ('../res/vehicles_speed.txt', 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.speed_file, '// after parsing {}\n' .format (self.usrs_loc_file_name))                
            self.print_speed()
    
    def parse_files (self, loc_file_names):
        """
        Parse one or more ".loc" files, named "files_prefix_i.loc", where i = 0, 1, ... num_of_files-1
        E.g. if files_prefix = vehicles and num_of_files = 2,
        this function will parse the files vehicles_0.loc, vehicles_1.loc
        for each of the parsed files, the function will:
        1. output the number of vehicles at each ap. AND/OR
        2. output the APs of all new/left/moved users at each time slot.
        The exact behavior is by the value of self.verbose
        """
        if (VERBOSE_AP in self.verbose):
            self.ap_file        = open ("../res/" + loc_file_names[0].split('.')[0] + ".ap", "w+")  
            printf (self.ap_file, '// File format:\n//for each time slot:\n')
            printf (self.ap_file, '//for each time slot:\n')
            printf (self.ap_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (self.ap_file, '"new_usrs" is a list of the new usrs, and their APs, e.g.: (0, 2)(1,3) means that new usr 0 is in cell 2, and new usr 1 is in cell 3.\n')
            printf (self.ap_file, '"old_usrs" is a list of the usrs who moved to another cell in the last time slot, and their current APs, e.g.: (0, 2)(1,3) means that old usr 0 is now in cell 2, and old usr 1 is now in cell 3.\n')
        
        for file_name in loc_file_names: #i in range (num_of_files):
            self.usrs_loc_file_name = file_name
            self.usrs_loc_file      = open ('../res/' + self.usrs_loc_file_name,  "r") 
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
    
    def print_num_of_vehs_diffs (self):
        ap = 12
        print ('diff 12 = {}' .format (np.average([(self.num_of_vehs_in_ap[ap][i+1] - self.num_of_vehs_in_ap[ap][i]) for i in range (len(self.num_of_vehs_in_ap[ap])-1) if 
             (self.num_of_vehs_in_ap[ap][i+1] - self.num_of_vehs_in_ap[ap][i])>= 0])))
    
        ap = 48
        print ('diff 48 = {}' .format (np.average([(self.num_of_vehs_in_ap[ap][i+1] - self.num_of_vehs_in_ap[ap][i]) for i in range (len(self.num_of_vehs_in_ap[ap])-1) if 
             (self.num_of_vehs_in_ap[ap][i+1] - self.num_of_vehs_in_ap[ap][i])>= 0])))
        ap = 51
        print ('diff 51 = {}' .format (np.average([(self.num_of_vehs_in_ap[ap][i+1] - self.num_of_vehs_in_ap[ap][i]) for i in range (len(self.num_of_vehs_in_ap[ap])-1) if 
             (self.num_of_vehs_in_ap[ap][i+1] - self.num_of_vehs_in_ap[ap][i])>= 0])))
    
    def print_num_of_vehs_per_server (self, output_file_name):
        """
        Print the number of vehicles in the sub-tree below each server, assuming that the simulated area is iteratively partitioned to rectangular cells,
        so that the number of cells is a power of 4. 
        """
        output_file = open ('../res/' + output_file_name, 'w')
        printf (output_file, 'avg num of cars per server\n')
        avg_num_of_vehs_per_ap = np.array ([np.average(self.num_of_vehs_in_ap[ap]) for ap in range(self.num_of_APs)]) 
        for lvl in range (self.max_power_of_4):
            self.tile2ap (lvl) # call a function that translates the number as "tile" to the ID of the covering AP.
            printf (output_file, '\nlevel {}\n******************\n' .format (lvl))
            self.print_as_sq_mat (output_file, self.vec_to_heatmap (avg_num_of_vehs_per_ap))
            reshaped_heatmap = avg_num_of_vehs_per_ap.reshape (int(len(avg_num_of_vehs_per_ap)/4), 4) # prepare the averaging for the next iteration
            avg_num_of_vehs_per_ap = np.array([np.sum(reshaped_heatmap[i][:])for i in range(reshaped_heatmap.shape[0])], dtype='int') #perform the averaging, to be used by the ext iteration.
        
if __name__ == '__main__': 
    # gamad = [[1,2,3], [4,5,6]]
    # print (gamad)    
    # gamad = [gamad[ap][1:] for ap in range (2)]
    # # for ap in range(2):
    # #     gamad[ap] = gamad [ap][1:]
    # #     # del (gamad[ap][0])
    # print (gamad)    
    # exit ()

    max_power_of_4 = 3
    my_loc2ap      = loc2ap_c (max_power_of_4 = max_power_of_4, use_sq_cells = True, verbose = [VERBOSE_AP, VERBOSE_CNT, VERBOSE_DEMOGRAPHY, VERBOSE_SPEED])
    my_loc2ap.time_period_str = '' #'0730_0740'
    my_loc2ap.parse_files (['vehicles_n_speed_0730.loc']) #(['vehicles_s_speed_0730.loc']) #, 'vehicles_0740.loc', 'vehicles_0750.loc', 'vehicles_0800.loc', 'vehicles_0810.loc', 'vehicles_0820.loc'])

    # my_loc2ap       = loc2ap_c (max_power_of_4 = max_power_of_4, use_sq_cells = True, verbose = VERBOSE_POST_PROCESSING)
    # input_file_name = 'num_of_vehs_per_ap_{}aps.txt' .format (4**max_power_of_4)
    # my_loc2ap.rd_num_of_vehs_per_ap  (input_file_name)
    # # my_loc2ap.print_num_of_vehs_diffs ()
    # output_file_name = 'num_of_vehs_per_server{}.txt' .format (4**max_power_of_4)
    # # my_loc2ap.plot_num_of_vehs_per_ap ()
    # my_loc2ap.print_num_of_vehs_per_server (output_file_name)
    # my_loc2ap.plot_num_of_vehs_per_ap_heatmap ()

    # For finding the maximum positional values of x and y in the .loc file(s), uncomment the line below 
    # my_loc2ap.find_max_X_max_Y ()    
