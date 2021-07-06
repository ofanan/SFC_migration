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
VERBOSE_AP_AND_CNT      = 3 # Generate both ".ap" and ".txt" files, as detailed above.
VERBOSE_DEMOGRAPHY      = 4 # Collect data about the # of vehicles entering / leaving each cell, at each time slot

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

        self.max_x, self.max_y = 13622, 11457 # size of the total area, in meters
        self.verbose           = verbose      # verbose level - defining which outputs will be written
        self.usrs              = []
        self.use_sq_cells      = use_sq_cells
        if (self.use_sq_cells):
            self.max_power_of_4    = max_power_of_4
            self.num_of_APs        = 4**max_power_of_4
        if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
            self.num_of_vehs_in_ap = [[] for ap in range(self.num_of_APs)]
        if (self.verbose in [VERBOSE_DEMOGRAPHY]):
            self.usrs_demography_file = open ('../res/vehicles.demography.txt', 'w+')
            self.joined = [[] for ap in range(self.num_of_APs)]
            self.left   = [[] for ap in range(self.num_of_APs)]
        if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
            self.num_of_vehs_file_name = '../res/num_of_vehs_per_ap_{}aps.txt' .format (4**self.max_power_of_4)
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_file_name, 'w+')
        
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
        new_usrs = list (filter (lambda usr: usr['new'], self.usrs))
        if (len (new_usrs) > 0):
            printf (self.ap_file, 'new_usrs: ')
            for usr in new_usrs: # for every new usr
                self.print_usr_ap (usr)

        moved_old_usrs = list (filter (lambda usr: (usr['new']==False) and (usr['nxt ap'] != usr['cur ap']), self.usrs))
        printf (self.ap_file, '\nold_usrs: ')
        for usr in moved_old_usrs: 
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
        Used for debug.
        Prints the number of vehicles that joined/left each cell during the last simulated time slot.
        """
        for ap in range(self.num_of_APs):
            printf (self.usrs_demography_file, 'ap {}: joined {}. left {}\n' .format (ap, self.joined[ap], self.left[ap]))
        printf (self.usrs_demography_file, '\n')                                        

    def calc_demography_per_ap (self):
        """
        calculates the number of vehicles that joined/left each cell during the last simulated time slot.
        """
        for ap in range(self.num_of_APs): 
            self.joined[ap].append (len (list (filter (lambda usr: usr['nxt ap'] == ap and usr['cur ap'] != ap, self.usrs) )))
            self.left[ap].append   (len (list (filter (lambda usr: usr['cur ap'] == ap and usr['nxt ap'] != ap, self.usrs) )))
        
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
        inverted_mat = np.empty (mat.shape, dtype = 'uint16')
        for i in range (mat.shape[0]):
            inverted_mat[i][:] = mat[mat.shape[0]-1-i][:]
        return inverted_mat        

    def heatmap_of_avg_num_of_vehs_per_ap (self):
        """
        Plot a heatmap, showing at each cell the average number of vehicles found at that cell, along the simulation.
        """

        self.tile2ap (lvl=0)
        avg_num_of_vehs_per_ap = np.array ([np.average(self.num_of_vehs_in_ap[ap]) for ap in range(self.num_of_APs)]) 
        n = int (math.sqrt(len(avg_num_of_vehs_per_ap)))
        heatmap_val = np.array ([int(avg_num_of_vehs_per_ap[self.tile_to_ap[i]]) for i in range (self.num_of_APs)]).reshape ( [n, n])
        
        # Unfortunately, we write matrix starting from the smallest value at the top, while plotting maps letting the "y" (north) direction "begin" at bottom, and increase towards the top.
        # Hence, need to swap the matrix upside-down
        heatmap_val = self.invert_mat_bottom_up(heatmap_val)
        my_heatmap = sns.heatmap (pd.DataFrame (heatmap_val, columns=["0","1","2","3","4","5","6","7"]), cmap="YlGnBu")
        plt.title ('avg num of cars per cell')
        # plt.show()
        plt.savefig('../res/heatmap.jpg')
        
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
                if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
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
                if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
                    printf(self.ap_file, '\n{}\n' .format (line)) # print the header of the current time: "t = ..."
                self.t = int(splitted_line[2])
                continue
    
            elif (splitted_line[0] == 'usrs_that_left:'):
                if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
                    printf(self.ap_file, '{}\n' .format (line))
                usrs_that_left = [int(id) for id in splitted_line[1:] if id!= '']                
                self.usrs = list (filter (lambda usr : (usr['id'] not in usrs_that_left), self.usrs))
                continue
    
            elif (splitted_line[0] == 'new_or_moved:'): 
                splitted_line = splitted_line[1:] # the rest of this line details the locations of users that are either new, or old (existing) users who moved during the last time slot
                if (splitted_line !=['']):

                    splitted_line = splitted_line[0].split (')') # split the line into the data given for each distinct usr
                    for tuple in splitted_line: 
                        if (len(tuple) <= 1):
                            break
                        tuple = tuple.split("(")
                        tuple   = tuple[1].split (',')

                        nxt_ap = self.loc2ap (float(tuple[2]), float(tuple[3]))
                        usr_id = np.uint16(tuple[1])
                        if (tuple[0] == 'n'): # new vehicle
                            self.usrs.append ({'id' : usr_id, 'cur ap' : nxt_ap, 'nxt ap' : nxt_ap, 'new' : True}) # for a new usr, we mark the cur_ap same as nxt_ap 
                        else: # existing user, who moved
                            list_of_usr = list (filter (lambda usr: usr['id'] == usr_id, self.usrs))
                            if (len(list_of_usr) == 0):
                                print  ('Error at t={}: input file={}. Did not find old usr {}' .format (self.t, self.usrs_loc_file_name, splitted_line[1]))
                                exit ()
                            list_of_usr[0]['nxt ap'] = nxt_ap

                # At this point we finished handling all the usrs (left / new / moved) reported by the input ".loc" file at this slot. So now, output the data to ".ap" file, and/or to a file, counting the vehicles at each cell
                if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
                    self.print_usrs_ap() # First, print the APs of the users in the PREVIOUS cycles
                if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
                    self.cnt_num_of_vehs_per_ap ()
                if (self.verbose in [VERBOSE_DEMOGRAPHY]):
                    self.calc_demography_per_ap ()
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
        if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
            printf(self.ap_file, "\n")   
        if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
            self.plot_num_of_vehs_per_ap ()
        if (self.verbose in [VERBOSE_DEMOGRAPHY]):
            self.usrs_demography_file = open ('../res/vehicles.demography.txt', 'w')
            print ('b4: {}' .format (self.left[5]))
            
            for ap in range (self.num_of_APs):
                #self.joined[ap].pop()
                self.left  [ap].pop()    
            print ('after: {}' .format (self.left[5]))
            self.print_demography()

     
    def print_intermediate_res (self): 
        """
        Print the current aggregate results; used for having intermediate results when running long simulation.
        """
        if (self.verbose in [VERBOSE_CNT, VERBOSE_AP_AND_CNT]):
            self.num_of_vehs_output_file = open ('../res/' + self.num_of_vehs_file_name, 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.num_of_vehs_output_file, '// after parsing the file {}\n' .format (self.usrs_loc_file_name))
            for ap in range (self.num_of_APs):
                printf (self.num_of_vehs_output_file, 'num_of_vehs_in_ap_{}: {}\n' .format (ap, self.num_of_vehs_in_ap[ap]))
        if (self.verbose == VERBOSE_DEMOGRAPHY): 
            self.usrs_demography_file   = open ('../res/vehicles.demography.txt', 'w') # overwrite previous content at the output file. The results to be printed now include the results printed earlier.
            printf (self.usrs_demography_file, '// after parsing {}\n' .format (self.usrs_loc_file_name))                
            self.print_demography()
    
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
        if (self.verbose in [VERBOSE_AP, VERBOSE_AP_AND_CNT]):
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
                printf (output_file, '{}\t' .format (mat[x][y]))
            printf (output_file, '\n')
    
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
            heatmap_val = np.array ([avg_num_of_vehs_per_ap[self.tile_to_ap[i]] for i in range (len(self.tile_to_ap))], dtype='int16').reshape (int(math.sqrt(len(self.tile_to_ap))), int(math.sqrt(len(self.tile_to_ap)))) # extract the required value at each relevant area, by averaging the values of avg_num_of_vehs_per_ap at that area 
            heatmap_val = self.invert_mat_bottom_up(heatmap_val)
            printf (output_file, '\nlevel {}\n******************\n' .format (lvl))
            self.print_as_sq_mat (output_file, heatmap_val)
            reshaped_heatmap = avg_num_of_vehs_per_ap.reshape (int(len(avg_num_of_vehs_per_ap)/4), 4) # prepare the averaging for the next iteration
            avg_num_of_vehs_per_ap = np.array([np.sum(reshaped_heatmap[i][:])for i in range(reshaped_heatmap.shape[0])], dtype='int') #perform the averaging, to be used by the ext iteration.
        
        printf (output_file, '\nlevel {}\n******************\n{}' .format (self.max_power_of_4, np.average(heatmap_val)))
            
    
if __name__ == '__main__':
    max_power_of_4 = 3
    # gamad_file = open ('../res/gamad.txt', 'w+')
    # printf (gamad_file, 'rgrgrg')
    # gamad_file = open ('../res/gamad.txt', 'w')
    # printf (gamad_file, 'abcd')
    
    my_loc2ap      = loc2ap_c (max_power_of_4 = max_power_of_4, use_sq_cells = True, verbose = VERBOSE_DEMOGRAPHY)
    my_loc2ap.parse_files (['short_0.loc'])#, 'vehicles_0910.loc', 'vehicles_0920.loc', 'vehicles_0930.loc', 'vehicles_0940.loc', 'vehicles_0950.loc'])

    # my_loc2ap       = loc2ap_c (max_power_of_4 = max_power_of_4, use_sq_cells = True, verbose = VERBOSE_POST_PROCESSING)
    # input_file_name = 'num_of_vehs_per_ap_{}aps.txt' .format (4**max_power_of_4)
    # my_loc2ap.rd_num_of_vehs_per_ap (input_file_name)
    # output_file_name = 'num_of_vehs_per_server{}.txt' .format (4**max_power_of_4)
    # my_loc2ap.plot_num_of_vehs_per_ap ()
    # my_loc2ap.print_num_of_vehs_per_server (output_file_name)
    # my_loc2ap.heatmap_of_avg_num_of_vehs_per_ap ()

    # For finding the maximum positional values of x and y in the .loc file(s), uncomment the line below 
    # my_loc2ap.find_max_X_max_Y ()    
