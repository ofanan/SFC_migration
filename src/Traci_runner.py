import sumolib
from sumolib import checkBinary  
import traci, sys, math, pickle
import numpy as np
from shapely.geometry import Polygon, box

# My own format print functions 
from printf import printf 
from secs2hour import secs2hour
import loc2poa_c

VERBOSE_LOC      = 2
VERBOSE_SPEED    = 3

# Indices of fields in input antenna loc files
radio_idx = 0 # Radio type: GSM, LTE etc.
mnc_idx   = 2 # Mobile Network Code
lon_pos_idx = 6 
lat_pos_idx = 7
                  
EPSILON=0.01

netFile = {'Lux' : r'../../LuSTScenario/scenario/lust.net.xml', 'Monaco' : r'../../MoSTScenario/scenario/in/most.net.xml'}

# Return the Euclidian dist between two points
dist = lambda p1, p2 : math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class Traci_runner (object):

    # Given the x,y position, return the x,y position within the simulated area (city center) 
    relative_to_abs_pos = lambda self, pos: np.array(pos, dtype='int16') + loc2poa_c.LOWER_LEFT_CORNER [self.city]

    # Given the x,y position, return the x,y position within the simulated area (city center) 
    abs_to_relative_pos = lambda self, pos: np.array(pos, dtype='int16') - loc2poa_c.LOWER_LEFT_CORNER [self.city]

    # Returns the relative location of a given vehicle ID. The relative location found after rotating the point (if needed), and then position it w.r.t. the lower left (south-west) corner of the simulated area.
    get_relative_position = lambda self, veh_key  : self.abs_to_relative_pos (self.rotate_point (traci.vehicle.getPosition(veh_key))) 

    # Given the lon, lat coordinates of a point, return the (x,y) coordinates of its relative position within the simulated area 
    lon_lat_to_relative_pos = lambda self, lon, lat : self.abs_to_relative_pos (self.rotate_point (traci.simulation.convertGeo (lon, lat, True)))

    # Generate a string, showing the start and end time of the sim, in a 24-h clock
    gen_time_str = lambda self, start_time, end_time : '{}_{}' .format (secs2hour(start_time), secs2hour(end_time)) 

    # My Sumo command, to start a Traci simulation
    mysumoCmd = lambda self : [checkBinary('sumo'), '-c', self.sumo_cfg_file, '-W', '-V', 'false', '--no-step-log', 'true']

    # My Sumo command, to start a dummy Traci simulation, used merely to calculate the total length of lanes within a given rectangle
    LaneLengthSumoCmd = lambda self : [checkBinary('sumo'), '-c', self.sumo_cfg_file, "--start", "--quit-on-end", '-W', '-V', 'false', '--no-step-log', 'true']

    def handle_Nan_point (self): 
        print ('Warning: encountered a Nan point at t={}' .format (self.t))
        return [-1, -1] # return a dumy point, that would be out of any simulated area
    
    def rotate_point (self, point, angle=None):
        """
        rotate a given point by self.rotate_angle radians counter-clockwise around self.pivot
        """ 
        if (math.isnan(point[0]) or math.isnan(point[0])):
            return self.handle_Nan_point()
        angle = angle if (angle!=None) else self.rotate_angle
        if (angle==0):
            return np.array (point)
        return np.array ([self.pivot[0] + math.cos(angle) * (point[0] - self.pivot[0]) - math.sin(angle) * (point[1] - self.pivot[1]),
                          self.pivot[1] + math.sin(angle) * (point[0] - self.pivot[0]) + math.cos(angle) * (point[1] - self.pivot[1])], 
                          dtype='int16')
    
    def relative_rttd_pos_to_abs_pos (self, point):
        """
        Given the relative position of a point (x,y) within the simulated area return its absolute (x,y) position.
        If the relative position was obtained by a rotation, this casting includes also the performas counter rotation. 
        """
        if (math.isnan(point[0]) or math.isnan(point[0])):
            self.handle_Nan_point()
            return None 
        if (not (loc2poa_c.is_in_simulated_area(self.city, point))):
            print ('Warning: relative_rttd_pos_to_abs_pos was called with point {} which is outside the simulated area' .format(point))
            return None
        
        # Now we know that this point is within the simulated area
        # abs_point = self.relative_to_abs_pos (point)
        # print ('abs_point={}' .format (abs_point))

        return self.rotate_point (self.relative_to_abs_pos (point), -self.rotate_angle)
    
    
    def relative_pos_to_lon_lat (self, point):
        """
        Given the relative position of a point (x,y) within the simulated area, return its [latitude, longitude] 
        """
        rttd_back_point = self.relative_rttd_pos_to_abs_pos (point)
        if (not(rttd_back_point.any() == None)): 
            return traci.simulation.convertGeo (rttd_back_point[0],rttd_back_point[1])
            
    def calc_relative_pos_corners (self):
        """
        Calculate the relative positions of the 4 corners of the simulated area
        """
        min_x, min_y = loc2poa_c.MIN_X[self.city] + EPSILON, loc2poa_c.MIN_Y[self.city] + EPSILON
        max_x, max_y = loc2poa_c.MAX_X[self.city] - EPSILON, loc2poa_c.MAX_Y[self.city] - EPSILON
        self.relative_sw_corner = [min_x, min_y]
        self.relative_nw_corner = [min_x, max_y]
        self.relative_se_corner = [max_x, min_y]
        self.relative_ne_corner = [max_x, max_y]
            
    def print_abs_pos_corners_of_simulated_area (self):
        
        """
        Print the absolute, nont-rotated, position of the 4 corners of the simulated area
        """
        self.calc_relative_pos_corners()
        point = self.relative_rttd_pos_to_abs_pos (self.relative_sw_corner)
        print ('sw_corner pos ={}, {}' .format (point[0], point[1]))

        point = self.relative_rttd_pos_to_abs_pos (self.relative_nw_corner)
        print ('nw_corner pos ={}, {}' .format (point[0], point[1]))

        point = self.relative_rttd_pos_to_abs_pos (self.relative_se_corner)
        print ('se_corner pos ={}, {}' .format (point[0], point[1]))

        point = self.relative_rttd_pos_to_abs_pos (self.relative_ne_corner)
        print ('ne_corner pos ={}, {}' .format (point[0], point[1]))
     
    def print_lon_lat_corners_of_simulated_area (self):
        
        """
        Print the latitude and longitude of the 4 corners of the simulated area
        """
        self.calc_relative_pos_corners()
        traci.start (self.mysumoCmd())
        lon_lat_sw_corner = self.relative_pos_to_lon_lat (self.relative_sw_corner)
        print ('sw_corner(lat,lon)={}, {}' .format (lon_lat_sw_corner[1], lon_lat_sw_corner[0]))

        lon_lat_nw_corner = self.relative_pos_to_lon_lat (self.relative_nw_corner)
        print ('nw_corner(lat,lon)={}, {}' .format (lon_lat_nw_corner[1],lon_lat_nw_corner[0]))

        lon_lat_se_corner = self.relative_pos_to_lon_lat (self.relative_se_corner)
        print ('se_corner(lat,lon)={}, {}' .format (lon_lat_se_corner[1],lon_lat_se_corner[0]))

        lon_lat_ne_corner = self.relative_pos_to_lon_lat (self.relative_ne_corner)
        print ('ne_corner(lat,lon)={}, {}' .format (lon_lat_ne_corner[1],lon_lat_ne_corner[0]))
        # print ('Check: casting back={}' .format(self.lon_lat_to_relative_pos(ne_corner[0], ne_corner[1])))
        traci.close()
     
    def __init__ (self, sumo_cfg_file='LuST.sumocfg'):
        self.sumo_cfg_file = sumo_cfg_file

        # Find out which city we're actually simulating
        if (sumo_cfg_file=='myLuST.sumocfg'):
            self.city = 'Lux'
            self.providers_mnc = {'post' : '1', 'tango' : '77', 'orange' : '99'}         # Mobile Network Codes of various operators in Luxembourg
        elif (sumo_cfg_file=='myMoST.sumocfg'):
            self.city = 'Monaco'
            self.providers_mnc = {'Telecom' : '10'}
        else:
            print ('Error: Tracei_runner init encountered an unknown cfg file')
            exit ()                        
        
        self.rotate_angle = -math.radians(54) if self.city=='Monaco' else 0 # angle to rotate the points. The requested angle degrees of clcokwise is converted to the radians value of rotating counter-clockwise used by rotate_point.
        self.pivot = [loc2poa_c.GLOBAL_MAX_X[self.city]/2, loc2poa_c.GLOBAL_MAX_Y[self.city]/2] # pivot point, around which the rotating is done

    def simulate_to_cnt_vehs_only (self, warmup_period=0, sim_length=10, len_of_time_slot_in_sec=1, verbose = []):
        """
        Simulate a fast Traci SUMO simulation, to cnt the number of active vehicles/pedestrians at each slot; and the total number of distinct cars along the sim.
        """       
        self.verbose            = verbose
        
        time_str = self.gen_time_str (warmup_period, warmup_period+sim_length)
        print ('Running Traci for the period {}' .format (time_str))
        traci.start(self.mysumoCmd())
        self.cnt_output_file_name = '../res/{}_{}_{}secs_cnt.res' .format (self.city, time_str, len_of_time_slot_in_sec) 
        self.cnt_output_file      = open (self.cnt_output_file_name, 'w')
               
        if (warmup_period > 0):
            traci.simulationStep (int(warmup_period)) # simulate without output until our required time (time starts at 00:00). 
            
        known_veh_keys      = () # will hold the set of known vehicles' keys
        tot_num_of_vehs_in_slot = []  # tot_num_of_vehs_in_slot [t] will hold the number of distinct vehs in the simulated area at each slot t          
            
        while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
            
            self.t = traci.simulation.getTime()
            
            # Finished the sim. Now, just make some post-processing. 
            if (self.t >= warmup_period + sim_length):
                print ('Number of distinct cars during the simulated period={}' .format (len(known_veh_keys)))
                printf (self.cnt_output_file, 'Number of distinct cars during the simulated period={}\n' .format (len(known_veh_keys))) 
                break
            
            cur_list_of_vehs = list ([veh_key for veh_key in traci.vehicle.getIDList() if loc2poa_c.is_in_simulated_area (self.city, self.get_relative_position(veh_key))]) 
                        
            # sys.stdout.flush()

            known_veh_keys       =  set (cur_list_of_vehs) | set (known_veh_keys) # Union known_veh_keys with the set of vehicles' keys of this cycle
            tot_num_of_vehs_in_slot.append (len(cur_list_of_vehs))
                   
            traci.simulationStep (self.t + len_of_time_slot_in_sec)
        traci.close()
        
        printf (self.cnt_output_file, 'avg_num_of_vehs_in_simulated_area={:.2f}\nmax_num_of_vehs_in_simulated_area={}\n' .format (np.average(tot_num_of_vehs_in_slot), max (tot_num_of_vehs_in_slot))) 
        printf (self.cnt_output_file, 'num_of_vehs_in_simulated_area={}\n' .format (tot_num_of_vehs_in_slot)) 
        
        with open ('../res/{}_{}_{}secs_cnt.pcl' .format (self.city, time_str, len_of_time_slot_in_sec), 'wb') as pcl_output_file:
            pickle.dump (tot_num_of_vehs_in_slot, pcl_output_file)

    def simulate (self, warmup_period=0, sim_length=10, len_of_time_slot_in_sec=1, num_of_output_files=1, verbose = []):
        """
        Simulate Traci, and print-out the locations and/or speed of cars, as defined by the verbose input
        """

        veh_key2id               = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
        # prsn_key2id              = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
        veh_ids2recycle          = [] # will hold a list of ids that are not used anymore, and therefore can be recycled (used by new users == garbage collection).
        vehs_left_in_this_cycle  = []
        # prsns_left_in_this_cycle = []
        self.verbose             = verbose
        
        time_str = self.gen_time_str (warmup_period, warmup_period+sim_length)
        print ('Running Traci for the period {}. Will write res to {} output files' .format (time_str, num_of_output_files))
        traci.start (self.mysumoCmd())
        
        if (warmup_period > 0):
            traci.simulationStep (int(warmup_period)) # simulate without output until our required time (time starts at 00:00). 
        for i in range (num_of_output_files):
            
            loc_output_file_name = '../res/loc_files/{}_{}_{}secs{}.loc' .format (self.city, time_str, len_of_time_slot_in_sec, ('_spd' if (VERBOSE_SPEED in self.verbose) else ''))   
            with open(loc_output_file_name, 'w') as loc_output_file:
                loc_output_file.write('')                
            loc_output_file  = open (loc_output_file_name,  "w")
            printf (loc_output_file, '// locations of vehicles. Format:\n')
            printf (loc_output_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (loc_output_file, '// format for vehicles that are new (just joined the sim), or moved:\n')
            if (VERBOSE_SPEED in self.verbose): 
                printf (loc_output_file, '// (veh_type,usr_id,x,y,s)   where:\n')
                printf (loc_output_file, '// veh_type is either o (old veh), or n (new veh in the sim). (x,y) are the coordinates of the vehicle with this usr_id. s is the speed of this user.\n')
            elif (VERBOSE_LOC in self.verbose):
                printf (loc_output_file, '// (veh_type,usr_id,x,y)   where:\n')
                printf (loc_output_file, '// veh_type is either o (old veh), or n (new veh in the sim). (x,y) are the coordinates of the vehicle with this usr_id.\n')
                printf (loc_output_file, '// Generated by Traci_runner.py .\n')
                
            while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
                
                self.t = traci.simulation.getTime()
                
                # Finished the sim. Now, just make some post-processing. 
                if (self.t >= warmup_period + sim_length*(i+1) / num_of_output_files):
                    print ('Successfully finished writing to file {}' .format (loc_output_file_name))
                    break
                  
                cur_list_of_vehs = [{'key' : veh_key, 'pos' : self.get_relative_position(veh_key)} for veh_key in traci.vehicle.getIDList() if loc2poa_c.is_in_simulated_area (self.city, self.get_relative_position(veh_key))]            
                printf (loc_output_file, '\nt = {:.0f}\n' .format (self.t))

                vehs_left_in_this_cycle = list (filter (lambda veh : (veh['key'] not in [item['key'] for item in cur_list_of_vehs] and 
                                                                   veh['id']  not in (veh_ids2recycle)), veh_key2id)) # The list of vehs left at this cycle includes all vehs that are not in the list of currently-active vehicles, and haven't already been listed as "vehs that left" (i.e., veh ids to recycle). 
                veh_key2id = list (filter (lambda veh : veh['id'] not in [veh['id'] for veh in vehs_left_in_this_cycle], veh_key2id)) # remove usrs that left from the veh_key2id map 
                printf (loc_output_file, 'usrs_that_left: ')
                if (len (vehs_left_in_this_cycle) > 0):
                    for veh in vehs_left_in_this_cycle:
                        printf (loc_output_file, '{:.0f} ' .format(veh['id']))
                    veh_ids2recycle = sorted (list (set (veh_ids2recycle) | set([item['id'] for item in vehs_left_in_this_cycle]))) # add to veh_ids2recycle the IDs of the cars that left in this cycle
                printf (loc_output_file, '\nnew_or_moved: ')
                for item in cur_list_of_vehs:
                    filtered_list = list (filter (lambda veh : veh['key'] == item['key'], veh_key2id)) # look for this veh in the list of already-known vehs
                    if (len(filtered_list) == 0): # first time this veh_key appears in the simulated area
                        veh_type = 'n' # will indicate that this is a new vehicle 
                        if (len (veh_ids2recycle) > 0): # can recycle an ID of a veh that left
                            # veh_type = 'r' # will indicate that this is a recycled vehicle's id 
                            veh_id     = veh_ids2recycle.pop(0) # recycle an ID of a veh that left, and remove it from the list of veh IDs to recycle
                        else:
                            # veh_type = 'n' # will indicate that this is a new vehicle 
                            veh_id = len (veh_key2id) # pick a new ID
                        veh_key2id.append({'key' : item['key'], 'id' : veh_id}) # insert the new veh into the db 
                    else: # already seen this veh_key in the sim' --> extract its id from the hash
                        if (traci.vehicle.getSpeed(item['key']) == 0):  
                            continue
                        veh_type = 'o' # will indicate that this is a old vehicle 
                        veh_id = filtered_list[0]['id'] 
                    if (VERBOSE_SPEED in self.verbose): 
                        printf (loc_output_file, "({},{},{},{},{:.0f})" .format (veh_type, veh_id, item['pos'][0], item['pos'][1], traci.vehicle.getSpeed(item['key'])))
                    elif (VERBOSE_LOC in self.verbose):
                        printf (loc_output_file, "({},{},{},{})" .format (veh_type, veh_id, item['pos'][0], item['pos'][1]))
        
                sys.stdout.flush()
                traci.simulationStep (self.t + len_of_time_slot_in_sec)
        traci.close()

    def gen_antloc_file (self, antenna_locs_file_name, provider=''):
        """
        Parse a raw antenna location file (downloaded from https://opencellid.org/), and extract for each antenna its X,Y position in the given SUMO configuration.
        Outputs a ".antloc" file, that details the x,y position of a all antennas within the simulated area.
        """
        
        antenna_loc_file = open ('../res/Antennas_locs/' + antenna_locs_file_name, 'r')

        traci.start (self.mysumoCmd())

        self.list_of_antennas = []
    
        for line in antenna_loc_file: 

            if (line == "\n" or line.split ("//")[0] == ""): # skip lines of comments and empty lines
                continue
            
            splitted_line = line.split(',')
            if (splitted_line[0]=='radio'):
                continue

            pos     = self.abs_to_relative_pos (self.rotate_point (traci.simulation.convertGeo (float (splitted_line[lon_pos_idx]), float (splitted_line[lat_pos_idx]), True)))
            radio   = splitted_line[radio_idx]
            mnc     = splitted_line[mnc_idx]
            
            if (not (loc2poa_c.is_in_simulated_area(self.city, pos))):# or (not (radio=='LTE'))):  
                continue
        
            self.list_of_antennas.append ({'radio' : radio, 'mnc' : mnc, 'x' : pos[0], 'y' : pos[1]})    
            
        traci.close()
        
        # self.list_of_antennas = list (filter (lambda item : item['radio']=='LTE', self.list_of_antennas)) # filter-out all non-LTE antennas
        print ('Num of antennas in the simulated area={}' .format (len(self.list_of_antennas)))
        
        PoAs_loc_file     = open ('../res/Antennas_locs/' + antenna_locs_file_name.split('.')[0] + ('.center.' if self.city=='Lux' else '.') + provider + '.antloc', 'w')
        printf (PoAs_loc_file, '// format: ID,X,Y\n// where X,Y is the position of the antenna in the given SUMO file\n')
        printf (PoAs_loc_file, '// Parsed antenna location file: {}\n' .format (antenna_locs_file_name))
        printf (PoAs_loc_file, '// SUMO cfg file: {}\n' .format (self.sumo_cfg_file))

        poa_id = 0
        antennas_to_print = list (filter (lambda item : item['mnc']==self.providers_mnc[provider], self.list_of_antennas)) 
        for poa in antennas_to_print:                       

            printf (PoAs_loc_file, '{},{},{}\n' .format (poa_id, poa['x'], poa['y']))
            poa_id += 1

    def Poly_of_rect_i_j (self, i, j):
        """
        Returns the polygon corresponding to the (i,j)-rectangle, when partitioning the simulated area into 4**self.max_power_of_4 * loc2poa_c.NUM_OF_TOP_LVL_SQS rectangles. 
        """
        
        # Position in this order Top Left, Top Right, Bottom Right, Bottom Left, Top Left
        # ROI = Polygon(rectangle)




    def calc_tot_lane_len_in_all_cells (self, max_power_of_4=0):
        
        self.max_power_of_4 = max_power_of_4
        traci.start(self.LaneLengthSumoCmd())
        net = sumolib.net.readNet(netFile[self.city])  # net file

        num_of_rows_in_tile = 2**max_power_of_4
        num_of_cols_in_tile = num_of_rows_in_tile * loc2poa_c.NUM_OF_TOP_LVL_SQS 
        
        tot_len_of_lanes_in_rect = np.empty ([num_of_rows_in_tile, num_of_cols_in_tile])
        
        for i in range (num_of_rows_in_tile):
            for j in range (num_of_cols_in_tile):
                self.tot_lane_len_in_rect[i][j] = self.tot_lane_len_in_rect (net, self.poly_of_i_j ()) 

                
            
        traci.close()

    def tot_lane_len_in_rect (self, net, ROI):
        """
        Calculate the total lengths of lanes in it.
        Input: 
        net - a net file
        rectangle - 4 corners, given in order: Top Left, Top Right, Bottom Right, Bottom Left. 
        The function assumes that a Traci simulation is already running 
        """

        totalLength = 0 # total length of lanes under the region of interest
        
        for edge in traci.edge.getIDList():
            # if edge ID starts with : then its a junction according to SUMO docs.
            if edge[0] == ":":
                # avoiding the junctions
                continue
            curEdge = net.getEdge(edge)
            # get bounding box of the edge
            curEdgeBBCoords = curEdge.getBoundingBox()
            # create the bounding box geometrically
            curEdgeBBox = box(*curEdgeBBCoords)

            # print ('len={:.1f}. num of lanes={}. area={:.1f}' .format (curEdge.getLength(), len(curEdge.getLanes()), curEdgeBBox.area))
            
            if ROI.contains(curEdgeBBox): # The given polygon contains that edge, so add the edge's length, multiplied by the # of lanes

                totalLength += curEdge.getLength() * len(curEdge.getLanes())
            
            # If ROI intersects with this edge then, as a rough estimation of the relevant length to add, divide the intersecting area by the total edge area
            elif (ROI.intersects(curEdgeBBox)):   
                
                totalLength += curEdge.getLength() * (ROI.intersection(curEdgeBBox).area / curEdgeBBox.area) 
                
        return totalLength

if __name__ == '__main__':
    
    city = 'Monaco'
    my_Traci_runner = Traci_runner (sumo_cfg_file='myLuST.sumocfg' if city=='Lux' else 'myMoST.sumocfg')
    my_Traci_runner.tot_lane_len_in_rect (rectangle=loc2poa_c.SIMULATED_AREA_RECT[city])
    # my_Traci_runner.print_lon_lat_corners_of_simulated_area()
    # my_Traci_runner.gen_antloc_file ('Monaco.txt', provider='Telecom')
    # my_Traci_runner.simulate (warmup_period=(3600*7.5), sim_length = 3600, len_of_time_slot_in_sec = 60, verbose=[VERBOSE_LOC, VERBOSE_SPEED]) #warmup_period = 3600*7.5
    # my_Traci_runner.simulate_to_cnt_vehs_only (warmup_period=(3), sim_length = 3600, len_of_time_slot_in_sec =1, verbose=[])
    # my_Traci_runner.print_lon_lat_corners_of_simulated_area()
    # my_Traci_runner.print_abs_pos_corners_of_simulated_area ()
