from sumolib import checkBinary  
import traci  
import sys
import numpy as np
import math
# import matplotlib.pyplot as plt

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

class Traci_runner (object):

    # Given the x,y position, return the x,y position within the simulated area (city center) 
    relative_to_abs_pos = lambda self, pos: np.array(pos, dtype='int16') + loc2poa_c.LOWER_LEFT_CORNER [self.city]

    # Given the x,y position, return the x,y position within the simulated area (city center) 
    abs_to_relative_pos = lambda self, pos: np.array(pos, dtype='int16') - loc2poa_c.LOWER_LEFT_CORNER [self.city]

    # Returns the relative location of a given vehicle ID. The relative location found after rotating the point (if needed), and then position it w.r.t. the lower left (south-west) corner of the simulated area.
    get_relative_position = lambda self, veh_key  : self.abs_to_relative_pos (self.rotate_point (traci.vehicle.getPosition(veh_key))) 

    # rotate a given point by self.rotate_angle radians counter-clockwise around self.pivot
    # rotate_point = lambda self, point : self.handle_Nan_point() if (math.isnan(point[0]) or math.isnan(point[0])) else (np.array (point if (self.rotate_angle==0) else [self.pivot[0] + math.cos(self.rotate_angle) * (point[0] - self.pivot[0]) - math.sin(self.rotate_angle) * (point[1] - self.pivot[1]),self.pivot[1] + math.sin(self.rotate_angle) * (point[0] - self.pivot[0]) + math.cos(self.rotate_angle) * (point[1] - self.pivot[1])], dtype='int16'))

    # Given the lon, lat coordinates of a point, return the (x,y) coordinates of its relative position within the simulated area 
    lon_lat_to_relative_pos = lambda self, lon, lat : self.abs_to_relative_pos (self.rotate_point (traci.simulation.convertGeo (lon, lat, True)))

    # Generate a string, showing the start and end time of the sim, in a 24-h clock
    gen_time_str = lambda self, start_time, end_time : '{}_{}' .format (secs2hour(start_time), secs2hour(end_time)) 
                    
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
    
    def relative_pos_to_lon_lat (self, point):
        """
        Given the relative position of a point (x,y) within the simulated area, return its [latitude, longitude] 
        """
        if (math.isnan(point[0]) or math.isnan(point[0])):
            self.handle_Nan_point()
            return 
        if (not (loc2poa_c.is_in_simulated_area(self.city, point))):
            print ('Warning: relative_pos_to_lon_lat was called with point {} which is outside the simulated area' .format(point))
            return
        
        # Now we know that this point is within the simulated area
        abs_point = self.relative_to_abs_pos (point)
        # print ('abs_point={}' .format (abs_point))

        rttd_back_point = self.rotate_point (abs_point, -self.rotate_angle)
        return traci.simulation.convertGeo (rttd_back_point[0],rttd_back_point[1])
            
    def print_lon_lat_corners_of_simulated_area (self):
        
        """
        Print the latitude and longitude of the 4 corners of the simulated area
        """
        traci.start([checkBinary('sumo'), '-c', self.sumo_cfg_file, '-W', '-V', 'false', '--no-step-log', 'true'])
        EPSILON=0.01
        min_x, min_y = loc2poa_c.MIN_X[self.city] + EPSILON, loc2poa_c.MIN_Y[self.city] + EPSILON
        max_x, max_y = loc2poa_c.MAX_X[self.city] - EPSILON, loc2poa_c.MAX_Y[self.city] - EPSILON
        sw_corner = self.relative_pos_to_lon_lat ([min_x, min_y])
        print ('sw_corner(lat,lon)={}, {}' .format (sw_corner[1], sw_corner[0]))
        # print ('Check: casting back={}' .format(self.lon_lat_to_relative_pos(sw_corner[0], sw_corner[1])))

        nw_corner = self.relative_pos_to_lon_lat ([min_x, max_y])
        print ('nw_corner(lat,lon)={}, {}' .format (nw_corner[1],nw_corner[0]))
        # print ('Check: casting back={}' .format(self.lon_lat_to_relative_pos(nw_corner[0], nw_corner[1])))

        se_corner = self.relative_pos_to_lon_lat ([max_x, min_y])
        print ('se_corner(lat,lon)={}, {}' .format (se_corner[1],se_corner[0]))
        # print ('Check: casting back={}' .format(self.lon_lat_to_relative_pos(se_corner[0], se_corner[1])))

        ne_corner = self.relative_pos_to_lon_lat ([max_x, max_y])
        print ('ne_corner(lat,lon)={}, {}' .format (ne_corner[1],ne_corner[0]))
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
        
        self.rotate_angle = -math.radians(54) if self.city=='Monaco' else 0 # angle to rotate the points. The requested angle degrees of clcokwise is converted to the radians value of rotating counter-clockwise used by rotate_point.
        self.pivot = [loc2poa_c.GLOBAL_MAX_X[self.city]/2, loc2poa_c.GLOBAL_MAX_Y[self.city]/2] # pivot point, around which the rotating is done

    def simulate_to_cnt_vehs_only (self, warmup_period=0, sim_length=10, len_of_time_slot_in_sec=1, verbose = []):
        """
        Simulate a fast Traci SUMO simulation, to cnt the number of active vehicles/pedestrians at each slot; and the total number of distinct cars along the sim.
        """       
        self.verbose            = verbose
        
        time_str = self.gen_time_str (warmup_period, warmup_period+sim_length)
        print ('Running Traci for the period {}' .format (time_str))
        traci.start([checkBinary('sumo'), '-c', self.sumo_cfg_file, '-W', '-V', 'false', '--no-step-log', 'true'])
        self.cnt_output_file_name = '../res/{}_{}_{}secs_cnt.res' .format (self.city, time_str, len_of_time_slot_in_sec) 
        self.cnt_output_file      = open (self.cnt_output_file_name, 'w')
               
        if (warmup_period > 0):
            traci.simulationStep (int(warmup_period)) # simulate without output until our required time (time starts at 00:00). 
            
        known_veh_keys      = () # will hold the set of known vehicles' keys
        num_of_vehs_in_slot = []  # num_of_vehs_in_slot [t] will hold the number of distinct vehs at each slot t          
            
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
            num_of_vehs_in_slot.append (len(known_veh_keys))
                   
            traci.simulationStep (self.t + len_of_time_slot_in_sec)
        traci.close()
        
        printf (self.cnt_output_file, 'avg_num_of_vehs_in_simulated_area={:.2f}\nmax_num_of_vehs_in_simulated_area={}\n' .format (np.average(num_of_vehs_in_slot), max (num_of_vehs_in_slot))) 
        printf (self.cnt_output_file, 'num_of_vehs_in_simulated_area={}\n' .format (num_of_vehs_in_slot)) 

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
        traci.start([checkBinary('sumo'), '-c', self.sumo_cfg_file, '-W', '-V', 'false', '--no-step-log', 'true'])
        
        if (warmup_period > 0):
            traci.simulationStep (int(warmup_period)) # simulate without output until our required time (time starts at 00:00). 
        for i in range (num_of_output_files):
            
            loc_output_file_name = '../res/loc_files/{}_{}_{}secs{}.loc' .format (self.city, time_str, len_of_time_slot_in_sec, ('_rttd{}' .format (-math.degrees(self.rotate_angle)) if (self.rotate_angle>0) else ''))   
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

        traci.start([checkBinary('sumo'), '-c', self.sumo_cfg_file, '-W', '-V', 'false', '--no-step-log', 'true'])

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

if __name__ == '__main__':
    
    my_Traci_runner = Traci_runner (sumo_cfg_file='myMoST.sumocfg')
    # my_Traci_runner.print_lon_lat_corners_of_simulated_area()
    # my_Traci_runner.gen_antloc_file ('Monaco.txt', provider='Telecom')
    # my_Traci_runner.simulate (warmup_period=(3600*7.5), sim_length = 3600, len_of_time_slot_in_sec = 60, verbose=[VERBOSE_LOC]) #warmup_period = 3600*7.5
    my_Traci_runner.simulate_to_cnt_vehs_only (warmup_period=(3600*7.5), sim_length = 3600, len_of_time_slot_in_sec =1, verbose=[])
     
