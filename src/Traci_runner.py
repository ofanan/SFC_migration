import sumolib
from datetime import datetime
from   tictoc import tic, toc
from sumolib import checkBinary  
import traci, sys, math, pickle
import numpy as np, matplotlib.pyplot as plt
from shapely.geometry import Polygon, box

# My own format print functions 
from printf import printf, printmat, invert_mat_bottom_up
from secs2hour import secs2hour
import loc2poa_c

# verbose enums that define the type of output to be generated
VERBOSE_LOC      = 2 # print-out the location of each vehicle in the chosen simulated area
VERBOSE_SPEED    = 3 # print-out the speed of each vehicle in the chosen simulated area

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


    #Given the position of a vehicle (pos[0], pos[1]), returns the poa of a veh found in this position
    pos2poa = lambda self, pos: self.my_loc2poa.loc2poa (pos[0], pos[1])

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
    my_sumo_cmd = lambda self, len_of_time_slot_in_sec=1 : [checkBinary('sumo'), '-c', self.sumo_cfg_file, '-W', '-V', 'false', '--no-step-log', 'true', '--step-length', '{:.3f}' .format (len_of_time_slot_in_sec)]

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
        traci.start (self.my_sumo_cmd())
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
        print ('Running Traci for the period {} to count vehs only' .format (time_str))
        traci.start(self.my_sumo_cmd())
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
        Simulate Traci, and print-out the locations and/or speed of cars, as defined by the verbose input.
        The output is written to '/../res/loc_files/loc_output_filename.loc', where loc_outputfilename.loc is a name that is automatically generated, and
        reflects the settings of the simulation, namely the city (Lux/Monaco); the simulated time, at a 24h format (e.g., 0000_0030); and the decision period. 
        The .loc files expresses the location of each vehicle at each decision period.
        The .loc files can be later converted to .poa files, that associates each vehicle that arrived/left/changed a cell to its current poa (Point Of Access).
        Inputs: 
        - warmup_period - The simulation defaults to start at 0000 (00 am). To begin at a later hour, specify the number of seconds after 00:00 after which the sim begins.
        - sim_length - in seconds.
        - len_of_time_slot_in_sec.
        - num_of_output_files - for case when the generated output files are too large, you may specify here a value > 1, to write the output sequentially to multiple files.
        - verbose - a list of the required outputs. E.g., VERBOSE_LOC to output the vehicles' location; VERBOSE_SPEED, to output the vehicles' speeds. 

        """

        veh_key2id               = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
        veh_ids2recycle          = [] # will hold a list of ids that are not used anymore, and therefore can be recycled (used by new users == garbage collection).
        vehs_left_in_this_cycle  = []
        self.verbose             = verbose
        
        time_str = self.gen_time_str (warmup_period, warmup_period+sim_length)
        traci.start (self.my_sumo_cmd(len_of_time_slot_in_sec))
        print ('Running Traci for the period {}. Will write res to {} output files' .format (time_str, num_of_output_files))
        
        if (warmup_period > 0):
            traci.simulationStep (float(warmup_period)) # simulate without output until our required time (time starts at 00:00). 
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
                printf (loc_output_file, '\nt = {:.2f}\n' .format (self.t))

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

    def simulate_gen_poa_file (self, warmup_period=0, sim_length_in_sec=10, len_of_time_slot_in_sec=1, num_of_output_files=1, verbose = [], max_power_of_4=None):
        """
        Simulate Traci, and print-out the locations and/or speed of cars, as defined by the verbose input.
        The output is written to '/../res/poa_files/poa_output_filename.loc', where poa_outputfilename.loc is a name that is automatically generated, and
        reflects the settings of the simulation, namely the city (Lux/Monaco); the simulated time, at a 24h format (e.g., 0000_0030); and the decision period. 
        Inputs: 
        - warmup_period - The simulation defaults to start at 0000 (00 am). To begin at a later hour, specify the number of seconds after 00:00 after which the sim begins.
        - sim_length - in seconds.
        - len_of_time_slot_in_sec.
        - num_of_output_files - for case when the generated output files are too large, you may specify here a value > 1, to write the output sequentially to multiple files.
        - verbose - a list of the required outputs. E.g., VERBOSE_LOC to output the vehicles' location; VERBOSE_SPEED, to output the vehicles' speeds. 
        """
        if (max_power_of_4==None):
            max_power_of_4 = 4 if self.city=='Lux' else 3 # default values 
        else: 
            if (max_power_of_4>4):
                print ('Error: max_power_of_4 should be at most 4')
                exit ()
            elif (max_power_of_4==4 and self.city=='Monaco'):
                print ('Error: for running Monaco scenario, max_power_of_4 should be at most 3')
                exit ()
        self.my_loc2poa = loc2poa_c.loc2poa_c (max_power_of_4=max_power_of_4, verbose = [], antloc_file_name = 'Lux.post.antloc' if self.city=='Lux' else 'Monaco.Telecom.antloc')
        known_vehs               = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
        veh_ids2recycle          = [] # will hold a list of ids that are not used anymore, and therefore can be recycled (used by new users == garbage collection).
        vehs_left_in_this_cycle  = []
        self.verbose             = verbose
        
        time_str = self.gen_time_str (warmup_period, warmup_period+sim_length_in_sec)
        traci.start (self.my_sumo_cmd(len_of_time_slot_in_sec))
        print ('Running Traci for the period {}. Will write res to {} output files' .format (time_str, num_of_output_files))
        
        if (warmup_period > 0):
            tic()
            traci.simulationStep (float(warmup_period)) # simulate without output until our required time (time starts at 00:00).
            print ("running warmup period of {} sec took" .format (warmup_period), toc (returnString=True))
        for i in range (num_of_output_files):
            
            poa_output_file_name = '../res/poa_files/{}_{}_{}secs.poa' .format (self.city, time_str, len_of_time_slot_in_sec)   
            with open(poa_output_file_name, 'w') as poa_output_file:
                poa_output_file.write('')                
            poa_output_file  = open (poa_output_file_name,  "w")
            printf (poa_output_file, '// File format:\n//for each time slot:\n')
            printf (poa_output_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (poa_output_file, '//"new_usrs" is a list of the new usrs, and their PoAs, e.g.: (0, 2)(1,3) means that new usr 0 is in cell 2, and new usr 1 is in cell 3.\n')
            printf (poa_output_file, '//"old_usrs" is a list of the usrs who moved to another cell in the last time slot, and their current PoAs, e.g.: (0, 2)(1,3) means that old usr 0 is now in cell 2, and old usr 1 is now in cell 3.\n')

            tic()
            while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
                
                self.t = traci.simulation.getTime()
                
                # Finished the sim. Now, just make some post-processing. 
                if (self.t >= warmup_period + sim_length_in_sec*(i+1) / num_of_output_files):
                    print ('Successfully finished writing to file {}' .format (poa_output_file_name))
                    break
                  
                # By default, the type of each  vechicle is 'u', namely, *Undefined". 
                cur_list_of_vehs = [{'key' : veh_key, 'type' : 'u', 'nxt_poa' : self.pos2poa (self.get_relative_position(veh_key))} for veh_key in traci.vehicle.getIDList() if loc2poa_c.is_in_simulated_area (self.city, self.get_relative_position(veh_key))]            

                vehs_left_in_this_cycle = list (filter (lambda veh : (veh['key'] not in [item['key'] for item in cur_list_of_vehs] and 
                                                                   veh['id']  not in (veh_ids2recycle)), known_vehs)) # The list of vehs left at this cycle includes all vehs that are not in the list of currently-active vehicles, and haven't already been listed as "vehs that left" (i.e., veh ids to recycle). 
                known_vehs = list (filter (lambda veh : veh['id'] not in [veh['id'] for veh in vehs_left_in_this_cycle], known_vehs)) # remove usrs that left from the known_vehs map 
                if (len (vehs_left_in_this_cycle) > 0):
                    veh_ids2recycle = sorted (list (set (veh_ids2recycle) | set([item['id'] for item in vehs_left_in_this_cycle]))) # add to veh_ids2recycle the IDs of the cars that left in this cycle
                for item in cur_list_of_vehs:
                    filtered_list = list (filter (lambda veh : veh['key'] == item['key'], known_vehs)) # look for this veh in the list of already-known vehs
                    if (len(filtered_list) == 0): # first time this veh_key appears in the simulated area
                        item['type'] = 'new' # this is a new vehicle 
                        if (len (veh_ids2recycle) > 0): # can recycle an ID of a veh that left
                            known_vehs.append({'key' : item['key'], 'id' : veh_ids2recycle.pop(0), 'cur_poa' : item['nxt_poa']}) # recycle an ID of a veh that left, and remove it from the list of veh IDs to recycle; prepare for the next cycle, by setting known_vehs[cur_poa] <-- nxt_poa  
                        else:
                            known_vehs.append({'key' : item['key'], 'id' : len (known_vehs), 'cur_poa' : item['nxt_poa']}) # pick a new ID; prepare for the next cycle, by setting known_vehs[cur_poa] <-- nxt_poa
                    else: # already seen this veh_key in the sim' --> extract its id from the hash
                        if (traci.vehicle.getSpeed(item['key'])== 0): # the veh hasn't moved since the last slot, so it also didn't change its loc and poa association  
                            continue
                        item['type'] = 'old' # will indicate that this is a old vehicle 
                        item['id'] = filtered_list[0]['id']

                new_usrs = [] # will hold a list of pairs of the form (v,p), where v is a new usr, and p is his poa
                for veh in list (filter (lambda veh :veh['type']=='new', cur_list_of_vehs)):
                    list_of_items_in_known_vehs = list (filter (lambda item_in_known_vehs : item_in_known_vehs['key'] == veh['key'], known_vehs)) # look for this veh in the list of already-known vehs
                    if (len(list_of_items_in_known_vehs)!=1):
                        print ("error: wrong number of items with key {} in known_vehs" .format (veh['key']))
                        exit ()
                    new_usrs.append ([list_of_items_in_known_vehs[0]['id'], veh['nxt_poa']])
                moved_usrs = [] # will hold a list of pairs of the form (v,p), where v is a usr who changed poa, and p is his new poa  
                for veh in list (filter (lambda veh :veh['type']=='old', cur_list_of_vehs)):
                    list_of_items_in_known_vehs = list (filter (lambda old_veh : old_veh['key'] == veh['key'], known_vehs)) # look for this veh in the list of already-known vehs
                    if (len(list_of_items_in_known_vehs)!=1): 
                        print ("error: wrong number of items with key {} in known_vehs" .format (veh['key']))
                        exit ()
                    if (list_of_items_in_known_vehs[0]['cur_poa']!=veh['nxt_poa']):
                        moved_usrs.append ([list_of_items_in_known_vehs[0]['id'], veh['nxt_poa']])
                        list_of_items_in_known_vehs[0]['cur_poa'] = veh['nxt_poa']

                if (len(vehs_left_in_this_cycle)==0 and len(new_usrs)==0 and len(moved_usrs)==0):
                    traci.simulationStep (self.t + len_of_time_slot_in_sec)
                    continue # nothing to report at this slot
                printf (poa_output_file, '\nt = {:.3f}' .format (self.t))
                if (len(vehs_left_in_this_cycle)>0):
                    printf (poa_output_file, '\nusrs_that_left: ')
                    for veh in vehs_left_in_this_cycle:
                        printf (poa_output_file, '{:.0f} ' .format(veh['id']))
                if (len(new_usrs)>0):
                    printf (poa_output_file, '\nnew_usrs: ')
                    for item in new_usrs:
                        printf (poa_output_file, '({},{})' .format (item[0],item[1]))
                printf (poa_output_file, '\nold_usrs: ')
                for item in moved_usrs:
                    printf (poa_output_file, '({},{})' .format (item[0],item[1]))
                sys.stdout.flush()
                traci.simulationStep (self.t + len_of_time_slot_in_sec)
        traci.close()
        print ("running {} sec using steps of {} sec took" .format (sim_length_in_sec, len_of_time_slot_in_sec), toc (returnString=True))

    def gen_antloc_file (self, antenna_locs_file_name, provider=''):
        """
        Parse a raw antenna location file (downloaded from https://opencellid.org/), and extract for each antenna its X,Y position in the given SUMO configuration.
        Outputs a ".antloc" file, that details the x,y position of a all antennas within the simulated area.
        """
        
        antenna_loc_file = open ('../res/Antennas_locs/' + antenna_locs_file_name, 'r')

        traci.start (self.my_sumo_cmd())

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

    def calc_tot_lane_len_in_all_rects (self, num_of_lvls=None):
        """
        Calculates the overall length of all lanes within all the rectangles in the simulated area.
        Prints the results to output .txt and .pcl file, and possibly plots the rectangles, and saves the plots.  
        """

        txt_output_file_name = '{}_lanes_len.txt' .format (self.city)                
        lanes_len_output_file = open ('../res/{}' .format(txt_output_file_name), 'a')
        tot_len_of_lanes = []
        traci.start(self.LaneLengthSumoCmd())
        self.net = sumolib.net.readNet(netFile[self.city])  # net file

        if (num_of_lvls==None):
            num_of_lvls = 3 if self.city=='Monaco' else 4
        else:
            num_of_lvls=num_of_lvls
        colors = {'Monaco' : ['black', 'blue', 'red', 'green'], 'Lux' : ['black', 'blue', 'red', 'green', 'cyan']}
        plt.figure()
    
        for max_power_of_4 in range (num_of_lvls+1):
            num_of_rows_in_tile = 2**max_power_of_4
            num_of_cols_in_tile = num_of_rows_in_tile * loc2poa_c.NUM_OF_TOP_LVL_SQS[self.city]
            rect_x_edge = loc2poa_c.X_EDGE[self.city] / num_of_cols_in_tile 
            rect_y_edge = loc2poa_c.Y_EDGE[self.city] / num_of_rows_in_tile 
    
            printf (lanes_len_output_file, '// City={}\n' .format (self.city))                    
            tot_len_of_lanes_in_rect = np.empty ([num_of_rows_in_tile, num_of_cols_in_tile])
            
            for row in range (num_of_rows_in_tile):
                lower_left_corner = loc2poa_c.SIMULATED_AREA_RECT[self.city][loc2poa_c.lower_left_idx] + row * rect_y_edge # the lower left of the next rect to consider is the low leftmost point in this row  
                for col in range (num_of_cols_in_tile):
                    # Coordinates (corners) of the current rectangle. To have a closed shape, we go from the lower left clock-wise, and end with the lower-left. 
                    coord =    [lower_left_corner + rect_y_edge,  
                                lower_left_corner + rect_x_edge + rect_y_edge,
                                lower_left_corner + rect_x_edge,
                                lower_left_corner,
                                lower_left_corner + rect_y_edge]
                    tot_len_of_lanes_in_rect[row][col] = self.tot_lane_len_in_rect (Polygon (coord))
                    # plt.scatter ((lower_left_corner + rect_y_edge)[0], (lower_left_corner + rect_y_edge)[1], c=colors[self.city][max_power_of_4]) 
                    # plt.scatter ((lower_left_corner + rect_x_edge + rect_y_edge)[0], (lower_left_corner + rect_x_edge + rect_y_edge)[1], c=colors[self.city][max_power_of_4] ) 
                    # plt.scatter ((lower_left_corner + rect_x_edge + rect_y_edge)[0], (lower_left_corner + rect_x_edge)[1], c=colors[self.city][max_power_of_4]) 
                    # plt.scatter ((lower_left_corner)[0], (lower_left_corner)[1], c=colors[self.city][max_power_of_4])
                    # plt.scatter ([lower_left_corner + rect_y_edge][0], [lower_left_corner + rect_y_edge][1]) # lower_left_corner + rect_x_edge + rect_y_edge, lower_left_corner + rect_x_edge,lower_left_corner])
                    
                    xs, ys = zip(*coord) #create lists of x and y values

                    plt.plot(xs,ys, c=colors[self.city][max_power_of_4]) 
                    lower_left_corner += rect_x_edge
            
            tot_len_of_lanes_in_rect = invert_mat_bottom_up (tot_len_of_lanes_in_rect)
            num_of_rects = num_of_rows_in_tile * num_of_cols_in_tile
            printf (lanes_len_output_file, '// num_of_rects={}. per_rect_lanes_len=\n' .format (num_of_rects))
            printmat (lanes_len_output_file, tot_len_of_lanes_in_rect, my_precision=2) # Print the total length in kms
            tot_len_of_lanes.append ({'num_of_rects' : num_of_rects, 'tot_len_of_lanes_in_rect' : tot_len_of_lanes_in_rect})
            # tot_len_of_lanes.append ({'{}' .format (num_of_rects) : tot_len_of_lanes_in_rect})
        
        with open ('../res/' + txt_output_file_name.split('.txt')[0] + '.pcl', 'wb') as pcl_output_file:
            pickle.dump (tot_len_of_lanes, pcl_output_file)
        plt.show ()
        traci.close()

    def tot_lane_len_in_rect (self, polygon):
        """
        Calculate the total lengths of lanes in it.
        Input: 
        polygon - a polygon representation of a rectangle. 
        The function assumes that a Traci simulation is already running
        Output - the total lengths of lanes within this polygon [km]
        """

        totalLength = 0 # total length of lanes under the region of interest
        
        for edge in traci.edge.getIDList():
            # if edge ID starts with : then its a junction according to SUMO docs.
            if edge[0] == ":":
                # avoiding the junctions
                continue
            curEdge = self.net.getEdge(edge)
            # get bounding box of the edge
            curEdgeBBCoords = curEdge.getBoundingBox()
            # create the bounding box geometrically
            curEdgeBBox = box(*curEdgeBBCoords)

            # print ('len={:.1f}. num of lanes={}. area={:.1f}' .format (curEdge.getLength(), len(curEdge.getLanes()), curEdgeBBox.area))
            
            if polygon.contains(curEdgeBBox): # The given polygon contains that edge, so add the edge's length, multiplied by the # of lanes

                totalLength += curEdge.getLength() * len(curEdge.getLanes())
                        
            elif (polygon.intersects(curEdgeBBox)): # If polygon intersects with this edge then, as a rough estimation of the relevant length to add, divide the intersecting area by the total edge area   
                
                totalLength += curEdge.getLength() * len(curEdge.getLanes()) * (polygon.intersection(curEdgeBBox).area / curEdgeBBox.area) 
                
        return totalLength/1000

if __name__ == '__main__':
    city = 'Lux'
    my_Traci_runner = Traci_runner (sumo_cfg_file='myLuST.sumocfg' if city=='Lux' else 'myMoST.sumocfg')
    my_Traci_runner.simulate_gen_poa_file (warmup_period=0*3600, sim_length_in_sec=1, len_of_time_slot_in_sec=0.50, num_of_output_files=1, verbose = [VERBOSE_LOC])
    # my_Traci_runner.simulate (warmup_period=1, sim_length=50, len_of_time_slot_in_sec=1, num_of_output_files=1, verbose = [VERBOSE_LOC])
    # my_Traci_runner.simulate (warmup_period=1*3600, sim_length=2, len_of_time_slot_in_sec=0.5, num_of_output_files=1, verbose = [VERBOSE_LOC])

    # for T in [6]: #[3, 5, 6, 7, 9, 10]:
    #     my_Traci_runner = Traci_runner (sumo_cfg_file='myLuST.sumocfg' if city=='Lux' else 'myMoST.sumocfg')
    #     my_Traci_runner.simulate (warmup_period=7.5*3600, sim_length=3600, len_of_time_slot_in_sec=T, num_of_output_files=1, verbose = [VERBOSE_LOC])
    # my_Traci_runner.calc_tot_lane_len_in_all_rects ()
    # my_Traci_runner.print_lon_lat_corners_of_simulated_area()
    # my_Traci_runner.gen_antloc_file ('Monaco.txt', provider='Telecom')
    # my_Traci_runner.simulate (warmup_period=(3600*7.5), sim_length = 3600, len_of_time_slot_in_sec = 60, verbose=[VERBOSE_LOC, VERBOSE_SPEED]) #warmup_period = 3600*7.5
    # my_Traci_runner.simulate_to_cnt_vehs_only (warmup_period=(3), sim_length = 3600, len_of_time_slot_in_sec =1, verbose=[])
    # my_Traci_runner.print_lon_lat_corners_of_simulated_area()
    # my_Traci_runner.print_abs_pos_corners_of_simulated_area ()
