import optparse
import random
from sumolib import checkBinary  
import traci  
import sys
import numpy as np

from printf import printf 
from secs2hour import secs2hour
import loc2ap_c

VERBOSE_LOC      = 2
VERBOSE_SPEED    = 3

class Traci_runner (object):

    # Returns the relative location of a given vehicle (namely, its position relatively to the smaller left corner of ths simulated area),
    get_relative_position = lambda self, veh_key  : np.array(traci.vehicle.getPosition(veh_key), dtype='int16') - loc2ap_c.LOWER_LEFT_CORNER

    # Checks whether the given vehicle is within the simulated area.
    # Input: key of a vehicle.
    # Output: True iff this vehicle is within the simulated area.
    is_in_simulated_area  = lambda self, position : False if (position[0] <= 0 or position[0] >= loc2ap_c.MAX_X or position[1] <= 0 or position[1] >= loc2ap_c.MAX_Y) else True
    
    is_in_lux_global_area = lambda self, position : False if (position[0] <= 0 or position[0] >= loc2ap_c.LUX_GLOBAL_MAX_X or position[1] <= 0 or position[1] >= loc2ap_c.LUX_GLOBAL_MAX_YY) else True
    
    def __init__ (self, sumo_cfg_file='LuST.sumocfg'):
        self.sumo_cfg_file = sumo_cfg_file

    def simulate_to_cnt_vehs_only (self, warmup_period=0, sim_length=10, len_of_time_slot_in_sec=1, verbose = []):
        """
        Simulate a SUMO simulation using Traci, and cnt the number of active cars at each slot; and the total number of distinct cars along the sim.
        """       
        self.verbose            = verbose
        
        traci.start([checkBinary('sumo'), '-c', self.sumo_cfg_file, '-W', '-V', 'false', '--no-step-log', 'true'])
        print ('Running Traci on the period from {:.0f} to {:.0f}' .format (warmup_period, warmup_period+sim_length))
        self.cnt_output_file_name = '../res/{}_{}secs_cnt.res' .format (secs2hour(traci.simulation.getTime()), len_of_time_slot_in_sec) 
        self.cnt_output_file      = open (self.cnt_output_file_name, 'w')
               
        if (warmup_period > 0):
            traci.simulationStep (int(warmup_period)) # simulate without output until our required time (time starts at 00:00). 
            
            known_veh_keys = () # will hold the set of known vehicles' keys            
                
            while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
                
                cur_sim_time = traci.simulation.getTime()
                
                # print ('cur_sim_time={}, warmup_period={}, sim_length={}' .format (cur_sim_time, warmup_period, sim_length))
                
                # Finished the sim. Now, just make some post-processing. 
                if (cur_sim_time >= warmup_period + sim_length):
                    print ('Number of distinct cars during the simulated period={}' .format (len(known_veh_keys)))
                    break
                
                # Union known_veh_keys with the set of vehicles' keys of this cycle
                cur_list_of_vehicles = [veh_key for veh_key in traci.vehicle.getIDList() if self.is_in_simulated_area (self.get_relative_position(veh_key))] # list of vehs currently found within the simulated area.
                printf (self.cnt_output_file, '{:.0f} ' .format (len(cur_list_of_vehicles)))
                sys.stdout.flush()

                known_veh_keys       = set (cur_list_of_vehicles) | set (known_veh_keys)
                       
                traci.simulationStep (cur_sim_time + len_of_time_slot_in_sec)
            print ('here')
        traci.close()

    def simulate (self, warmup_period=0, sim_length=10, len_of_time_slot_in_sec=1, num_of_output_files=1, verbose = []):
        """
        Simulate Traci, and print-out the locations and/or speed of cars, as defined by the verbose input
        """
    
        veh_key2id              = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
        ids2recycle             = [] # will hold a list of ids that are not used anymore, and therefore can be recycled
        left_in_this_cycle      = []
        self.verbose            = verbose
        
        traci.start([checkBinary('sumo'), '-c', 'my.sumocfg', '-W', '-V', 'false', '--no-step-log', 'true'])
        print ('Running Traci on the period from {:.0f} to {:.0f}. Will write res to {} output files' .format (warmup_period, warmup_period+sim_length, num_of_output_files))
        
        if (warmup_period > 0):
            traci.simulationStep (int(warmup_period)) # simulate without output until our required time (time starts at 00:00). 
        for i in range (num_of_output_files):
            
            output_file_name = '../res/{}_{}secs.loc' .format (secs2hour(traci.simulation.getTime()), len_of_time_slot_in_sec)  
            with open(output_file_name, 'w') as loc_output_file:
                loc_output_file.write('')                
            loc_output_file  = open (output_file_name,  "w")
            printf (loc_output_file, '// locations of vehicles. Format:\n')
            printf (loc_output_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (loc_output_file, '// format for vehicles that are new (just joined the sim), or moved:\n')
            if (VERBOSE_SPEED in self.verbose): 
                printf (loc_output_file, '// (veh_type,usr_id,x,y,s)   where:\n')
                printf (loc_output_file, '// veh_type is either o (old veh), or n (new veh in the sim). (x,y) are the coordinates of the vehicle with this usr_id. s is the speed of this user.\n')
            elif (VERBOSE_LOC in self.verbose):
                printf (loc_output_file, '// (veh_type,usr_id,x,y)   where:\n')
                printf (loc_output_file, '// veh_type is either o (old veh), or n (new veh in the sim). (x,y) are the coordinates of the vehicle with this usr_id.\n')
            
                
            while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
                
                cur_sim_time = traci.simulation.getTime()
                
                # Finished the sim. Now, just make some post-processing. 
                if (cur_sim_time >= warmup_period + sim_length*(i+1) / num_of_output_files):
                    print ('Successfully finished writing to file {}' .format (output_file_name))
                    break
                
                cur_list_of_vehicles = [veh_key for veh_key in traci.vehicle.getIDList() if self.is_in_simulated_area (self.get_relative_position(veh_key))] # list of vehs currently found within the simulated area.
                printf (loc_output_file, '\nt = {:.0f}\n' .format (cur_sim_time))
            
                left_in_this_cycle   = list (filter (lambda veh : (veh['key'] not in (cur_list_of_vehicles) and 
                                                                   veh['id']  not in (ids2recycle)), veh_key2id)) 
                veh_key2id = list (filter (lambda veh : veh['id'] not in [veh['id'] for veh in left_in_this_cycle], veh_key2id)) # remove usrs that left from the veh_key2id map 
                printf (loc_output_file, 'usrs_that_left: ')
                if (len (left_in_this_cycle) > 0):
                    for veh in left_in_this_cycle:
                        printf (loc_output_file, '{:.0f} ' .format(veh['id']))
                    ids2recycle = sorted (list (set (ids2recycle) | set([item['id'] for item in left_in_this_cycle]))) # add to ids2recycle the IDs of the cars that left in this cycle
                printf (loc_output_file, '\nnew_or_moved: ')
                for veh_key in cur_list_of_vehicles:
                    filtered_list = list (filter (lambda veh : veh['key'] == veh_key, veh_key2id)) # look for this veh in the list of already-known vehs
                    if (len(filtered_list) == 0): # first time this veh_key appears in the simulated area
                        veh_type = 'n' # will indicate that this is a new vehicle 
                        if (len (ids2recycle) > 0): # can recycle an ID of a veh that left
                            # veh_type = 'r' # will indicate that this is a recycled vehicle's id 
                            veh_id     = ids2recycle.pop(0) # recycle an ID of a veh that left, and remove it from the list of veh IDs to recycle
                        else:
                            # veh_type = 'n' # will indicate that this is a new vehicle 
                            veh_id = len (veh_key2id) # pick a new ID
                        veh_key2id.append({'key' : veh_key, 'id' : veh_id}) # insert the new veh into the db 
                    else: # already seen this veh_key in the sim' --> extract its id from the hash
                        if (traci.vehicle.getSpeed(veh_key) == 0):  
                            continue
                        veh_type = 'o' # will indicate that this is a old vehicle 
                        veh_id = filtered_list[0]['id'] 
                    position = self.get_relative_position (veh_key)
                    if (VERBOSE_SPEED in self.verbose): 
                        printf (loc_output_file, "({},{},{},{},{:.0f})" .format (veh_type, veh_id, position[0], position[1], traci.vehicle.getSpeed(veh_key)))
                    elif (VERBOSE_LOC in self.verbose):
                        printf (loc_output_file, "({},{},{},{})" .format (veh_type, veh_id, position[0], position[1]))
        
                sys.stdout.flush()
                traci.simulationStep (cur_sim_time + len_of_time_slot_in_sec)
        traci.close()


    def parse_antenna_locs_file (self, antenna_locs_file_name):
        
        antenna_loc_file = open ('../res/Antennas locs/' + antenna_locs_file_name, 'r')
        APs_loc_file     = open ('../res/Antennas locs/' + antenna_locs_file_name.split('.')[0] + 'locs_of_APs.txt', 'w')
        
        traci.start([checkBinary('sumo'), '-c', self.sumo_cfg_file, '-W', '-V', 'false', '--no-step-log', 'true'])
        AP_id = 0
        lon, lat = 6.120712,49.54384
        # print ('{},{}' .format (AP_id, traci.simulation.convertGeo (lon, lat, fromGeo=True)))
    
        for line in antenna_loc_file: 
            splitted_line = line.split(',')
            if (splitted_line[0]=='radio'):
                continue

            # lon, lat = float (splitted_line[6]), float (splitted_line[7]) 
            pos = traci.simulation.convertGeo (float (splitted_line[6]), float (splitted_line[7]), True)
            printf (APs_loc_file, '{},{},{},{}\n' .format (AP_id, pos[0], pos[1], self.is_in_lux_global_area (pos)))
            AP_id += 1
        traci.close()
            
if __name__ == '__main__':
    
    my_Traci_runner = Traci_runner (sumo_cfg_file='myMoST.sumocfg')
    my_Traci_runner.parse_antenna_locs_file ('short.txt')

    # my_Traci_runner = Traci_runner (sumo_cfg_file='myMoST.sumocfg')
    #
    # my_Traci_runner.convert_geo_to_loc ()

    # my_Traci_runner.simulate_to_cnt_vehs_only (
    #                 warmup_period           = 3600*7.5, #3600*7.5,
    #                 sim_length              = 3600*1,
    #                 len_of_time_slot_in_sec = 1,
    #                 verbose                 = [])

    # my_Traci_runner.simulte (warmup_period          = 3600*7.5,
    #                         sim_length              = 3600*1,
    #                         len_of_time_slot_in_sec = 1,
    #                         num_of_output_files     = 1, 
    #                         verbose                 = [])
