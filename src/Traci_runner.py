import optparse
import random
from sumolib import checkBinary  
import traci  
import sys
import numpy as np

from printf import printf 
from secs2hour import secs2hour
import loc2ap_c

VERBOSE_LOC_ONLY      = 1
VERBOSE_DEBUG_ONLY    = 2
VERBOSE_LOC_AND_DEBUG = 3
VERBOSE_LOC_AND_SPEED = 4

class Traci_runner (object):

    
    def __init__ (self, warmup_period=0, sim_length=10, len_of_time_slot_in_sec=1, num_of_output_files=1, verbose = VERBOSE_LOC_ONLY):
        traci.start([checkBinary('sumo'), '-c', 'my.sumocfg', '-W', '-V', 'false', '--no-step-log', 'true'])
        veh_key2id              = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
        ids2recycle             = [] # will hold a list of ids that are not used anymore, and therefore can be recycled
        left_in_this_cycle      = []
        self.verbose            = verbose
        
        if (self.verbose in [VERBOSE_DEBUG_ONLY, VERBOSE_LOC_AND_DEBUG]):
            debug_file_name = '../res/Traci_runner_debug.res'
            with open(debug_file_name, 'w') as debug_file:
                debug_file.write('')
                debug_file  = open (debug_file_name,  "w")
            
        if (warmup_period > 0):
            traci.simulationStep (int(warmup_period)) # simulate without output until our required time (time starts at 00:00). 
        for i in range (num_of_output_files):
            
            output_file_name = '../res/vehicles_' + secs2hour(traci.simulation.getTime()) + '.loc' 
            with open(output_file_name, 'w') as loc_output_file:
                loc_output_file.write('')                
            loc_output_file  = open (output_file_name,  "w")
            printf (loc_output_file, '// locations of vehicles. Format:\n')
            printf (loc_output_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (loc_output_file, '// format for vehicles that are new (just joined the sim), or moved:\n')
            if (self.verbose == VERBOSE_LOC_AND_SPEED): 
                printf (loc_output_file, '// (type,usr_id,x,y,s)   where:\n')
                printf (loc_output_file, '// type is either o (old veh), or n (new veh in the sim). (x,y) are the coordinates of the vehicle with this usr_id. s is the speed of this user.\n')
            else:
                printf (loc_output_file, '// (type,usr_id,x,y)   where:\n')
                printf (loc_output_file, '// type is either o (old veh), or n (new veh in the sim). (x,y) are the coordinates of the vehicle with this usr_id.\n')
                
            while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
                
                cur_sim_time = traci.simulation.getTime() 
                if (cur_sim_time >= warmup_period + sim_length*(i+1) / num_of_output_files):
                    print ('Successfully finished writing to file {}' .format (output_file_name))
                    break
                
                printf (loc_output_file, '\nt = {:.0f}\n' .format (cur_sim_time))
                cur_list_of_vehicles = [veh_key for veh_key in traci.vehicle.getIDList() if self.get_relative_position(veh_key)] # list of vehs currently found within the simulated area.
                left_in_this_cycle   = list (filter (lambda veh : (veh['key'] not in (cur_list_of_vehicles) and 
                                                                   veh['id'] not in (ids2recycle)), veh_key2id)) 
                printf (loc_output_file, 'usrs_that_left: ')
                for veh in left_in_this_cycle:
                    printf (loc_output_file, '{:.0f} ' .format(veh['id']))
                printf (loc_output_file, '\n')
                if (len (left_in_this_cycle) > 0):
                    ids2recycle = sorted (list (set (ids2recycle) | set([item['id'] for item in left_in_this_cycle]))) # add to ids2recycle the IDs of all cars that left
                printf (loc_output_file, 'new_or_moved: ')
                for veh_key in cur_list_of_vehicles:
                    if (not(self.get_relative_position(veh_key))): # Skip vehs that are outside the simulated area
                        continue
                    filtered_list = list (filter (lambda veh : veh['key'] == veh_key, veh_key2id)) # look for this veh in the list of already-known vehs
                    if (len(filtered_list) == 0): # first time this veh_key appears in the simulation
                        type = 'n' # will indicate that this is a new vehicle 
                        if (len (ids2recycle) > 0): # can recycle an ID of a veh that left
                            # type = 'r' # will indicate that this is a recycled vehicle's id 
                            veh_id = ids2recycle.pop(0) # recycle an ID of a veh that left, and remove it from the list of veh IDs to recycle
                            veh_key2id = list (filter (lambda veh : veh['id'] != veh_id, veh_key2id)) # remove the old {veh_key, veh_id} tuple from the veh_key2id map 
                        else:
                            # type = 'n' # will indicate that this is a new vehicle 
                            veh_id = len (veh_key2id) # pick a new ID
                        veh_key2id.append({'key' : veh_key, 'id' : veh_id}) # insert the new veh into the db 
                    else: # already seen this veh_key in the sim' --> extract its id from the hash
                        if (traci.vehicle.getSpeed(veh_key) == 0): #$$$$ # the usr didn't move in the last slot - no need to report of its movement  
                            continue
                        type = 'o' # will indicate that this is a old vehicle 
                        veh_id = filtered_list[0]['id'] 
                    if (self.verbose == VERBOSE_LOC_AND_SPEED): 
                        printf (loc_output_file, "({},{},{:.0f},{:.0f},{:.0f})" .format (type, veh_id, self.position[0], self.position[1], traci.vehicle.getSpeed(veh_key)))
                    else:
                        printf (loc_output_file, "({},{},{:.0f},{:.0f})" .format (type, veh_id, self.position[0], self.position[1]))
        
                sys.stdout.flush()
                traci.simulationStep (cur_sim_time + len_of_time_slot_in_sec)
        traci.close()

    def get_relative_position (self, veh_key):
       """
       Writes the relative location of a given vehicle (namely, its position relatively to the smaller left corner of ths simulated area),
        and checks whether this vehicle is within the simulated area.
       Input: key of a vehicle.
       Output: True iff this vehicle is within the simulated area.
       In addition, the function writes to self.position the relative [x,y] position of this vehicle.
       """ 
       self.position = np.array(traci.vehicle.getPosition(veh_key)) - loc2ap_c.LOWER_LEFT_CORNER
       if (self.position[0] < 0 or self.position[0] > loc2ap_c.MAX_X or self.position[1] < 0 or self.position[1] > loc2ap_c.MAX_Y):
           return False
       return True 

if __name__ == '__main__':
    my_Traci_runner = Traci_runner (warmup_period           = 3600*7.5,
                                    sim_length              = 3600*1,
                                    len_of_time_slot_in_sec = 1,
                                    num_of_output_files     = 6, 
                                    verbose                 = VERBOSE_LOC_AND_SPEED)
