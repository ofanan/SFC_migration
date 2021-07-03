import optparse
import random
from sumolib import checkBinary  
import traci  
import sys

from printf import printf 

VERBOSE_LOC_FILE      = 1
VERBOSE_DEBUG_ONLY    = 2
VERBOSE_LOC_AND_DEBUG = 3

class Traci_runner (object):

    
    def __init__ (self, warmup_period=0, sim_length=10, len_of_time_slot_in_sec=1, num_of_output_files=1, verbose = VERBOSE_LOC_FILE):
        traci.start([checkBinary('sumo'), '-c', 'my.sumocfg', '-W', '-V', 'false', '--no-step-log', 'true']) 
        warmup_period           = warmup_period 
        sim_length              = sim_length 
        len_of_time_slot_in_sec = len_of_time_slot_in_sec  
        veh_key2id              = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
        ids2recycle             = [] # will hold a list of ids that are not used anymore, and therefore can be recycled
        left_in_this_cycle      = []
        self.verbose            = verbose
        
        if (self.verbose == VERBOSE_LOC_AND_DEBUG):
            debug_file_name = '../res/Traci_runner_debug.res'
            with open(debug_file_name, 'w') as debug_file:
                debug_file.write('')
                debug_file  = open (debug_file_name,  "w")
            
        if (warmup_period > 0):
            traci.simulationStep (warmup_period) # simulate without output until our required time (time starts at 00:00). 
        for i in range (num_of_output_files):
            
            output_file_name = '../res/vehicles_' + str(i) + '.loc' 
            with open(output_file_name, 'w') as loc_output_file:
                loc_output_file.write('')                
            loc_output_file  = open (output_file_name,  "w")
            printf (loc_output_file, '// locations of vehicles. Format:\n')
            printf (loc_output_file, '// "usrs_that_left" is a list of IDs that left at this cycle, separated by spaces.\n')
            printf (loc_output_file, '// format for vehicles that are new (just joined the sim), or moved:\n')
            printf (loc_output_file, '// (type,usr_id,x,y)   where:\n')
            printf (loc_output_file, '// type is either o (old veh), or n (new veh in the sim). (x,y) are the coodinates of the vehicle with this usr_id.\n')
                       
            while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
                
                cur_sim_time = traci.simulation.getTime() 
                if (cur_sim_time > warmup_period + sim_length*(i+1) / num_of_output_files):
                    print ('Successfully finished writing to file {}' .format (output_file_name))
                    break
                
                if (self.verbose == VERBOSE_LOC_AND_DEBUG):
                    printf (debug_file, 't = {:.0f}\nTraci: vehs={}\n' .format (cur_sim_time, traci.vehicle.getIDCount() ))

                # print statistics of the number of cars along the simulation
                # if (cur_sim_time % 60 ==0):
                #     printf (loc_output_file, 't={}: act={} moving={}\n' 
                #              .format (cur_sim_time, traci.vehicle.getIDCount(), traci.vehicle.getIDCount() - len ([v for v in traci.vehicle.getIDList() if traci.vehicle.getSpeed(v)==0])))
                printf (loc_output_file, '\nt = {:.0f}\n' .format (cur_sim_time))
                cur_list_of_vehicles = [veh_key for veh_key in traci.vehicle.getIDList()] # if traci.vehicle.getSpeed(veh_key)>0]
                left_in_this_cycle   = list (filter (lambda veh : (veh['key'] not in (cur_list_of_vehicles) and 
                                                                   veh['id'] not in (ids2recycle)), veh_key2id)) 
                if (self.verbose == VERBOSE_LOC_AND_DEBUG):
                    printf (debug_file, 'My   : vehs={} left={} ' .format (len(cur_list_of_vehicles), len(left_in_this_cycle)))
                printf (loc_output_file, 'usrs_that_left: ')
                for veh in left_in_this_cycle:
                    printf (loc_output_file, '{:.0f} ' .format(veh['id']))
                printf (loc_output_file, '\n')
                if (len (left_in_this_cycle) > 0):
                    ids2recycle = sorted (list (set (ids2recycle) | set([item['id'] for item in left_in_this_cycle]))) # add to ids2recycle the IDs of all cars that left
                printf (loc_output_file, 'new_or_moved: ')
                if (self.verbose == VERBOSE_LOC_AND_DEBUG):
                    new_veh_cnt = 0
                for veh_key in cur_list_of_vehicles: 
                    filtered_list = list (filter (lambda veh : veh['key'] == veh_key, veh_key2id)) # look for this veh in the list of already-known vehs
                    if (len(filtered_list) == 0): # first time this veh_key appears in the simulation
                        type = 'n' # will indicate that this is a new vehicle 
                        if (self.verbose == VERBOSE_LOC_AND_DEBUG):
                            new_veh_cnt += 1
                        if (len (ids2recycle) > 0): # can recycle an ID of a veh that left
                            veh_id = ids2recycle.pop(0) # recycle an ID of a veh that left, and remove it from the list of veh IDs to recycle
                            veh_key2id = list (filter (lambda veh : veh['id'] != veh_id, veh_key2id)) # remove the old {veh_key, veh_id} tuple from the veh_key2id map 
                        else:
                            veh_id = len (veh_key2id) # pick a new ID
                            veh_key2id.append({'key' : veh_key, 'id' : veh_id}) #, 'new' : True}) 
                    else: # already seen this veh_key in the sim' --> extract its id from the hash
                        if (traci.vehicle.getSpeed(veh_key) == 0): # the usr didn't move in the last slot - no need to report of its movement 
                            continue
                        type = 'o' # will indicate that this is a old vehicle 
                        veh_id = filtered_list[0]['id'] 
                    position = traci.vehicle.getPosition(veh_key)
                    printf (loc_output_file, "({},{},{:.0f},{:.0f})" .format (type, veh_id, position[0], position[1]))
        
                if (self.verbose == VERBOSE_LOC_AND_DEBUG):
                    printf (debug_file, 'new={}\n' .format (new_veh_cnt))
                sys.stdout.flush()
                traci.simulationStep (cur_sim_time + len_of_time_slot_in_sec)
        traci.close()

if __name__ == '__main__':
    my_Traci_runner = Traci_runner (warmup_period           = 0, #3600*7.5 
                                    sim_length              = 80, #3600*1.5
                                    len_of_time_slot_in_sec = 20,
                                    num_of_output_files     = 1, 
                                    verbose                 = VERBOSE_LOC_AND_DEBUG)
