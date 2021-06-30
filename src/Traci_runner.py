import optparse
import random
from sumolib import checkBinary  
import traci  
import sys

from printf import printf 

if __name__ == "__main__":

    traci.start([checkBinary('sumo'), '-c', 'my.sumocfg', '-W', '-V', 'false', '--no-step-log', 'true']) 
    warmup_period = 3600*7.5
    traci.simulationStep (warmup_period) # simulate without output until our required time (time starts at 00:00). 
    veh_key2id            = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
    ids2recycle           = [] # will hold a list of ids that are not used anymore, and therefore can be recycled
    left_in_this_cycle    = []
    num_of_vehicles       = 0
    num_of_output_files   = 9
    
    for i in range (num_of_output_files):
        
        output_file_name = '../res/vehicles_' + str(i) + '.loc' 
        with open(output_file_name, 'w') as loc_output_file:
            loc_output_file.write('')                
        loc_output_file  = open (output_file_name,  "w")
        printf (loc_output_file, '// locations of vehicles\n// format:\n// type usr_id x y, where:\n// type is either \'o\' (old veh), or \'n\' (new veh in the sim). (x,y) are the coodinates of the vehicle with this usr_id.\n' )
        # 3600*1.5
        while (traci.simulation.getTime() < warmup_period + 3600*1.5*(i+1) / num_of_output_files and traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
            
            # # print statistics of the number of cars along the simulation
            # if (traci.simulation.getTime() % 60 ==0):
            #     printf (loc_output_file, 't={}: act={} moving={}\n' 
            #              .format (traci.simulation.getTime(), traci.vehicle.getIDCount(), traci.vehicle.getIDCount() - len ([v for v in traci.vehicle.getIDList() if traci.vehicle.getSpeed(v)==0])))
            printf (loc_output_file, '\nt = {:.0f}\n' .format (traci.simulation.getTime()))
            cur_list_of_vehicles = [veh for veh in traci.vehicle.getIDList() if traci.vehicle.getSpeed(veh)>0]
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
                filtered_list = list (filter (lambda veh : veh['key'] == veh_key, veh_key2id)) # look for this veh in the list of already-known vehs
                if (len(filtered_list) == 0): # first time this veh_key appears in the simulation
                    type = 'n' # will indicate that this is a new vehicle 
                    if (len (ids2recycle) > 0): # can recycle an ID of a veh that left
                        veh_id = ids2recycle.pop(0) # recycle an ID of a veh that left, and remove it from the list of veh IDs to recycle
                        veh_key2id = list (filter (lambda veh : veh['id'] != veh_id, veh_key2id)) # remove the old {veh_key, veh_id} tuple from the veh_key2id map 
                    else:
                        veh_id = num_of_vehicles # pick a new ID
                        num_of_vehicles += 1
                    veh_key2id.append({'key' : veh_key, 'id' : veh_id}) #, 'new' : True}) 
                else: # already seen this veh_key in the sim' --> extract its id from the hash 
                    type = 'o' # will indicate that this is a old vehicle 
                    veh_id = filtered_list[0]['id'] 
                position = traci.vehicle.getPosition(veh_key)
                printf (loc_output_file, "({},{},{:.0f},{:.0f})" .format (type, veh_id, position[0], position[1]))
    
            sys.stdout.flush()
            traci.simulationStep()
    traci.close()
