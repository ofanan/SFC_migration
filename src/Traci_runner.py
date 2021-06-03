# This file simulates using Sumo.
# Output: 
# - location of each vehicle in a convenient format
# - can collect and print also some basic statistics - e.g., number of active cars at each moment. 
from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import os
import sys
import optparse
import random
from sumolib import checkBinary  # noqa
import traci  # noqa

from printf import printf 

# # An accessory inline function, to filter out some ids from a list
# filter_out = lambda id, list_of_ids: not (id in list_of_ids)

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def Traci_exit ():
    """
    Clean-exit both Traci, and the whole Python program.
    """
    traci.close()
    sys.stdout.flush()
    exit ()

if __name__ == "__main__":

    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    sumoBinary = checkBinary('sumo')
    path_to_scenario = '../../LuSTScenario/scenario/'

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs #--no-step-log #--verbose #"'--duration-log.statistics', 'false' 
    traci.start([sumoBinary, '-c', 'my.sumocfg', '-W', '-V', 'false', '--no-step-log', 'false']) #"--tripinfo-output", "tripinfo.xml"])
    with open('vehicles.pos', 'w') as pos_output_file:
        pos_output_file.write('')                
    pos_output_file  = open ('../res/vehicles.pos',  "w")
    
    num_of_simulated_secs = 36# 00*24
    num_of_vehicles       = 0
    step                  = 0
    veh_key2id            = [] # will hold pairs of (veh key, veh id). veh_key is given by Sumo; veh_id is my integer identifier of currently active car at each step.
    ids2recycle           = [] # will hold a list of ids that are not used anymore, and therefore can be recycled
    X, Y                  = [], [] # Will be used for ptyhon-plots            

    while (step < num_of_simulated_secs and traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
        cur_list_of_vehicles = traci.vehicle.getIDList()
        printf (pos_output_file, 't = {:.0f}, {} active vehicles\n' .format (traci.simulation.getTime(), len(cur_list_of_vehicles)))
        vehs_that_left = list (filter (lambda veh : veh['key'] not in (cur_list_of_vehicles), veh_key2id)) #list of cars that left the simulation
        if (len (vehs_that_left) > 0):
            ids2recycle = sorted (list (set (ids2recycle) | set([item['id'] for item in vehs_that_left]))) # add to ids2recycle the IDs of all cars that left
        X.append (step), Y.append (len(cur_list_of_vehicles))
        for veh_key in cur_list_of_vehicles: 
            filtered_list = list (filter (lambda veh : veh['key'] == veh_key, veh_key2id)) # look for this veh in the list of already-known vehs 
            if (len(filtered_list) == 0): # first time this veh_key appears in the simulation
                if (len (ids2recycle) > 0): # can recycle an ID of a veh that left
                    veh_id = ids2recycle.pop(0) # recycle an ID of a veh that left, and remove it from the list of veh IDs to recycle
                    veh_key2id = list (filter (lambda veh : veh['id'] != veh_id, veh_key2id)) # remove the old {veh_key, veh_id} tuple from the veh_key2id map 
                else:
                    veh_id = num_of_vehicles # pick a new ID
                    num_of_vehicles += 1
                veh_key2id.append({'key' : veh_key, 'id' : veh_id}) 
            elif (len(filtered_list) == 1): # already seen this veh_key in the sim' --> extract its id from the hash 
                veh_id = filtered_list[0]['id']
            elif (len(filtered_list) == 2):
                printf (pos_output_file, 'In-naal raback\n')
                Traci_exit ()
            position = traci.vehicle.getPosition(veh_key)
            printf (pos_output_file, "user {} \tID={}\t ({:.0f},{:.0f})\n" .format (veh_key, veh_id, position[0], position[1]))
        traci.simulationStep()
        step += 1
    # plt.scatter(X, Y)
    printf (pos_output_file, '\nX = {}\nY = {}\n' .format(X, Y))
    plt.plot (X,Y)
    plt.show()
    traci.close()
    sys.stdout.flush()
