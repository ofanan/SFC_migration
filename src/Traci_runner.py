#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2021 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random

from printf import printf 

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa



def run():
    """execute the TraCI control loop"""
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlight.setPhase("0", 2)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if traci.trafficlight.getPhase("0") == 2:
            # we are not already switching
            if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
                # there is a vehicle from the north, switch
                traci.trafficlight.setPhase("0", 3)
            else:
                # otherwise try to keep green for EW
                traci.trafficlight.setPhase("0", 2)
        step += 1
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    sumoBinary = checkBinary('sumo')
    path_to_scenario = '../../LuSTScenario/scenario/'

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", path_to_scenario + "due.actuated.sumocfg"]) #"--tripinfo-output", "tripinfo.xml"])
    with open('vehicles.pos', 'w') as FD:
        FD.write('')                
    FD  = open ('vehicles.pos',  "w")
    
    num_of_simulated_secs = 3000
    num_of_vehicles = 0
    step = 0
    veh_key2id = []
    while (step < 3000 and traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
        printf (FD, 't = {:.2f}\n' .format (traci.simulation.getTime()))
        for veh_key in traci.vehicle.getIDList(): 
            filtered_list = list (filter (lambda veh : veh['key'] == veh_key, veh_key2id))
            if (len(filtered_list) == 0): # first time this veh_key appears in the simulation
                veh_id = num_of_vehicles
                veh_key2id.append({'key' : veh_key, 'id' : veh_id})
                num_of_vehicles += 1
            elif (len(filtered_list) == 1): # already seen this veh_key in the sim' --> extract its id from the hash 
                veh_id = filtered_list[0]['id']
            elif (len(filtered_list) == 2):
                printf (FD, 'In-naal raback\n')
                exit ()
            position = traci.vehicle.getPosition(veh_key)
            printf (FD, "user {} \tID={}\t ({:.1f},{:.1f})\n" .format (veh_key, veh_id, position[0], position[1]))
            # exit ()
        traci.simulationStep()
        step += 1
    run()
