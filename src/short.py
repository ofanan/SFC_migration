import optparse
import random
from sumolib import checkBinary  
import traci  
import sys

from printf import printf 

if __name__ == "__main__":

    with open('vehicles.pos', 'w') as pos_output_file:
        pos_output_file.write('')                
    pos_output_file  = open ('../res/vehicles.pos',  "w")

    traci.start([checkBinary('sumo'), '-c', 'my.sumocfg', '-W', '-V', 'false', '--no-step-log', 'true']) 
    while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
        if (traci.simulation.getTime() % 60 ==0):
            printf (pos_output_file, 't={}: act={} moving={}\n' 
                     .format (traci.simulation.getTime(), traci.vehicle.getIDCount(), traci.vehicle.getIDCount() - len ([v for v in traci.vehicle.getIDList() if traci.vehicle.getSpeed(v)==0])))
        sys.stdout.flush()
        traci.simulationStep()
    traci.close()
