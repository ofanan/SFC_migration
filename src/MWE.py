import optparse
import random
from sumolib import checkBinary  
import traci  
import sys

if __name__ == "__main__":

    traci.start([checkBinary('sumo'), '-c', 'my.sumocfg', '-W', '-v']) 
    while (traci.simulation.getMinExpectedNumber() > 0): # There're still moving vehicles
        if (traci.simulation.getTime() % 60 ==0):
            print ('t=', traci.simulation.getTime(),
                   'my_TOT ', traci.vehicle.getIDCount(), 
                   'moving vehicles ', traci.vehicle.getIDCount() - len ([v for v in traci.vehicle.getIDList() if traci.vehicle.getSpeed(v)==0]))
        sys.stdout.flush()
        traci.simulationStep()
        sys.stdout.flush()
    traci.close()
