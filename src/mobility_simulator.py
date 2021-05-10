import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf

# Constant for indices of X and Y in vectors' coordinates
Y = 1
X = 0 

# Constant, denoting the types of events (either migration, or arrival to the destination) 
EVENT_T_MIG        = 0 # A usr arrived to area covered by another AP - need to migrate
EVENT_T_START_MOV  = 1 # A usr should start moving
EVENT_T_ARRIVE     = 2 # A usr arrived to its destination. Need to write the new location, and schedule a new movement
EVENT_T_PRINT_LOCS = 3 # A periodical event, for printing the locations of all usrs.

class my_mobility_simulator (object):
    """
    Event-driven mobility simulator.
    Initial locations of usrs are picked u.a.r. from a rectangle.
    Next, each usr moves each time in the direction of either +x, -x, +y, or -y.
    The area is covered by sub-rectangles, each covered exclusively by a single Access Point (AP).
    The simulator verifies that usrs always stay within the large rectangle, 
    and that the next location is always within the territory of an AP different from its current one.
    Each time a usr switches AP, the assignment of all usrs to APs is printed to an output file.
    """

    #ap        = lambda self, usr                  : 2 * int (usr['loc'][self.Y] / (0.5*self.edge)) + int (self.usr[u]['loc'][self.X] / (0.5*self.edge))
    min_mov   = lambda self, u, dim              : abs(0.5*self.edge - self.usr[u]['loc'][dim]) # caclulate the min move required to verify that the usr switches to another AP
    arr_t     = lambda self, cur_loc, dst, speed : self.cur_time + math.sqrt (pow(cur_loc[X] - dst[X], 2) + pow(cur_loc[Y] - dst[Y], 2)) / speed
    nxt_loc   = lambda self, u, dst, dim         : [dst, self.usr[u]['loc'][self.Y]] if (dim == self.X) else [self.usr[u]['loc'][self.X], dst]  
    mig_t     = lambda self, u, min_mov          : self.cur_time + min_mov / self.usr[u]['max speed']
    loc2ap    = lambda self, loc : int (math.floor ((loc[Y] / self.cell_edge) ) * self.num_of_APs_in_row + math.floor ((loc[X] / self.cell_edge) )) 
      
    def start_mov_usr (self, usr):
        """
        Start a move of a usr.
        The usr moves in parallel to either the X or the Y axis (chosen u.a.r.). 
        The function verifies that the destination is within the coverage area of an Access Point (AP) that differs from the current access point.  
        This includes generating and inserting 2 event into the events queue:
        1. Event for the migration - scheduled for the time when the usr enters the coverage area of another AP.
        2. Event for the arrival of the usr to its destination.
        The exact destination point is chosen randomly.
        """

        usr['final loc'] = self.edge * np.random.rand (2)  # destination of movement
        usr['final ap' ] = self.loc2ap (usr['final loc'])
        self.eventQ.append ({'event type'   : EVENT_T_ARRIVE,
                             'time'         : self.arr_t (usr['cur loc'], usr['final loc'], usr['max speed']),
                             'usr'          : usr
                             })
        if (usr['final ap' ] == usr['cur ap']): # the move is within the coverage of the current AP - no need to schedule additional events 
            return
        
        theta = np.arctan (abs( (usr['final loc'][Y] - usr['cur loc'][Y]) / (usr['final loc'][X] - usr['cur loc'][X]) )) # angle of progression
        usr['speed'] = [math.cos(theta)*usr['max speed'], math.sin(theta)*usr['max speed']]  #Projection of the speed on the X, Y, direction
        
        if (usr['final loc'][X] > )
        
        self.eventQ.append ({'event type'   : EVENT_T_MIG,
                             'time'         : self.mig_t (usr, min_mov), 
                             'usr'         : usr
                             })

    def nxt_ap (self):
        """
        calculate the next AP
        """
            
    def print_eventQ (self):
        """
        Print the event queue. Used for debugging
        """
        for event in self.eventQ:
            print (event)

    def print_locs (self):
        printf (self.loc_output_file, 'time = {:.4f} : \n' .format (self.cur_time)) 
        for u in range(self.NUM_OF_usrS):
            printf (self.loc_output_file, 'usr {} {:.1f} {:.1f}\n' .format (u, self.usr[u]['loc'][self.X], self.usr[u]['loc'][self.Y]))
        printf (self.loc_output_file, '\n')    

    def print_APs (self):
        
        printf (self.ap_output_file, 'time = {:.4f} : ' .format (self.cur_time)) 
        for u in range(self.NUM_OF_usrS):
            printf (self.ap_output_file, '({},{})' .format (u, self.usr[u]['cur ap']))
        printf (self.ap_output_file, '\n')
            
    def simulate (self):
        """
        Run a simulation.
        The simulation begins by letting all the usrs start move, thus also scheduling future events.
        From there and on, an event-driven simulation is run.
        - While (stop condition isn't met) 
            - Advance the time to the closest future event.
            - Dequeue that event from the event queue, and handle it.
                - If needed, schedule future events
        """

        printf (self.ap_output_file,  '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of usr X at time t\n\n')
        printf (self.loc_output_file, '// usrs mobility - output by mobility_simulator.py\n')
        printf (self.loc_output_file, '// File format:\n//time = t:\n')
        printf (self.loc_output_file, '// usr U (X, Y)\n//where (X,Y) is the location of usr U at time t\n')
        printf (self.loc_output_file, '// the simulation is done in a rectangle of size MAX_X * MAX_Y, where\n')       
        printf (self.loc_output_file, 'MAX_X {:.0f} MAX_Y {:.0f}\n\n' .format(self.edge, self.edge))

        # Schedule initial event for typing usrs' locations 
        self.eventQ.append ({'event type' : EVENT_T_PRINT_LOCS,
                                 'time'   : 0})                 
           
        # Start move the usrs   
        for u in range(self.NUM_OF_usrS):
            self.start_mov_usr (self.usr[u])
    
        while (1):
        
            if (self.eventQ): # event queue isn't empty
                
                # Pop the closest-future event, advance time and handle this event
                self.eventQ = sorted (self.eventQ, key = lambda event : event['time'])
                event = self.eventQ.pop (0)
                self.cur_time = event['time']
        
                if (event['event type'] == EVENT_T_MIG):
                    usr        = event['usr']  
                    usr ['cur ap']  = event['nxt AP']
                    usr ['cur loc'] = event['nxt loc']
                    self.print_APs ()
                    self.mov_usr (usr)
                elif (event['event type'] == EVENT_T_ARRIVE):
                    usr = event['usr']  
                    usr ['cur loc'] = event['nxt loc']
                    self.eventQ.append ({'event type' : EVENT_T_START_MOV,
                                         'time'       : self.cur_time + np.random.rand () * (self.usr[event['usr']]['max stnd time'] - self.usr[event['usr']]['min stnd time']),
                                         'usr'        : usr
                                         })
                elif (event['event type'] == EVENT_T_START_MOV):
                    self.start_mov_usr(event['usr'])
                elif (event['event type'] == EVENT_T_PRINT_LOCS):
                    self.print_locs ()
                    self.eventQ.append ({'event type' : EVENT_T_PRINT_LOCS,
                                         'time'         : self.cur_time + self.T_BETWEEN_PRINT_LOCS
                                        })                    
                else:
                    print ('Error: unsupported event type')
                    exit ()
        
            if (self.cur_time > self.MAX_TIME):
                exit ()           
    
    def __init__ (self):
        
        self.edge = 100 # edge of the rectangle in which usr move [m]
        self.num_of_APs_in_row = 2   
        self.cell_edge = self.edge / self.num_of_APs_in_row
        self.NUM_OF_usrS  = 5
        self.ap_output_file  = open ("../res/my_mob_sim.ap",  "w")
        self.loc_output_file = open ("../res/my_mob_sim.loc", "w")

        # Constant, denoting the moving direction (either X or Y)
        self.X = 0
        self.Y = 1
        self.usr = [{} for u in range (self.NUM_OF_usrS)]
        for u in range (self.NUM_OF_usrS):
            self.usr[u] = {'max speed'      : (30 + 5*u)/ 3.6, 
                            'cur loc'       : np.random.rand (2) * self.edge,
                            'min stnd time' : 0.5,
                            'max stnd time' : 1.5
                            }  # speed [m/sec]
            self.usr[u]['cur ap'] = self.loc2ap (self.usr[u]['cur loc'])
        
        self.eventQ   = []
        self.cur_time = 0
        self.MAX_TIME = 20

        self.T_BETWEEN_PRINT_LOCS = 1 # time between sequencing prints of all usrs' locations        

if __name__ == "__main__":
    sim = my_mobility_simulator ()
    sim.simulate ()
    exit
        
