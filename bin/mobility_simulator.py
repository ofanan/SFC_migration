import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf

# Constant, denoting the types of events (either migration, or arrival to the destination) 
EVENT_T_MIG        = 0 # A user arrived to area covered by another AP - need to migrate
EVENT_T_ARRIVE     = 1 # A user arrived to its destination. Need to write the new location, and schedule a new movement
EVENT_T_PRINT_LOCS = 2 # A periodical event, for printing the locations of all users.

class my_mobility_simulator (object):
    """
    Event-driven mobility simulator.
    Initial locations of users are picked u.a.r. from a rectangle.
    Next, each user moves each time in the direction of either +x, -x, +y, or -y.
    The area is covered by sub-rectangles, each covered exclusively by a single Access Point (AP).
    The simulator verifies that users always stay within the large rectangle, 
    and that the next location is always within the territory of an AP different from its current one.
    Each time a user switches AP, the assignment of all users to APs is printed to an output file.
    """

    ap        = lambda self, u                  : 2 * int (self.user[u]['loc'][self.Y] / (0.5*self.edge)) + int (self.user[u]['loc'][self.X] / (0.5*self.edge))
    min_mov   = lambda self, u, dim             : abs(0.5*self.edge - self.user[u]['loc'][dim]) # caclulate the min move required to verify that the user switches to another AP
    direction = lambda self, u, dim             : 1 if (self.user[u]['loc'][dim] < 0.5 * self. edge) else -1
    dst       = lambda self, direction          : 0.5 * self.edge * (1 + np.random.rand () * direction) #if (self.user[u]['loc'][dim] < 0.5 * self.edge) else 0.5 * self.edge * (1 - np.random.rand ())  
    arr_t     = lambda self, u, dst, dim        : self.cur_time + abs (dst - self.user[u]['loc'][dim]) / self.user[u]['speed']
    nxt_loc   = lambda self, u, dst, dim        : [dst, self.user[u]['loc'][self.Y]] if (dim == self.X) else [self.user[u]['loc'][self.X], dst]  
    mig_t     = lambda self, u, min_mov         : self.cur_time + min_mov / self.user[u]['speed']
    nxt_ap    = lambda self, u, direction, dim  : self.user[u]['ap'] + direction if dim == self.X else self.user[u]['ap'] + 2 * direction 
#     nxt_ap  = lambda self, u, dim          : 2 * int(self.user[u]['ap'] / 2) + 1 - int (self.user[u]['ap'] % 2) if (dim == x) else \
      
            
    def move_user (self, u):
        """
        Start a move of a user.
        The user moves in parallel to either the X or the Y axis (chosen u.a.r.). 
        The function verifies that the destination is within the coverage area of an Access Point (AP) that differs from the current access point.  
        This includes generating and inserting 2 event into the events queue:
        1. Event for the migration - scheduled for the time when the user enters the coverage area of another AP.
        2. Event for the arrival of the user to its destination.
        The exact destination point is chosen randomly.
        """
        
        dim       = random.getrandbits(1)   # dimension of nxt movement (either self.X or self.Y)
        min_mov   = self.min_mov   (u, dim) # minimal move required to verify that the user switches to another AP
        direction = self.direction (u, dim) # either +1 (increasing) or -1 (decreasing)
        dst       = self.dst       (direction)  # destination of movement
        self.eventQ.append ({'event type'   : EVENT_T_ARRIVE,
                             'time'         : self.arr_t (u, dst, dim),
                             'nxt loc'      : self.nxt_loc (u, dst, dim),
                             'user'         : u
                             })
        
        nxt_ap = self.nxt_ap (u, direction, dim)
        self.eventQ.append ({'event type'   : EVENT_T_MIG,
                             'time'         : self.mig_t (u, min_mov), 
                             'nxt AP'       : self.nxt_ap (u, direction, dim),
                             'user'         : u
                             })

    def print_eventQ (self):
        """
        Print the event queue. Used for debugging
        """
        for event in self.eventQ:
            print (event)

    def print_locs (self):
        printf (self.loc_output_file, 'time = {:.4f} : \n' .format (self.cur_time)) 
        for u in range(self.NUM_OF_USERS):
            printf (self.loc_output_file, 'user {} ({:.1f},{:.1f})\n' .format (u, self.user[u]['loc'][self.X], self.user[u]['loc'][self.Y]))
        printf (self.loc_output_file, '\n')    

    def print_APs (self):
        
        printf (self.ap_output_file, 'time = {:.4f} : ' .format (self.cur_time)) 
        for u in range(self.NUM_OF_USERS):
            printf (self.ap_output_file, '({},{})' .format (u, self.user[u]['ap']))
        printf (self.ap_output_file, '\n')
            
    def simulate (self):
        """
        Run a simulation.
        The simulation begins by letting all the users start move, thus also scheduling future events.
        From there and on, an event-driven simulation is run.
        - While (stop condition isn't met) 
            - Advance the time to the closest future event.
            - Dequeue that event from the event queue, and handle it.
                - If needed, schedule future events
        """

        printf (self.ap_output_file,  '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of user X at time t\n\n')
        printf (self.loc_output_file, '// File format:\n//time = t:\n')
        printf (self.loc_output_file, '// user U (X, Y)\n//where (X,Y) is the location of user U at time t\n\n')

        # Schedule initial event for typing users' locations 
        self.eventQ.append ({'event type' : EVENT_T_PRINT_LOCS,
                                 'time'   : 0})                 
           
        # Start move the users   
        for u in range(self.NUM_OF_USERS):
            self.move_user (u)
    
        while (1):
        
            if (self.eventQ): # event queue isn't empty
                
                # Pop the closest-future event, advance time and handle this event
                self.eventQ = sorted (self.eventQ, key = lambda event : event['time'])
                event = self.eventQ.pop (0)
                self.cur_time = event['time']
        
                if (event['event type'] == EVENT_T_MIG):
                    cur_usr = event['user']  
                    self.user[cur_usr]['ap'] = event['nxt AP']
                    self.print_APs ()
                elif (event['event type'] == EVENT_T_ARRIVE):
                    cur_usr = event['user']  
                    self.user[cur_usr]['loc'] = event['nxt loc']
                    self.move_user(cur_usr)
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
        
        self.edge = 100 # edge of the rectangle in which user move [m]
        self.NUM_OF_USERS  = 5
        self.ap_output_file  = open ("my_mob_sim.ap",  "w")
        self.loc_output_file = open ("my_mob_sim.loc", "w")

        # Constant, denoting the moving direction (either X or Y)
        self.X = 0
        self.Y = 1
        self.user = [{} for u in range (self.NUM_OF_USERS)]
        for u in range (self.NUM_OF_USERS):
            self.user[u] = {'speed' : (30 + 5*u)/ 3.6, 'loc'   : np.random.rand (2) * self.edge}  # speed [m/sec]
            self.user[u]['ap'] = self.ap (u)
        
        self.eventQ = []
        self.cur_time = 0
        self.MAX_TIME = 20
        self.T_BETWEEN_PRINT_LOCS = 1 # time between sequencing prints of all users' locations        

if __name__ == "__main__":
    sim = my_mobility_simulator ()
    sim.simulate ()
    exit
        
