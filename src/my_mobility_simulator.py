import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf

class my_mobility_simulator (object):

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
        self.eventQ.append ({'event type'   : self.arrive,
                             'time'         : self.arr_t (u, dst, dim),
                             'nxt loc'      : self.nxt_loc (u, dst, dim),
                             'user'         : u
                             })
        
        nxt_ap = self.nxt_ap (u, direction, dim)
        print ('u = {}, cur loc = {}, cur ap = {}, direction = {}, dim = {}, nxt ap = {}' .format (u, self.user[u]['loc'], self.user[u]['ap'], direction, dim, nxt_ap))
        if (nxt_ap > 3):
            exit ()
        self.eventQ.append ({'event type'   : self.mig,
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


    def print_APs (self):
        
        printf (self.output_file, 'time = {:.4f} : ' .format (self.cur_time)) 
        for u in range(self.NUM_OF_USERS):
            printf (self.output_file, '({},{})' .format (u, self.user[u]['ap']))
        printf (self.output_file, '\n')
            
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

        printf (self.output_file, '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of user X at time t\n\n')

        for u in range(self.NUM_OF_USERS):
            self.move_user (u)
    
        while (1):
        
            if (self.eventQ): # event queue isn't empty
                
                # Pop the closest-future event, advance time and handle this event
                self.eventQ = sorted (self.eventQ, key = lambda event : event['time'])
                event = self.eventQ.pop (0)
                self.cur_time = event['time']
        
                if (event['event type'] == self.mig):
                    self.user[event['user']]['ap'] = event['nxt AP']
                    self.print_APs ()
                else:
                    self.user[event['user']]['loc'] = event['nxt loc']
                    self.move_user(u)
        
            if (self.cur_time > self.max_time):
                exit ()           
    
    def __init__ (self):
        
        self.edge = 100 # edge of the rectangle in which user move [m]
        self.NUM_OF_USERS  = 3
        self.maxt_time = 1
        self.output_file = open ("../res/my_mob_sim.ap",  "w")


        # Constant, denoting the moving direction (either X or Y)
        self.X = 0
        self.Y = 1
        self.user = [{} for u in range (self.NUM_OF_USERS)]
        for u in range (self.NUM_OF_USERS):
            self.user[u] = {'speed' : (30 + 5*u)/ 3.6, 'loc'   : np.random.rand (2) * self.edge}  # speed [m/sec]
            self.user[u]['ap'] = self.ap (u)
        
        self.eventQ = []
        self.cur_time = 0
        self.max_time = 10
        
        # Constant, denoting the types of events (either migration, or arrival to the destination) 
        self.mig    = True
        self.arrive = False
        

if __name__ == "__main__":
    sim = my_mobility_simulator ()
    sim.simulate ()
    exit
        
