import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf
# from numpy.f2py.crackfortran import verbose

# Constant for indices of X and Y in vectors' coordinates
Y = 1
X = 0 

# Constant, denoting the types of events (either migration, or arrival to the destination) 
EVENT_T_MIG        = 0 # A usr arrived to area covered by another AP - need to migrate
EVENT_T_START_MOV  = 1 # A usr should start moving
EVENT_T_ARRIVE     = 2 # A usr should arrive to its destination. Need to write the new location, and schedule a new movement
EVENT_T_PERIODICAL = 3 # A periodical event, e.g., for printing the current APs of all usrs.

# Levels of verbose (which output is generated)
VERBOSE_NO_OUTPUT             = 0
VERBOSE_ONLY_EVENTS           = 1 # Write to a file users' APs only upon a user migration
VERBOSE_ONLY_PERIODICAL       = 2 # Write to a file users' APs only periodically 
VERBOSE_EVENTS_AND_PERIODICAL = 3 # Write to a file users' APs upon either a user migration, or a period elapsed (e.g., every sec.).

EPSILON = 0.0001

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

    arr_t     = lambda self, cur_loc, dst, speed : self.cur_time + math.sqrt (pow(cur_loc[X] - dst[X], 2) + pow(cur_loc[Y] - dst[Y], 2)) / speed
    loc2ap    = lambda self, loc : int (math.floor ((loc[Y] / self.cell_edge) ) * self.num_of_APs_in_row + math.floor ((loc[X] / self.cell_edge) )) 
      
    def start_mov_usr (self, usr):
        """
        Start a move of a usr: 
        - Pick the destination (usr['final loc']) u.a.r from the area
        - Generate an event for the arrival to the destination, and insert it into the eventQ.
        - If the destination is within the coverage area of another AP, generate an additional event, for the migration to that AP.
        """

        usr['final loc'] = self.edge * np.random.rand (2)  # destination of movement
        if (any ([usr['final loc'][i]>self.edge for i in [0,1]])):
            print (usr['final loc'])
            exit ()
        usr['final ap' ] = self.loc2ap (usr['final loc'])
        self.eventQ.append ({'event type'   : EVENT_T_ARRIVE,
                             'time'         : self.arr_t (usr['cur loc'], usr['final loc'], usr['max speed']),
                             'usr'          : usr
                             })
        if (usr['final ap' ] == usr['cur ap']): # the move is within the coverage of the current AP - no need to schedule additional events 
            return
        theta = np.arctan (abs( (usr['final loc'][Y] - usr['cur loc'][Y]) / (usr['final loc'][X] - usr['cur loc'][X]) )) # angle of progression
        usr['speed'] = np.array ([math.cos(theta)*usr['max speed'], math.sin(theta)*usr['max speed']])  #Projection of the speed on the X, Y, direction
        self.mov_usr (usr)

    def mov_usr (self, usr):        
        time_to_mig = np.ones(2) * np.inf
        for dim in (X,Y): # for each dimension
            lower_borderline = self.cell_edge * math.floor(usr['cur loc'][dim] / self.cell_edge)
            offset = usr['cur loc'][dim] % self.cell_edge # offset of the usr's current location within its current cell
            if (usr['final loc'][dim] < lower_borderline):
                time_to_mig[dim] = offset / usr['speed'][dim]
            elif (usr['final loc'][dim] > lower_borderline + self.cell_edge):
                time_to_mig[dim] = (self.cell_edge - offset) / usr['speed'][dim]

        time_to_mig = min (time_to_mig)
        if (time_to_mig == np.inf): # the final destination is within the current cell - no need to schedule a future mig' event
            return
        
        # Deter fatal problem when the time to migration is 0 (happens when the user is exactly on the border between coverage areas) 
        time_to_mig = max (time_to_mig, EPSILON)
        
        direction = [1 if (usr['final loc'][dim] > usr['cur loc'][dim]) else -1 for dim in [X,Y]]
        usr['nxt loc'] = usr['cur loc'] + [time_to_mig*x*y for (x,y) in zip(usr['speed'], direction)]
        self.eventQ.append ({'event type'   : EVENT_T_MIG,
                             'time'         : self.cur_time + time_to_mig,
                             'usr'          : usr
                             })    
        
    def print_eventQ (self):
        """
        Print the event queue. Used for debugging
        """
        for event in self.eventQ:
            print (event)

    def print_locs (self):
        printf (self.loc_output_file, 'time = {:.4f} : \n' .format (self.cur_time)) 
        for u in range(self.NUM_OF_USRS):
            printf (self.loc_output_file, 'usr {} {:.1f} {:.1f}\n' .format (u, self.usr[u]['cur loc'][X], self.usr[u]['cur loc'][Y]))
        printf (self.loc_output_file, '\n')    

    def print_APs (self):
        
        printf (self.ap_output_file, 'time = {:.4f} : ' .format (self.cur_time)) 
        for u in range(self.NUM_OF_USRS):
            printf (self.ap_output_file, '({},{})' .format (u, self.usr[u]['cur ap']))
        printf (self.ap_output_file, '\n')
            
    def simulate (self, verbose):
        """
        Run a simulation.
        The simulation begins by letting all the usrs start move, thus also scheduling future events.
        From there and on, an event-driven simulation is run.
        - While (stop condition isn't met) 
            - Advance the time to the closest future event.
            - Dequeue that event from the event queue, and handle it.
                - If needed, schedule future events
        """
        self.verbose = verbose
        
        if (self.verbose > 0):
            # Overwrite old output file, if exists
            with open('../res/my_mob_sim.ap', 'w') as self.ap_output_file:
                self.ap_output_file.write('')
                
            self.ap_output_file  = open ("../res/my_mob_sim.ap",  "w")
            printf (self.ap_output_file,  '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of usr X at time t\n\n')

            # if (self.print_locations):                 
            #     self.loc_output_file = open ("../res/my_mob_sim.loc", "w")
            #     printf (self.loc_output_file, '// mobility - output by mobility_simulator.py\n')
            #     printf (self.loc_output_file, '// File format:\n//time = t:\n')
            #     printf (self.loc_output_file, '// usr U (X, Y)\n//where (X,Y) is the location of usr U at time t\n')
            #     printf (self.loc_output_file, '// the simulation is done in a rectangle of size MAX_X * MAX_Y, where\n')       
            #     printf (self.loc_output_file, 'MAX_X {:.0f} MAX_Y {:.0f}\n\n' .format(self.edge, self.edge))
 
        if (self.verbose > 1): # If the verbose level requires that, begin a series of periodical events  
            self.eventQ.append ({'event type' : EVENT_T_PERIODICAL,
                                     'time'   : 0})                 
           
        if (self.verbose > 0): # If the verbose level requires that, write to file the initial APs of each user
            self.print_APs()   
            
        # Start move the usrs   
        for u in range(self.NUM_OF_USRS):
            self.start_mov_usr (self.usr[u])
    
        while (1):
        
            if (self.eventQ): # event queue isn't empty
                
                # Pop the closest-future event, advance time and handle this event
                self.eventQ = sorted (self.eventQ, key = lambda event : event['time'])
                event = self.eventQ.pop (0)
                self.cur_time = event['time']
        
                if (event['event type'] == EVENT_T_MIG):
                    usr             = event['usr']  
                    usr ['cur loc'] = usr['nxt loc']
                    usr ['cur ap']  = self.loc2ap (usr['cur loc'])
                    if (self.verbose == 1 or self.verbose == 3):
                        self.print_APs ()
                    self.mov_usr (usr) 
                elif (event['event type'] == EVENT_T_ARRIVE):
                    usr = event['usr']  
                    usr ['cur loc'] = usr['final loc']
                    self.eventQ.append ({'event type' : EVENT_T_START_MOV,
                                         'time'       : self.cur_time + np.random.rand () * (usr['max stnd time'] - usr['min stnd time']),
                                         'usr'        : usr
                                         })
                elif (event['event type'] == EVENT_T_START_MOV):
                    self.start_mov_usr(event['usr'])
                elif (event['event type'] == EVENT_T_PERIODICAL):
                    self.print_APs()
                    self.eventQ.append ({'event type' : EVENT_T_PERIODICAL,
                                         'time'         : self.cur_time + self.T_BETWEEN_PERIODICAL_EVENTS
                                        })                    
                else:
                    print ('Error: unsupported event type')
                    exit ()
        
            if (self.cur_time > self.MAX_TIME):
                exit ()           
    
    def __init__ (self):
        
        self.edge = 100 # edge of the rectangle in which usr move [m]
        self.num_of_APs_in_row = 7
        self.cell_edge = self.edge / self.num_of_APs_in_row
        self.NUM_OF_USRS  = 5

        self.usr = [{} for u in range (self.NUM_OF_USRS)]
        for u in range (self.NUM_OF_USRS):
            self.usr[u] = {'max speed'      : (30 + 5*u)/ 3.6, 
                            'cur loc'       : np.random.rand (2) * self.edge,
                            'min stnd time' : 0.5,
                            'max stnd time' : 1.5
                            }  # speed [m/sec]
            self.usr[u]['cur ap'] = self.loc2ap (self.usr[u]['cur loc'])
        
        self.eventQ   = []
        self.cur_time = 0
        self.MAX_TIME = 20

        self.T_BETWEEN_PERIODICAL_EVENTS = 1 # time between sequencing prints of all usrs' locations        

if __name__ == "__main__":
    sim = my_mobility_simulator ()
    # sim.print_locations = True # For printing also users' locations. Currently un-supported, because finding fresh accurate users' location requires complicated calculations. 
    sim.simulate (verbose = VERBOSE_EVENTS_AND_PERIODICAL)
    exit
        