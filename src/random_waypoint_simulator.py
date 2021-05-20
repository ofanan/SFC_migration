import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf # a patch for easy format-printing into files.

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

class random_waypoint_simulator (object):
    """
    Event-driven mobility simulator.
    Users randomly move within a square, using a random waypoint model.
    The square in which users move is partitioned into an n X n grid of squares, where each square is exclusively covered by a single Access Point (AP).

    Each time a user enters an area covered by an AP different from his current AP, an event happens.
    
    Output: 
    The current APs serving all users are written to a file when either 
    1. A migration event happens (namely, a user enters an area covered by an AP different from his current AP).
    2. Periodically.  
    It is possible to select the level of "verbose", that is, whether the output file is written either upon 1 and/or 2 from above.
     

    By default, output files are written to a sibling directory named "../res". If you don't have, please generate such a directory. 
    The extension of the output file is ".ap", for Access Point.
    """

    # Calculate the arrival time, given the current location, the destination and the speed.
    arr_t      = lambda self, cur_loc, dst, speed : self.cur_time + math.sqrt (pow(cur_loc[X] - dst[X], 2) + pow(cur_loc[Y] - dst[Y], 2)) / speed

    # Calculate which AP covers a concrete point. The input, "loc", is a vector, containing the X,Y locations.
    loc2ap     = lambda self, loc : int (math.floor ((loc[Y] / self.cell_edge) ) * self.num_of_APs_in_row + math.floor ((loc[X] / self.cell_edge) )) 
    
    # Update the location of a usr. New location is calculated as: 
    # last_updated_location + time_since_last_location_update * directional_speed, for each dimension (that is, X and Y).
    update_loc = lambda self, usr : usr['cur loc'] + (self.cur_time - usr['last update time']) * usr['speed']
      
    def start_mov_usr (self, usr):
        """
        Start a move of a usr: 
        - Pick the destination (usr['final loc']) u.a.r from the area
        - Generate an event for the arrival to the destination, and insert it into the eventQ.
        - If the destination is within the coverage area of another AP, generate an additional event, for the migration to that AP.
        """

        usr['final loc'] = self.edge * np.random.rand (2)  # destination of movement
        usr['final ap'        ] = self.loc2ap (usr['final loc'])
        self.calc_usr_speed(usr)
        self.eventQ.append ({'event type'   : EVENT_T_ARRIVE,
                             'time'         : self.arr_t (usr['cur loc'], usr['final loc'], usr['max speed']),
                             'usr'          : usr
                             })
        if (usr['final ap' ] == usr['cur ap']): # the move is within the coverage of the current AP - no need to schedule additional events 
            return
        self.mov_usr (usr)

    def calc_usr_speed (self, usr): 
        """
        Calculate the projectory of the speed of a given usr in each direction, as follows:
        - pick a speed u.a.r. from [usr['min speed'], usr['max speed']].
        - calculate the projection of the speed on the X, Y axes; take into the account (speed[x] is negative when the direction is to smaller x; same with speed[y]). 
        - Write the speed to usr['speed']  
        """
        speed = random.uniform (usr['min speed'], usr['max speed']) 
        theta = np.arctan (abs( (usr['final loc'][Y] - usr['cur loc'][Y]) / (usr['final loc'][X] - usr['cur loc'][X]) )) # angle of progression
        usr['speed'] = np.array ([math.cos(theta)*usr['max speed'], math.sin(theta)*usr['max speed']])  #Projection of the speed on the X, Y, direction
        usr['speed'] = np.array([usr['speed'][dim] if (usr['final loc'][dim] > usr['cur loc'][dim]) else -usr['speed'][dim] for dim in [X,Y]])        

    def mov_usr (self, usr):        
        time_to_mig = np.ones(2) * np.inf
        for dim in (X,Y): # for each dimension
            lower_borderline = self.cell_edge * math.floor(usr['cur loc'][dim] / self.cell_edge)
            offset = usr['cur loc'][dim] % self.cell_edge # offset of the usr's current location within its current cell
            if (usr['final loc'][dim] < lower_borderline):
                time_to_mig[dim] = offset / abs(usr['speed'][dim])
            elif (usr['final loc'][dim] > lower_borderline + self.cell_edge):
                time_to_mig[dim] = (self.cell_edge - offset) / abs(usr['speed'][dim])

        time_to_mig = min (time_to_mig)
        if (time_to_mig == np.inf): # the final destination is within the current cell - no need to schedule a future mig' event
            return
        
        # Deter fatal problem when the time to migration is 0 (happens when the user is exactly on the border between coverage areas) 
        time_to_mig = max (time_to_mig, EPSILON)
        
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
        """
        Print the exact locations - namely, (x,y) coordinates, of each user.
        Currently unused.
        """
        printf (self.loc_output_file, 'time = {:.4f} : \n' .format (self.cur_time)) 

        for usr in self.usr: 
            usr['cur loc']          = self.update_loc (usr) 
            usr['last update time'] = self.cur_time
            printf (self.loc_output_file, 'usr {} {:.1f} {:.1f}\n' .format (usr['id'], usr['cur loc'][X], usr['cur loc'][Y]))
        printf (self.loc_output_file, '\n')    

    def print_APs (self):
        """
        Print the current time, and the list of the IDs of APS currently covering each AP. 
        """        
        printf (self.ap_output_file, 'time = {:.4f} :\n' .format (self.cur_time)) 
        for u in range(self.NUM_OF_USRS):
            printf (self.ap_output_file, '({},{})' .format (u, self.usr[u]['cur ap']))
        printf (self.ap_output_file, '\n')
            
    def print_output (self):
        if (self.print_APs_flag):
            self.print_APs()
        if (self.print_locs_flag):
            self.print_locs()  
        if (self.print_trajectory_of_usr >= 0): # and any ([self.usr[self.print_trajectory_of_usr]['speed'][dim] > 0 for dim in [X,Y]])):
            printf (self.trajectory_output_file, '({:.2f}, {:.2f})' .format (self.usr[self.print_trajectory_of_usr]['cur loc'][X],
                                                                             self.usr[self.print_trajectory_of_usr]['cur loc'][Y]
                                                                             ))        
            
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
            with open('../res/mob_sim.ap', 'w') as self.ap_output_file:
                self.ap_output_file.write('')                
            self.ap_output_file  = open ("../res/mob_sim.ap",  "w")
            
            printf (self.ap_output_file,  '// File format:\n//time = t: (1,a1),(2,a2), ...\n//where aX is the Point-of-Access of usr X at time t\n\n')

            if (self.print_locs_flag):                 
                self.loc_output_file = open ("../res/mob_sim.loc", "w")
                printf (self.loc_output_file, '// mobility - output by mobility_simulator.py\n')
                printf (self.loc_output_file, '// File format:\n//time = t:\n')
                printf (self.loc_output_file, '// usr U (X, Y)\n//where (X,Y) is the location of usr U at time t\n')
                printf (self.loc_output_file, '// the simulation is done in a rectangle of size MAX_X * MAX_Y, where\n')       
                printf (self.loc_output_file, 'MAX_X {:.0f} MAX_Y {:.0f}\n\n' .format(self.edge, self.edge))
                
            if (self.print_trajectory_of_usr >= 0):
                self.trajectory_output_file = open ("../res/mob_sim.trj", "w")
 
        if (self.verbose > 0): # If the verbose level requires that, write to file the initial APs of each user
            self.print_output()
            
            if (self.verbose > 1): # If the verbose level requires that, schedule the first periodical event (the next event will schedule each other)  
                self.eventQ.append ({'event type' : EVENT_T_PERIODICAL,
                                         'time'   : self.T_BETWEEN_PERIODICAL_EVENTS})                 
               
        # Start move the usrs   
        for usr in self.usr:
            self.eventQ.append ({'event type' : EVENT_T_START_MOV,
                                 'time'       : random.uniform(usr['min stnd time'], usr['max stnd time']), 
                                 'usr'        : usr
                                 })
    
        while (1):
        
            if (self.eventQ): # event queue isn't empty
                
                # Pop the closest-future event, advance time and handle this event
                self.eventQ = sorted (self.eventQ, key = lambda event : event['time'])
                event = self.eventQ.pop (0)
                self.cur_time = event['time']
        
                if (event['event type'] == EVENT_T_MIG):
                    usr = event['usr']  
                    usr ['cur loc'         ] = self.update_loc (usr)
                    usr ['cur ap'          ] = self.loc2ap (usr['cur loc'])
                    usr ['last update time'] = self.cur_time
                    if (self.verbose == 1 or self.verbose == 3):
                        self.print_output ()
                    self.mov_usr (usr) 
                elif (event['event type'] == EVENT_T_ARRIVE):
                    usr = event['usr']  
                    usr ['cur loc'         ] = usr['final loc']
                    usr ['last update time'] = self.cur_time
                    usr ['speed'           ] = np.zeros (2)
                    self.eventQ.append ({'event type' : EVENT_T_START_MOV,
                                         'time'       : self.cur_time + random.uniform(usr['min stnd time'], usr['max stnd time']),
                                         'usr'        : usr
                                         })
                    if (usr['id'] == self.print_trajectory_of_usr):
                        printf (self.trajectory_output_file, '\n\n')
                elif (event['event type'] == EVENT_T_START_MOV):
                    usr['last update time'] = self.cur_time
                    self.start_mov_usr(event['usr'])
                elif (event['event type'] == EVENT_T_PERIODICAL):
                    self.print_output()
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
        self.num_of_APs_in_row = 7 # The total number of APs covering the (square) area is self.num_of_APs_in_row * self.num_of_APs_in_row 
        self.cell_edge = self.edge / self.num_of_APs_in_row # Edge of a single square cell, exclusively covered by a single AP.
        self.NUM_OF_USRS  = 5

        self.usr = [{} for u in range (self.NUM_OF_USRS)]
        for u in range (self.NUM_OF_USRS):
            self.usr[u] = {'id'               : u,
                           'min speed'        : 30 / 3.6,
                           'max speed'        : 80 / 3.6,                            
                           'min stnd time'    : 0.5,
                           'max stnd time'    : 1.5,
                           'cur loc'          : np.random.rand (2) * self.edge,
                           'last update time' : 0.0,  # last time were location was calculated 
                           'speed'            : np.zeros(2) # positional speed to [X,Y] directional. Initialized to zero 
                            }  # speed [m/sec]
            self.usr[u]['final loc'] = self.usr[u]['cur loc'] 
            self.usr[u]['cur ap']    = self.loc2ap (self.usr[u]['cur loc'])
        
        self.eventQ   = []
        self.cur_time = 0
        self.MAX_TIME = 20

        self.T_BETWEEN_PERIODICAL_EVENTS = 0.1 # time between sequencing prints of all usrs' locations        

if __name__ == "__main__":
    sim = random_waypoint_simulator ()
    sim.print_APs_flag  = True
    sim.print_locs_flag = True # For printing also users' locations. Currently un-supported, because finding fresh accurate users' location requires complicated calculations.
    sim.print_trajectory_of_usr = 0 # Print the trajectory of the requested user. For plotting no trajectories, assign -1.   
    sim.simulate (verbose = VERBOSE_ONLY_PERIODICAL)
        
