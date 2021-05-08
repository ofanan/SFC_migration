import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf
#from builtins import True

class my_mobility_simulator (object):

    min_mov   = lambda self, u, dim          : abs(0.5*self.edge - self.user[u]['loc'][dim]) # caclulate the min move required to verify that the user switches to another AP
    direction = lambda self, u, dim       : 1 if (self.user[u]['loc'][dim] < 0.5 * self. edge) else -1
    dst       = lambda self, direction    : 0.5 * self.edge * (1 + np.random.rand () * direction) #if (self.user[u]['loc'][dim] < 0.5 * self.edge) else 0.5 * self.edge * (1 - np.random.rand ())  
    arr_t     = lambda self, u, dst, dim     : abs (dst - self.user[u]['loc'][dim]) / self.user[u]['speed']
    nxt_loc   = lambda self, u, dst, dim     : [ self.user[u]['loc'][self.X], dst] if (dim == self.X) else [dst, self.user[u]['loc'][self.X] ]  
    mig_t     = lambda self, u, min_mov      : min_mov / self.user[u]['speed']
    nxt_ap    = lambda self, u, direction, dim : self.user[u]['ap'] + direction if dim == self.X else self.user[u]['ap'] + 2 * direction 
#     nxt_ap  = lambda self, u, dim          : 2 * int(self.user[u]['ap'] / 2) + 1 - int (self.user[u]['ap'] % 2) if (dim == x) else \
      
            
    def move_user (self, u):
        
        #dim     = random.getrandbits(1) # dimension of nxt movement (either self.X or self.Y)
        dim = self.Y
        min_mov   = self.min_mov (u, dim) # minimal move required to verify that the user switches to another AP
        direction = self.direction (u, dim)
        dst       = self.dst     (direction)  # destination of movement
#         print ('cur loc = ', self.user[u]['loc'][self.X], 'min mov = ', min_mov, 'dst = ', dst)
#         self.eventQ.append ({'event type'   : self.arrive,
#                              'nxt loc'      : self.nxt_loc (u, dst, dim),
#                              'time'         : self.arr_t (u, dst, dim)
#                              })
        
        self.eventQ.append ({'event type'   : self.mig,
                             'time'         : self.mig_t (u, dim), 
                             'nxt AP'       : self.nxt_ap (u, direction, dim)
                             })
        print ('cur ap = ', self.user[u]['ap'])
        print (self.eventQ)
        # time    =  
        # if (dim == self.X):  
        #
        #     # add an event for the arrival of the user to its dest
        #     self.eventQ.append ({'event type'   : self.arrive,
        #                          'dst'          : dst (u, min_mov, dim)
        #                          'time'         : min_move / self.speed_of_user[u],
        #                          })
        #
        #     #print (np.random.rand () * (0.5*self.edge - min_move))
        #     print (dst)
        #     # self.eventQ.append ({'time : '
        #     #                     })
        # else:
        #     print ('y')


    def simulate (self):
        self.move_user (1)
    
    
    def __init__ (self):
        
        self.edge = 100 # edge of the rectangle in which user move 
        self.NUM_OF_user  = 3

        # self.user = np.empty(self.NUM_OF_user, dytpe = 'object')
        # moving direction (either X or Y)
        self.X = 0
        self.Y = 1
        self.user = [{} for u in range (self.NUM_OF_user)]
        for u in range (self.NUM_OF_user):
            self.user[u] = {'speed' : 30 + 5*u, 'loc'   : np.random.rand (2) * self.edge}
            self.user[u]['ap'] = 2 * int (self.user[u]['loc'][self.Y] / (0.5*self.edge)) + int (self.user[u]['loc'][self.X] / (0.5*self.edge))
        
        self.eventQ = []
        self.cur_time = 0
        self.max_time = 100
        
        self.mig    = True
        self.arrive = False
        
 
        # print (np.random.choice (a=[False, True], size=(self.NUM_OF_user)))


        
        # for u in range (self.NUM_OF_user):
        #     self.eventQ.append ({'time'       : 5,
        #                          'event type' : self.mig}
        #
        #
        #                         )
        
        # while (1):
        #
        #     if (self.eventQ): # event queue isn't empty
        #         exit ()
        #
        #     self.cur_time += 1
        #     if (self.cur_time > self.max_time):
        #         print ('egeg')
        #         exit () 
            


if __name__ == "__main__":
    sim = my_mobility_simulator ()
    sim.simulate ()
    exit
        
