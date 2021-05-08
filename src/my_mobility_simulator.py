import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf
#from builtins import True

class my_mobility_simulator (object):

    calc_min_mov = lambda u, dim          : abs(0.5*self.edge - self.user[u]['loc'][dim]) # caclulate the min move required to verify that the user switches to another PoA
    calc_dst     = lambda u, min_mov, dim : (self.user[u]['loc'][dim] + np.random.rand () * (0.5*self.edge - min_mov)) % self.edge 
    calc_arr_t   = lambda u, dst, dim     : abs (dst - self.user[u]['loc'][dim]) / self.users[u]['speed']
    lambda_func_try = lambda x : x 
            
    def move_user (self, u):
        
        print ('lambda = ', lambda_func_try (x))
        # dim     = random.getrandbits(1) # dimension of nxt movement (either self.X or self.Y)
        # min_mov = calc_min_mov (u, dim)
        # dst     = calc_dst     (u, min_mov, dim)  
        # self.eventQ.append ({'event type'   : self.arrive,
        #                      'nxt loc'      : [self.user[u]['loc'][self.x], dst],
        #                      'time'         : calc_arr_t (u, dst, dim)
        #                      })
        # time    =  
        # if (dim == self.X):  
        #
        #     # add an event for the arrival of the user to its dest
        #     self.eventQ.append ({'event type'   : self.arrive,
        #                          'dst'          : calc_dst (u, min_mov, dim)
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
            self.user[u]['PoA'] = 2 * int (self.user[u]['loc'][self.Y] / (0.5*self.edge)) + int (self.user[u]['loc'][self.X] / (0.5*self.edge))
        
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
        
