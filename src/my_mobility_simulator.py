import networkx as nx
import numpy as np
import math
import itertools 
import time
import random

from printf import printf
#from builtins import True

class my_mobility_simulator (object):

    def move_user (self, u):
        
        gamad = self.X
        if (gamad == self.X): #(random.getrandbits(1) == self.X): 
            
            # add an event for the arrival of the user to its dest
            min_move = abs(0.5*self.edge - self.cur_loc_of_usr[u][self.X]) 
            self.eventQ.append ({'time' : min_move / self.speed_of_user[u],
                                 'event type' : self.arrive})
            
            dst = (self.cur_loc_of_usr[u][self.X] + np.random.rand () * (0.5*self.edge - min_move)) % self.edge 
            #print (np.random.rand () * (0.5*self.edge - min_move))
            print (dst)
            # self.eventQ.append ({'time : '
            #                     })
        else:
            print ('y')


    def simulate (self):
        self.move_user (1)
    
    
    def __init__ (self):
        
        self.edge = 100 # edge of the rectangle in which users move 
        self.NUM_OF_USERS  = 3
        self.speed_of_user  = np.array ([(30 + 5*i) for i in range (self.NUM_OF_USERS)])
        self.cur_loc_of_usr = np.random.rand (self.NUM_OF_USERS, 2) * self.edge 
        
        self.eventQ = []
        self.cur_time = 0
        self.max_time = 100
        
        self.mig    = True
        self.arrive = False
        
        # moving direction (either X or Y)
        self.X = 0
        self.Y = 1
 
        print (np.random.choice (a=[False, True], size=(self.NUM_OF_USERS)))


        
        # for u in range (self.NUM_OF_USERS):
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
        
