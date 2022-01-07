import random

min_rand_id = 1
max_rand_id = 1000

class usr_c (object):

    """
    class of "user", used by algorithmic solutions for the mig' problem (e.g., first fit, cpvnf, and our BUPU algorithm). 
    """ 
    def __init__ (self, id, theta_times_lambda, target_delay, C_u, mig_cost=1, cur_s = -1, nxt_s = -1):
        self.id                   = id
        self.theta_times_lambda   = theta_times_lambda    
        self.target_delay         = target_delay
        self.mig_cost             = mig_cost              # total mig' cost of this chain
        self.C_u                  = C_u   # max # of CPU units that this usr is allowed to use
        self.lvl                  = -1    # mark the user as un-allocated
        self.S_u                  = []    # List of servers that are delay-feasible for this usr.
        self.cur_s                = cur_s # current placement (server)
        self.nxt_s                = nxt_s # next (scheduled) server
        self.rand_id              = random.randint (min_rand_id, max_rand_id)
        self.criticality_duration = int(1) # For how many slots this usr is already critical. Upon init the value is 1, because a new usr is already critical for 1 (smallest) time unit, by definition.
             
    def __lt__ (self, other):
        """
        Used to sort usrs, based on the number of CPU units they use
        """
        return (self.B[self.lvl] < other.B[other.lvl])
