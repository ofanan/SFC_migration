class usr_c (object):
    """
    class of "user", used by alg_top 
    """ 
    def __init__ (self, id, theta_times_lambda=[1,1,1], target_delay=6, mig_cost=1, C_u=15, cur_s = -1, nxt_s = -1):
        self.id                 = id
        self.theta_times_lambda = theta_times_lambda    
        self.target_delay       = target_delay
        self.mig_cost           = mig_cost              # total mig' cost of this chain
        self.C_u                = C_u   # max # of CPU units that this usr is allowed to use
        self.lvl                = -1    # mark the user as un-allocated
        self.S_u                = []    # List of servers that are delay-feasible for this usr.
        self.cur_s              = cur_s # current placement (server)
        self.nxt_s              = nxt_s # next (scheduled) server
         
    def __lt__ (self, other):
        """
        Used to sort usrs, based on the number of CPU units they use
        """
        return (self.B[self.lvl] < other.B[other.lvl])
