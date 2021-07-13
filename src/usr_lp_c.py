class usr_lp_c (object):
    """
    class of "user", used by the lp 
    """ 
    def __init__ (self, id, theta_times_lambda=[1,1,1], target_delay=6, mig_cost=1, C_u=15):
        self.id                 = id
        self.theta_times_lambda = theta_times_lambda    
        self.target_delay       = target_delay
        self.mig_cost           = mig_cost              # total mig' cost of this chain
        self.C_u                = C_u   # max # of CPU units that this usr is allowed to use
        self.S_u                = []    # List of servers that are delay-feasible for this usr.
         
