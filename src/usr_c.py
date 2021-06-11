class usr_c (object):
    """
    class for of "user" 
    """ 
    def __init__ (self, id, cur_cpu=0, theta_times_lambda=[1,1,1], target_delay=15, mig_cost=1, C_u=15, cur_s = -1, nxt_s = -1):
        self.id                 = id
        self.cur_cpu            = cur_cpu
        self.theta_times_lambda = theta_times_lambda
        self.target_delay       = target_delay
        self.mig_cost           = mig_cost
        self.C_u                = C_u
        self.lvl                = -1 # mark the user as un-allocated
        self.S_u                = []
        self.cur_s              = cur_s # current placement (server)
        self.nxt_s              = nxt_s # next (scheduled) server
         
    def __lt__ (self, other):
        return (self.B[self.lvl] < other.B[other.lvl])
