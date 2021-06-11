class usr_c (object):
    """
    class for of "user" 
    """ 
    def __init__ (self, id, cur_cpu=0, theta_times_lambda=[1,1,1], target_delay=15, mig_cost=1, C_u=0):
        self.id                 = id
        self.cur_cpu            = cur_cpu
        self.theta_times_lambda = theta_times_lambda
        self.target_delay       = target_delay
        self.mig_cost           = mig_cost
        self.C_u                = C_u
        self.lvl                = -1
        self.S_u                = []
         
    def __lt__ (self, other):
        return (self.B[self.lvl] < other.B[other.lvl])
