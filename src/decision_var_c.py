class decision_var_c (object):
    """
    class for of "decision variable" 
    """ 
    def __init__ (self, lp_var, id, usr, lvl=0, s=0):
        self.lp_var = lp_var
        self.id  = id
        self.usr = usr
        self.lvl = lvl
        self.s   = s
         
