class decision_var_c (object):
    """
    class for of "decision variable" 
    """ 
    def __init__ (self, plp_lp_var, id, usr, lvl=0, s=0):
        self.plp_lp_var = plp_lp_var
        self.id  = id
        self.usr = usr
        self.lvl = lvl
        self.s   = s
         
