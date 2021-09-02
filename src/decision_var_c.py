class decision_var_c (object):
    """
    class for a "decision variable". 
    When simulating, the "next state decision var" becomes the "current state parameter"; 
    the "cur_st_val" is the value found by the fractional LP solution for the prob' in the previous time slot. 
    """ 
    def __init__ (self, usr, plp_var=None, d_var_id=-1, lvl=0, s=0, cur_st=0):
        self.plp_var  = plp_var
        self.d_var_id = d_var_id
        self.usr      = usr # this decision_var is about locating this usr on server s, whose level is lvl
        self.s        = s
        self.lvl      = lvl
        self.cur_st   = cur_st # val of this decision_var at the current state 