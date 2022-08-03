class decision_var_c (object):
    """
    class for a "decision variable". 
    When simulating, the "next state decision var" becomes the "current state parameter"; 
    the "cur_st_val" is the value found by the fractional LP solution for the prob' in the previous time slot. 
    """ 
    def __init__ (self, usr, plp_var=None, grb_var=None, d_var_id=-1, lvl=0, s=0, cur_st=0):
        self.plp_var  = plp_var
        self.grb_var  = grb_var
        self.d_var_id = d_var_id
        self.usr      = usr # this decision_var is about locating this usr on server s, whose level is lvl
        self.s        = s
        self.lvl      = lvl
        self.cur_st   = cur_st # value of this decision_var at the current state 
        self.val      = 0
        
    def getValue (self):
        """
        returns the value associated with this decision var.
        If no value is associated with this decision var (no value was written to the decision variable yet, e.g., the problem wasn't solved yet), return 0.
        """
        if (self.plp_var != None):
            return self.plp_var.value()
        elif (self.grb_var != None):
            return self.val
        else:
            return 0
