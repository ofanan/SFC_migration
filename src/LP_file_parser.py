import numpy as np
import itertools 
import re

class LP_file_parser (object):
    
    
    def parse_lin_comb (self, lin_comb_string):
        filtered = filter(None, re.split("[, \-+]+", lin_comb_string))
        for item in filtered:
            print (item)
        
    
    
    def parse_line (self, line):
        splitted_line = line.split (" ")
         
        if (splitted_line[0] == 'subject' and splitted_line[1] == 'to'):
            constraint_string_splitted = line.split(":")[1].split("=")
            constraint_string_split_lt = constraint_string_splitted[0].split("<")
            op = '='
            if (len (constraint_string_split_lt)==2):                
                op = '<='
            else:
                opt = '='
                self.parse_lin_comb(constraint_string_split_lt[0])
                print ('******')
        elif (splitted_line[0] == 'minimize' and splitted_line[1] == 'z:'):
            objective_func_string = line.split(":")[1]

    
    def parse_LP_file (self, input_file_name):
        """
        Parse a file of a LP format. Extract the constraints.
        """
        
        self.input_file  = open ("../res/" + input_file_name,  "r")
        # self.output_file = open ("../res/" + input_file_name.split(".")[0] + ".dat", "a")  
        lines = (line.rstrip() for line in self.input_file) # "lines" contains all lines in input file
        lines = (line for line in lines if line)       # Discard blank lines

        self.cost_func   = []
        self.constraints = []
        self.const_num   = 0
    
    
        for line in lines:
        
            # Discard lines with comments / verbose data
            if (line.split ("//")[0] == ""):
                continue
           
            self.parse_line(line)
