import numpy as np
import itertools 


class LP_file_parser (object):
    
    
    def parse_line (self, line):
        splitted_line = line.split (" ")
         
        if (splitted_line[0] == 'subject' and splitted_line[1] == 'to'):
            print ('gamad') 
        elif (splitted_line[0] == 'minimize' and splitted_line[1] == 'z:'):
            print ('nanas')

    
    def parse_LP_file (self, input_file_name):
        
        self.input_file  = open ("../res/" + input_file_name,  "r")
        # self.output_file = open ("../res/" + input_file_name.split(".")[0] + ".dat", "a")  
        lines = (line.rstrip() for line in self.input_file) # "lines" contains all lines in input file
        lines = (line for line in lines if line)       # Discard blank lines
        self.cost_func   = []
        self.constraints = []
    
    
        for line in lines:
        
            # Discard lines with comments / verbose data
            if (line.split ("//")[0] == ""):
                continue
           
            self.parse_line(line)
