import numpy as np

from toy_example import toy_example
from LP_file_parser import LP_file_parser
from Check_LP_sol import Check_LP_sol

run_toy = True
if (run_toy):
    my_toy_example = toy_example (verbose = 1)
else:
    X = np.zeros (18)
    X[6] = 1
    X[15] = 1
    res = Check_LP_sol (X)
    if (res):
        print ('feasible')
    else:
        print ('NOT feasible')


# my_LP_file_parser = LP_file_parser ()
# my_LP_file_parser.parse_LP_file ('custom_tree.LP')
