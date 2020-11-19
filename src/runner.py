import sys
import numpy as np

from toy_example import toy_example
from LP_file_parser import LP_file_parser

my_toy_example = toy_example (verbose = 1)


if (str(sys.argv[0])=="0"):
    my_toy_example.run (gen_LP = True,  run_brute_force = False) # Generate code for the LP
else:
    my_toy_example.run (gen_LP = False, run_brute_force = True)  # Brute-force solve the LP

# my_LP_file_parser = LP_file_parser ()
# my_LP_file_parser.parse_LP_file ('custom_tree.LP')
