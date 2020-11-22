import sys
import numpy as np

from toy_example import toy_example
from LP_file_parser import LP_file_parser

gen_prob = 0

my_toy_example = toy_example (verbose = 1)

uniform_mig_cost = float (str(sys.argv[2]))

if (int(sys.argv[1])==gen_prob):
    my_toy_example.run (uniform_mig_cost = uniform_mig_cost, gen_LP = True,  run_brute_force = False) # Generate code for the LP
else:
    my_toy_example.run (uniform_mig_cost = uniform_mig_cost, gen_LP = False, run_brute_force = True)  # Brute-force solve the LP

