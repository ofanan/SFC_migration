import sys
import numpy as np

from toy_example import toy_example

gamad = None

my_toy_example = toy_example (verbose = 1)

my_toy_example.run (gen_LP = True,  run_brute_force = False) # Generate code for the LP
