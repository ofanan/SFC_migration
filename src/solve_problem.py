import sys
import numpy as np

from toy_example import toy_example

my_toy_example = toy_example (verbose = 1)

my_toy_example.run (gen_LP = False, run_brute_force = True)  # Brute-force solve the LP
