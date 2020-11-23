import sys
import numpy as np

from toy_example import toy_example
from LP_file_parser import LP_file_parser

my_toy_example = toy_example (verbose = 1)

if (len(sys.argv) > 2):  # run a parameterized sim'   
    chain_target_delay = float (str(sys.argv[2]))
     
    if (str(sys.argv[1])=="G"): # G --> Gen problem. "S" --> Solve problem
        my_toy_example.run (chain_target_delay = chain_target_delay, gen_LP = True,  run_brute_force = False) # Generate code for the LP
    else:
        my_toy_example.run (chain_target_delay = chain_target_delay, gen_LP = False, run_brute_force = True)  # Brute-force solve the LP
else:
    if (str(sys.argv[1])=="G"): # G --> Gen problem. "S" --> Solve problem
        my_toy_example.run (gen_LP = True,  run_brute_force = False) # Generate code for the LP
    else:
        my_toy_example.run (gen_LP = False, run_brute_force = True)  # Brute-force solve the LP
