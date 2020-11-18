import numpy as np

from toy_example import toy_example
from LP_file_parser import LP_file_parser
from Check_LP_sol import Check_LP_sol



my_toy_example = toy_example (verbose = 1)
gen_LP          = False
run_brute_force = True
my_toy_example.run (gen_LP = gen_LP, run_brute_force = run_brute_force)
# my_LP_file_parser = LP_file_parser ()
# my_LP_file_parser.parse_LP_file ('custom_tree.LP')
