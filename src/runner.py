import sys
import numpy as np
# import cplex
# from doopl.factory import *
# import docplex

# from toy_example import toy_example
# from LP_file_parser import LP_file_parser

#cpx = cplex.Cplex ("short.lp")

# with cplex.Cplex () as cpx:
#     cpx.read ("þþshort.lp", "lp")
    #C:\Users\ofanan\Documents\GitHub\SFC_migration\src

# with create_opl_model (model="shorter.mod") as opl:
#     opl.run ()
#     print ('rgrg') 
    #print("Table names are: "+ str(opl.output_table_names))
    #print("Table names are: "+ str(cpx.output_table_names))

#         opl.set_input("TupleSet1", tuples)


# cpx = cplex.Cplex()
# cpx.read ('short.mod')
# #out = cpx.set_results_stream ('res.cplx', "w")
# out = cpx.set_log_stream(None)
# cplex_solver.read("example.mps")
# cplex_solver.solve()
# cplex_solver.solution.get_status_string()
# my_toy_example = toy_example (verbose = 1)
# 
# if (len(sys.argv) > 2):  # run a parameterized sim'   
#     chain_target_delay = float (str(sys.argv[2]))
#      
#     if (str(sys.argv[1])=="G"): # G --> Gen problem. "S" --> Solve problem
#         my_toy_example.run (chain_target_delay = chain_target_delay, gen_LP = True,  run_brute_force = False) # Generate code for the LP
#     else:
#         my_toy_example.run (chain_target_delay = chain_target_delay, gen_LP = False, run_brute_force = True)  # Brute-force solve the LP
# else:
#     if (str(sys.argv[1])=="G"): # G --> Gen problem. "S" --> Solve problem
#         my_toy_example.run (gen_LP = True,  run_brute_force = False) # Generate code for the LP
#     else:
#         my_toy_example.run (gen_LP = False, run_brute_force = True)  # Brute-force solve the LP
