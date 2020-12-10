import sys
import numpy as np
import cplex
from doopl.factory import *
import docplex

problem = cplex.Cplex ("shorter.lp") 
problem.solve ()
sol = problem.solution
status = sol.get_status()
print ('Sol status = ', status)
print ('Sol found is: x1 = {}, x2 = {}' .format (sol.get_values ('x1'), sol.get_values ('x2')))
print ("Solution value  = ", sol.get_objective_value())

# if (problem.solve ()):
#     print ('b4')
#     print(cplex.getObjValue())
#     print ('after')
# else:
#     print ('Did not find a sol')
#     