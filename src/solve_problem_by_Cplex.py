import sys
import numpy as np
import cplex
from doopl.factory import *
import docplex
from printf import printf

def solve_problem_by_Cplex (input_file_name, num_of_vars):
    """
    Accept as input a file containing a LP problem, at an .lp format, and the number of decision variables.
    Solves the problem using Cplex, and then prints the sol's status, the obj's func value, and the decision var' vals. 
    """

    output_file = open ('../res/' + input_file_name.split(".lp")[0] + '.cpx.sol', "w")
    problem = cplex.Cplex (input_file_name) 
    problem.solve ()
    sol     = problem.solution
    status  = sol.get_status()
    printf (output_file, 'Sol status = {}\n' .format (status))
#     val = sol.get_objective_value()
    printf (output_file, 'Solution value  = {:.2f}\n' .format (sol.get_objective_value()))
    for var in range (1, num_of_vars+1):
        printf (output_file, 'x{} = {:.2f}, ' .format (var, sol.get_values ('x' + str(var))))
    printf (output_file, '\n')

if __name__ == "__main__":
    solve_problem_by_Cplex ('shorter.lp', 2)
