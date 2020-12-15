import sys
import numpy as np
import cplex
from doopl.factory import *
import docplex
from printf import printf



def solve_problem_by_Cplex (input_file_name):
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
    printf (output_file, 'Solution value  = {:.2f}\n' .format (sol.get_objective_value()))
    print  (             'lp sol cost = {:.2f}\n' .format (sol.get_objective_value()))
    printf (output_file, '\nList of non-zeros decision variables\n************************************\n' .format (sol.get_objective_value()))
    list_of_set_vars = []
    for var in problem.variables.get_names():
        var_val = sol.get_values (var)
        if (var_val == 0):
            continue
        if (var_val != 1):
            print ('Warning: Non-boolean value in the solution: {} = {}' .format (var, var_val))
        printf (output_file, '{} = {:.2f}, ' .format (var, sol.get_values (var)))
        list_of_set_vars.append ( int(var.split('x')[1]) )
    printf (output_file, '\n')
    return list_of_set_vars

if __name__ == "__main__":
    solve_problem_by_Cplex ('../res/problem.lp')

