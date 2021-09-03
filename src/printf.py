from __future__ import print_function

def printf(output_file, str, *args):
    """
    Format-print the requested string to a given output file 
    """
    print(str % args, end='', file = output_file, flush = True)

