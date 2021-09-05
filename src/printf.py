from __future__ import print_function

def printf(output_file, str2print, *args):
    """
    Format-print the requested str2printing to a given output file 
    """
    print(str2print % args, end='', file = output_file, flush = True)

