from __future__ import print_function
import numpy as np

def printf(output_file, str2print, *args):
    """
    Format-print the requested str2printing to a given output file 
    """
    print(str2print % args, end='', file = output_file, flush = True)


def printar (output_file, ar):
    ar=np.array(ar)
    printf (output_file, '{}\n' .format(str(ar).replace('\n', '')))
