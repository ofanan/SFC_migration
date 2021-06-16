import time
"""
An implementation of tic-toc for timing performance.

Usage:

Example (1):
    tic()
    time.sleep(5)
    toc() # returns "Elapsed time: 5.00 seconds."

Example (2):
    

Source:
    https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
"""
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)