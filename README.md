## Service Function Chains Migration

This project provides simulator tools to test scenarios and algorithms for the migration of Service Function Chains (SFC). 

# Mobility simulator

The file mobility_simulator.py implements a simple [random waypoint simulator](https://en.wikipedia.org/wiki/Random_waypoint_model). 
The simulator simulates users moving within a large square, that is covered by n X n Access Points (APs).
Each AP covers an area of 1/(n*n) of the large square.

# Mobility simulator output files
By default, output files are written to a sibling directory named "../res". If you don't have, please generate such a directory. 

Whenever a user migrates (namely, crosses to an area covered by another AP), the simulator prints to a file the time, and the AP ID of each user. 
The extension of this output file is ".ap" 

In addition, the simulator may periodically outputs to the ".ap" file the current AP of all users.

The simulator may also generate a file with the extension ".loc" (for "location"), to which it writes the locations (as (X,Y) coordinates) of all users upon every event, and/or periodically.

Further documentation can be found within the code files.

