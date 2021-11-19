import os, sys
from contextlib import suppress
import math

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
import sumolib

# %%
# sumoCmd
sumoCmd = [r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo",  # sumo binary
           "-c",
           r"C:\PhD_Files\scenarios\LuSTScenario-master\scenario\due.actuated.sumocfg",  # sumo config file
           "--start",
           "--quit-on-end"]
traci.start(sumoCmd)

net = sumolib.net.readNet(r"C:\PhD_Files\scenarios\LuSTScenario-master\scenario\lust.net.xml")  # net file
# mention 4 corners positions of the area of interest as a rectangle.
# Position in this order Top Left, Top Right, Bottom Right, Bottom Left, Top Left
ROI = Polygon([(5089.78, 6805.97), (5779.82, 6811.18), (5793.73, 6494.84), (5103.68, 6439.22), (5088.04, 6805.97)])
edges = traci.edge.getIDList()
edgesOfInterest = []
edgesInfo = {}
# total length of lanes under the region of interest
totalLength = 0
for edge in edges:
    # if edge ID starts with : then its a junction according to SUMO docs.
    if edge[0] == ":":
        # avoiding the junctions
        continue
    curEdge = net.getEdge(edge)
    # get bounding box of the edge
    curEdgeBBCoords = curEdge.getBoundingBox()
    # create the bounding box geometrically
    curEdgeBBox = box(*curEdgeBBCoords)
    # check if the edge is inside the region of interest
    isInside = ROI.intersects(curEdgeBBox) or ROI.contains(curEdgeBBox)
    if isInside:
        # store the valid edges in a list
        edgesOfInterest.append(edge)
        # store valid edge informations in a dictionary
        edgesInfo[edge] = {}
        edgesInfo[edge]["length"] = curEdge.getLength()
        # total length based on edge
        totalLength = totalLength + edgesInfo[edge]["length"]
        # edge length will be the same as lane length just to check additional info about multiple lanes in the edge
        lanes = curEdge.getLanes()
        edgesInfo[edge]["Lanes"] = {}
        for lane in lanes:
            laneIndex = lane.getIndex()
            edgesInfo[edge]["Lanes"][f"Lane{laneIndex}"] = {}
            edgesInfo[edge]["Lanes"][f"Lane{laneIndex}"]["length"] = lane.getLength()
