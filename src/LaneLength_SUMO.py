import traci, sumolib
from sumolib import checkBinary  
from shapely.geometry import Polygon, box

# Present the rectangle of interest as a Polygon
polygon = Polygon([(0, 10000), (10000, 10000), (10000, 0), (0,0)])  
traci.start([checkBinary('sumo'), '-c', 'myLuST.sumocfg', "--start", "--quit-on-end", '-W', '-V', 'false', '--no-step-log', 'true'])
net = sumolib.net.readNet(r'../../LuSTScenario/scenario/lust.net.xml')  # net file

totalLength = 0 # Will hold the total length of lanes within the polygon

for edge in traci.edge.getIDList():
    # if edge ID begins with ':' then its a junction according to SUMO docs.
    if edge[0] == ":": 
        continue # discard junctions
    
    curEdge = net.getEdge(edge)
    # get bounding box of the edge
    curEdgeBBCoords = curEdge.getBoundingBox()
    curEdgeBBox = box(*curEdgeBBCoords) # create the bounding box geometrically

    if polygon.contains(curEdgeBBox): # The given polygon contains that edge, so add the edge's length, multiplied by the # of lanes

        totalLength += curEdge.getLength() * len(curEdge.getLanes())
    
    # If polygon intersects with this edge then, as a rough estimation of the relevant length to add, divide the intersecting area by the total edge area
    elif (polygon.intersects(curEdgeBBox)):   
        
        totalLength += curEdge.getLength() * (polygon.intersection(curEdgeBBox).area / curEdgeBBox.area) 

traci.close()
print ('total length of lanes in the given rectangle is {}' .format (totalLength))        
