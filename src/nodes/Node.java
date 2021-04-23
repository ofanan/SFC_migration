/*
 * Node.java
 *
 * Created on 30 de junio de 2008, 12:19
 *
 * Represents a node of the simulation, moving through the streets and interacting
 * with other nodes
 */

package nodes;

import java.util.ArrayList;

import config.MyConfig;
import sim.PointOfInterest;
import sim.ScenarioCreationException;
import sim.Simulation;
import sim.Street;

/**
 *
 * @author Manuel Fogué Cortés
 */
public class Node {
    
    /* Direction constants */
    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;
    
    /* Special constants */
    public static final double UNDEFINED = -1;
    public static final int NO_LANE = -1;
    
    /* 
     * Index of current node
     */
    private int index;
    
    /*
     * Current and target positions
     */
    private Position currentPosition = new Position(0,0);
    private Position targetPosition = new Position(0,0);
    
    /*
     * Time to reach current target
     */
    private double timeToTarget = UNDEFINED;
    
    /*
     * Next junction near this node (distance < DELTA)
     */
    private Street junctionStreet = null;
    private Position junctionPosition = new Position(0,0);
    
    /*
     * Current speed (Km/h)
     */
    private double speed;
    private double formerSpeed;
    
    /*
     * Current direction (UP, DOWN, LEFT, RIGHT)
     */
    private int direction;
    
    /*
     * Simulation this node belongs to
     */
    private Simulation sim;
    
    /*
     * Street (and lane int the street) where the node is moving
     */
    private Street street;
    private int lane;
    
    /*
     * Time to be stopped at a semaphore
     */
    private double pauseTime;
    
    /*
     * Indicates whether this node is stopped at a semaphore or a junction
     */
    private boolean stoppedAtSemaphore = false;
    private boolean stoppedAtJunction = false;
    
    /*
     * Indicates whether this node is accidented
     */
    private boolean accident = false;
    
    /*
     * Indicates whether this node is in downtown (indeed, visiting an interest point, 
     * the name is the same to mantain coherency with previous versions)
     */
    private boolean inDowntown;
    private PointOfInterest poi = null;
    
    /*
     * In order to save its movements
     */
    private ArrayList<String> movements = new ArrayList<String>();
    
    /*
     * Reason for a speed change (used as information for the user)
     */
    //private String speedChangeReason;
    
        
    /** Creates a new instance of Node */
    public Node(int index, Simulation sim) {
        this.index = index;
        this.sim = sim;
        this.inDowntown = false;    // Not in downtown
    }
    
    public Node(int index, Simulation sim, PointOfInterest poi) {
        this.index = index;
        this.sim = sim;
        this.inDowntown = true;     // In downtown
        this.poi = poi;
    }
        
    /*
     * Determines if two nodes are the same
     */
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null)
            return false;
        if (!(o instanceof Node))
            return false;
        Node other = (Node) o;
        return this.index == other.index;          
    }
    
        
    /*
     * Prints current movement at current speed (in ns-2 format)
     */
    public void printCurrentMovement() {
        movements.add("$ns_ at " + sim.getCurrentTime() + " \"$node_(" + index + ") setdest " + 
                currentPosition.getX() + " " + currentPosition.getY() + " " + this.speed + "\"");
    }
    
    /*
     * Prints current movement at current speed (in ns-2 format)
     */
    public void printCurrentMovement(double formerSpeed) {
        movements.add("$ns_ at " + sim.getCurrentTime() + " \"$node_(" + index + ") setdest " + 
                currentPosition.getX() + " " + currentPosition.getY() + " " + formerSpeed + "\"");
    }
    
    /*
     * Prints target movement at current speed (in ns-2 format)
     */
    public void printTargetMovement() {
        movements.add("#Node " + index + ": TARGET REACHED " + this.targetPosition + " (" + speed + " Km/h)");
        if (timeToTarget != UNDEFINED) {
            movements.add("$ns_ at " + timeToTarget + " \"$node_(" + index + ") setdest " + 
                    targetPosition.getX() + " " + targetPosition.getY() + " " + speed + "\"");
        } else {
            movements.add("$ns_ at " + sim.getCurrentTime() + " \"$node_(" + index + ") setdest " + 
                    targetPosition.getX() + " " + targetPosition.getY() + " " + speed + "\"");            
        }
    }
    
    /*
     * Print all movements of the node
     */
    public void printMovements() {
        sim.println("##### NODE " + index + " MOVEMENTS #####");
        if (!accident) {
            printCurrentMovement();
        }
        for (String mov: movements) {
            sim.println(mov);
        }
    }
    
    
    /*
     * Prints node position 
     */
    public void printNodePos (String outputFileName) {
        MyConfig.writeStringToFile (outputFileName, String.format("node %d %.1f %.1f\n", index, currentPosition.getX(), currentPosition.getX()));
    }
    
    
    /*
     *  Returns a vector [X, Y], where X and Y are the current X,Y positions of this node 
     */
    public double getNodePos () {
    	return currentPosition.getX();
    }
    
    public String toString() {
        return "Node " + index + ": " + currentPosition + " --> " + targetPosition + " " + this.speed + " Km/h " +  this.timeToTarget + " s " + this.street + " Lane: " + this.lane;
    }
     
    /*
     * Return the node in front of this one (in the same street)
     */
    public Node getFrontNode() {
        return street.nodeInFront(this);
    }
    
    /*
     * Return the node behind this one (in the same street)
     */
    public Node getBackNode() {
        return street.nodeBehind(this);
    }
    
    /*
     * Checks if the current situation of the node needs a speed change to be
     * coherent
     */
    public void checkSpeedChanges() {
        if (!accident) {
            // Check conflicts with front node
            Node frontNode = getFrontNode();
            checkSpeedChanges(frontNode);
        }
        
    }
        
    /*
     * Checks if there is any conflict with the node in front of this one, or if
     * this node can continue moving without collisions with nearby nodes.
     * There is conflict when the front node is near and this node is moving faster
     * than the front node.
     */
    public void checkSpeedChanges(Node frontNode) {
        
        if (!accident) {
        
            // Check possible collision
            if (frontNode != null && distanceTo(frontNode) <= sim.DELTA && this.speed > frontNode.speed) {

                // Collision!!

                // Look for a new lane where the node could mantain its speed
                int newLane = findNewLane();

                if (newLane != NO_LANE) {   // New lane found
                    this.lane = newLane;
                } else {
                    // Adjust speed to front node speed
                    //this.speedChangeReason = "Front node (" + frontNode.index + 
                    //        ") too slow";
                    setSpeed(frontNode.speed);
                }

            // Restart movement after a stop (not stopped at a semaphore or a junction)
            } else if (!stoppedAtSemaphore && !stoppedAtJunction && speed == 0) {

                // Front node is far enough to avoid collision
                if (frontNode == null || distanceTo(frontNode) > sim.DELTA) {

                    setRandomSpeed();
                    //System.out.println(sim.getCurrentTime() + ": *** Node " + index + " " + currentPosition + " has restarted moving " + getSDirection() + " to " + targetPosition + " " + this.speed + " Km/h");

                // Front node is near
                } else {

                    // Look for a new lane where the node could move
                    int newLane = findNewLane();
                    
                    if (newLane != NO_LANE) {   // New lane found
                        this.lane = newLane;
                        setRandomSpeed();
                        
                    }  else {
                        // Adjust speed to front node speed
                        setSpeed(frontNode.speed);
                    }
                    
                    //this.speedChangeReason = "Moving again after a stop";

                }

            }

            // Checks if the back node is moving too fast and near
            Node backNode = getBackNode();
            if (backNode != null) {
                backNode.checkSpeedChanges(this);
            }
        
        }
    }
    
    /*
     * Looks for a new lane (different from current one) where the node could keep
     * moving without colliding with other nodes
     */
    private int findNewLane() {
        
        if (sim.LANES > 1) {        
            // Look for a new lane where the node could restart moving
            boolean foundNewLane = false;
            int newLane = NO_LANE;

            for (int possibleLane = 0; possibleLane < sim.LANES && !foundNewLane; possibleLane++) {

                if (possibleLane != this.lane && !street.collision(this, direction, possibleLane)) {
                    newLane = possibleLane;
                    foundNewLane = true;
                }
            }

            return newLane;
        } else {
            return NO_LANE;
        }
        
    }
            
    /*
     * The node has had an accident
     */
    public void accident() {
        
        targetPosition = currentPosition;
        this.speed = 0;
        this.accident = true;
        movements.add("#Node " + index + " had an ACCIDENT:");
        printCurrentMovement(0);
        movements.add("#END ACCIDENT");
        
    }
    
    /*
     * The node is stopped at a semaphore until 'pauseTime'
     */
    public void semaphore() {
        this.stoppedAtSemaphore = true;
        this.speed = 0;
        updateTimeToTarget();
        
        // Random time stopped at the semaphore
        double pause = sim.random(sim.SEMAPHORE_PAUSE);
        this.pauseTime = sim.getCurrentTime() + pause;   
        movements.add("#Node " + index + ": RED SEMAPHORE UNTIL " + pauseTime);
             
    }
        
    /*
     * Checks any event that could happen due to the current situation of the node
     * (junction reached, target reached, green semaphore...)
     */
    public void checkEvents() {
                
        if (!accident) {
            
            boolean targetReached = false;
            
            if (stoppedAtSemaphore) {    // Node stopped at a semaphore

                if (sim.getCurrentTime() >= pauseTime) {   // Green semaphore
                    
                    movements.add("#Node " + index + ": GREEN SEMAPHORE");
                    
                    stoppedAtSemaphore = false;
                    setRandomTarget();  // New target for the node 
                    //System.out.println(sim.getCurrentTime() + ": *** Node " + index + " " + currentPosition + " is moving again to " + targetPosition + " " + stoppedAtSemaphore);
                                        
                    // Checks possible speed changes in the new street
                    checkSpeedChanges(); 
                    
                }
                
            } else if (stoppedAtJunction) { // Node stopped at junction
                
                // Checks if junction is now safe
                checkJunction();

            } else {

                // Checks if the node has reached its target
                if (timeToTarget != UNDEFINED && timeToTarget <= sim.getCurrentTime()) {
                    
                    // Target reached
                    targetReached = true;
                    printTargetMovement();
                    setCurrentPosition(this.targetPosition);
                    
                    if (sim.random(100) < sim.SEMAPHORE_PROB*100) { // Node stopped at a semaphore
                        semaphore();
                        //System.out.println(sim.getCurrentTime() + ": - Node " + index + " " + currentPosition + " " + getSDirection() + " is stopped at a semaphore until " + pauseTime + " s " + stoppedAtSemaphore);
                    
                    } else {
                        //System.out.println(sim.getCurrentTime() + ": /// Node " + index + " " + currentPosition + " " + getSDirection() + " has reached its destiny " + targetPosition + " " + stoppedAtSemaphore + " " + this.timeToTarget + " s");
                        setRandomTarget();
                        //System.out.println(sim.getCurrentTime() + ": Node " + index + " new destiny: " + targetPosition + " " + getSDirection() + " " + speed + " Km/h " + this.timeToTarget + " s " + stoppedAtSemaphore);
                    }
                    
                // Checks if the node has reached a junction
                } else if (nearJunction()) {
                    
                    // Checks if junction is safe
                    checkJunction();
                    
                }
                             
                // Checks possible speed changes
                checkSpeedChanges();
            }

            if (!targetReached && this.formerSpeed != this.speed) {
                printCurrentMovement(this.formerSpeed);
                movements.add("#Node " + index + ": SPEED CHANGE (" + this.formerSpeed + 
                        " Km/h --> " + this.speed + " Km/h)");
            }
            
            formerSpeed = speed;
            
        }
        
    }
    
    
    /*
     * Indicates whether this node is near a junction
     */
    private boolean nearJunction() {
        double streetPos = 0;
        Position junction = null;
        Street street = null;

        if (direction == UP) {
            streetPos = ((int)(getY() / sim.STREET_DIST)) * sim.STREET_DIST;
            if (streetPos == getY()) {
                streetPos -= sim.STREET_DIST;
            }
            if (streetPos > 0 && streetPos < sim.MAX_Y) {
                street = sim.getHStreet(streetPos);
                junction = new Position(getX(), streetPos);
            }

        } else if (direction == DOWN) {
            streetPos = ((int)((getY() + sim.STREET_DIST) / sim.STREET_DIST)) * sim.STREET_DIST;
            if (streetPos > 0 && streetPos < sim.MAX_Y) {
                street = sim.getHStreet(streetPos);
                junction = new Position(getX(), streetPos);
            }

        } else if (direction == LEFT) {
            streetPos = ((int)(getX() / sim.STREET_DIST)) * sim.STREET_DIST;
            if (streetPos == getX()) {
                streetPos -= sim.STREET_DIST;
            }
            if (streetPos > 0 && streetPos < sim.MAX_X) {
                street = sim.getVStreet(streetPos);
                junction = new Position(streetPos, getY());
            }

        } else if (direction == RIGHT) {
            streetPos = ((int)((getX() + sim.STREET_DIST) / sim.STREET_DIST)) * sim.STREET_DIST;
            if (streetPos > 0 && streetPos < sim.MAX_X) {
                street = sim.getVStreet(streetPos);
                junction = new Position(streetPos, getY());
            }
        }

        // Junction found and near the node (distance < DELTA)
        if (junction != null && distanceTo(junction) < distanceTo(targetPosition) && 
                distanceTo(junction) < sim.DELTA) {
            this.junctionStreet = street;
            this.junctionPosition = junction;
            return true;
        } else {
            this.junctionStreet = null;
            this.junctionPosition = null;
            return false;
        }
    }
    
    /*
     * Checks if there could be collisions with the nodes crossing the junction
     */
    private void checkJunction() {
                
        if (direction == Node.UP) {
            if (junctionStreet.collision(this, Node.RIGHT) || junctionStreet.collision(this, Node.LEFT)) {
                stoppedAtJunction = true;
                setSpeed(0);    // Stops to avoid collision with crossing nodes
                //this.speedChangeReason = "Stopped at junction";
                //System.out.println(sim.getCurrentTime() + ": _-_- Node " + index + " " + currentPosition + " " + getSDirection() + " is stopped at junction " + junctionPosition + " " + speed + " m/s");
            } else {
                if (speed == 0) {  // Restart movement after a stop
                    stoppedAtJunction = false;
                    setRandomSpeed();
                    //this.speedChangeReason = "Moving again after a stop at a junction";
                    checkSpeedChanges();
                    //System.out.println(sim.getCurrentTime() + ": .... Node " + index + " " + currentPosition + " " + getSDirection() + " is near junction " + junctionPosition + " moving again " + speed + " m/s");
                }
            }
        } else if (direction == Node.LEFT || direction == Node.RIGHT) {
            if (junctionStreet.collision(this, Node.DOWN)) {
                stoppedAtJunction = true;
                setSpeed(0);    // Stops to avoid collision with crossing nodes
                //this.speedChangeReason = "Stopped at junction";
                //System.out.println(sim.getCurrentTime() + ": _-_- Node " + index + " " + currentPosition + " " + getSDirection() + " is stopped at junction " + junctionPosition + " " + speed + " m/s");
            } else {
                if (speed == 0) {   // Restart movement after a stop
                    stoppedAtJunction = false;
                    setRandomSpeed();
                    //this.speedChangeReason = "Moving again after a stop at a junction";
                    checkSpeedChanges();
                    //System.out.println(sim.getCurrentTime() + ": .... Node " + index + " " + currentPosition + " " + getSDirection() + " is near junction " + junctionPosition + " moving again " + speed + " m/s");
                }
            }
        }
    }
    
    /*
     * Updates the position of a node considering its speed and direction
     */
    public void updatePosition(double timeInterval) {
        
        // Update position of the node
        if (speed > 0) {
            
            double speedInM_S = speed*1000/3600;

            if (direction == UP) {
                currentPosition.setY(currentPosition.getY() - speedInM_S*timeInterval);
            } else if (direction == DOWN) {
                currentPosition.setY(currentPosition.getY() + speedInM_S*timeInterval);
            } else if (direction == LEFT) {
                currentPosition.setX(currentPosition.getX() - speedInM_S*timeInterval);
            } else if (direction == RIGHT) {
                currentPosition.setX(currentPosition.getX() + speedInM_S*timeInterval);
            }
        }

    }
        
    /*
     * Sets a random speed for the node
     */
    public void setRandomSpeed() {
        if (!inDowntown) {
            setSpeed(sim.random(sim.MIN_SPEED, sim.MAX_SPEED));
        } else {
            setSpeed(sim.random(sim.MIN_SPEED_DOWN, sim.MAX_SPEED_DOWN));
        }
    }
    
    /*
     * Sets the initial position for the node
     */
    public void setRandomInitialPosition() throws ScenarioCreationException {
        int tries = 0, maxTries = 100;  // Max number of random tries
        boolean found = false;
        do {
            if (inDowntown) {
                setRandomPositionDowntown();
            } else {
                setRandomPositionNotDowntown();
            }
            this.lane = (int) sim.random(sim.LANES);
            
            tries++;
            found = !sim.collision(this);
        } while (!found && tries < maxTries);
                
        if (!found) {
            
            // Recover from infinite loop
            throw new ScenarioCreationException(this.index);
            
        } else {
            
            formerSpeed = speed;
            sim.setStreet(this);            
            sim.addNode(this);
            
            //updateNextJunction();
            
            // Update Time to reach target
            updateTimeToTarget();
            
            // Clear outputting of node position
//            printNodePos();
        }
        
    }
    
    public void setRandomPositionDowntown() {
        direction = (int) sim.random(4);
        if (direction == UP || direction == DOWN) {
            double x = sim.street(sim.random(poi.getMinX(), poi.getMaxX()));
            double y = sim.random(poi.getMinY(), poi.getMaxY());
            currentPosition = new Position(x, y);
            targetPosition = new Position(x, sim.street(sim.random(poi.getMinY(), poi.getMaxY())));
            if (targetPosition.getY() < currentPosition.getY()) {
                direction = UP;
            } else {
                direction = DOWN;
            }
        } else {
            double x = sim.random(poi.getMinX(), poi.getMaxY());
            double y = sim.street(sim.random(poi.getMinY(), poi.getMaxY()));
            currentPosition = new Position(x, y);
            targetPosition = new Position(sim.street(sim.random(poi.getMinX(), poi.getMaxX())), y);
            if (targetPosition.getX() < currentPosition.getX()) {
                direction = LEFT;
            } else {
                direction = RIGHT;
            }
        }
        
        speed = sim.random(sim.MIN_SPEED_DOWN, sim.MAX_SPEED_DOWN);
            
    }
    
    public void setRandomPositionNotDowntown() {
        direction = (int) sim.random(4);
        if (direction == UP || direction == DOWN) { // Vertical direction
            double x = sim.street(sim.random(sim.MAX_X));
            double y = sim.random(sim.MAX_Y);
            currentPosition = new Position(x, y);
            targetPosition = new Position(x, sim.street(sim.random(sim.MAX_Y)));
            if (targetPosition.getY() < currentPosition.getY()) {
                direction = UP;
            } else {
                direction = DOWN;
            }
        } else {                                    // Horizontal direction
            double x = sim.random(sim.MAX_X);
            double y = sim.street(sim.random(sim.MAX_Y));
            currentPosition = new Position(x, y);
            targetPosition = new Position(sim.street(sim.random(sim.MAX_X)), y);
            if (targetPosition.getX() < currentPosition.getX()) {
                direction = LEFT;
            } else {
                direction = RIGHT;
            }
        }
        
        speed = sim.random(sim.MIN_SPEED, sim.MAX_SPEED);
            
    }
          
    
    /*
     * Sets a new target for a node
     */
    public void setRandomTarget() {
        
        // Change the street where the node is moving
        this.street.remove(this);
        
        if (inDowntown) {
            setRandomTargetDowntown();
        } else {
            setRandomTargetNotDowntown();            
        }
        //this.speedChangeReason = "New target " + this.speed + " Km/h";
        
        this.lane = (int) sim.random(sim.LANES);
        
        // Update street where the node is
        sim.setStreet(this);
        
        // Check for conflicts in the new street
        checkSpeedChanges();
        
        movements.add("#Node " + index + ": NEW TARGET " + this.targetPosition);
    }
    
    public void setRandomTargetDowntown() {
        if (direction == UP || direction == DOWN) {                                
            // Set new target
            double lastX = targetPosition.getX();
            this.targetPosition.setX(sim.street(sim.random(poi.getMinX(), poi.getMaxX())));
            while (targetPosition.getX() == lastX) {
                this.targetPosition.setX(sim.street(sim.random(poi.getMinX(), poi.getMaxX())));
            }
            if (targetPosition.getX() < currentPosition.getX()) {
                direction = LEFT;
            } else {
                direction = RIGHT;
            }
        } else {
            double lastY = targetPosition.getY();
            this.targetPosition.setY(sim.street(sim.random(poi.getMinY(), poi.getMaxY())));
            while (targetPosition.getY() == lastY) {
                this.targetPosition.setY(sim.street(sim.random(poi.getMinY(), poi.getMaxY())));
            }
            if (targetPosition.getY() < currentPosition.getY()) {
                direction = UP;
            } else {
                direction = DOWN;
            }
        }

        // Set new speed
        setSpeed(sim.random(sim.MIN_SPEED_DOWN, sim.MAX_SPEED_DOWN));
        
    }
    
    public void setRandomTargetNotDowntown() {
        if (direction == UP || direction == DOWN) {                                
            // Set new target
            double lastX = targetPosition.getX();
            this.targetPosition.setX(sim.street(sim.random(sim.MAX_X)));
            while (targetPosition.getX() == lastX) {
                this.targetPosition.setX(sim.street(sim.random(sim.MAX_X)));
            }
            if (targetPosition.getX() < currentPosition.getX()) {
                direction = LEFT;
            } else {
                direction = RIGHT;
            }
        } else {
            double lastY = targetPosition.getY();
            this.targetPosition.setY(sim.street(sim.random(sim.MAX_Y)));
            while (targetPosition.getY() == lastY) {
                this.targetPosition.setY(sim.street(sim.random(sim.MAX_Y)));
            }
            if (targetPosition.getY() < currentPosition.getY()) {
                direction = UP;
            } else {
                direction = DOWN;
            }
        }

        // Set new speed
        setSpeed(sim.random(sim.MIN_SPEED, sim.MAX_SPEED));
    }
    
    /*
     * Sets a new speed for the node
     */
    public void setSpeed(double newSpeed) {
        
        double oldSpeed = this.speed;
        
        if (oldSpeed != newSpeed) {
            
            //printCurrentMovement();
            
            this.speed = newSpeed;        
            updateTimeToTarget();
            
        }
            
    }
    
    /*
     * Update time to reach current target
     */
    public void updateTimeToTarget() {
        if (speed == 0) {
            timeToTarget = UNDEFINED;
        } else {
            double speedInM_S = speed*1000/3600;
            timeToTarget = sim.getCurrentTime() + distanceTo(targetPosition) / speedInM_S; 
        }
    }
        
    /*
     * Distance to another node
     */
    public double distanceTo(Node n) {
        return this.currentPosition.distanceTo(n.currentPosition);
    }
    
    /*
     * Distance to a concrete position (x, y, z)
     */
    public double distanceTo(Position p) {
        return this.currentPosition.distanceTo(p);
    }
    
    public int getIndex() {
        return index;
    }

    public Position getCurrentPosition() {
        return currentPosition;
    }

    public void setCurrentPosition(Position currentPosition) {
        this.currentPosition.copy(currentPosition);
    }

    public Position getTargetPosition() {
        return targetPosition;
    }

    public void setTargetPosition(Position targetPosition) {
        this.targetPosition.copy(targetPosition);
    }

    public double getSpeed() {
        return speed;
    }

    public int getDirection() {
        return direction;
    }

    public String getSDirection() {
        switch (this.direction) {
            case UP: return "UP";
            case DOWN: return "DOWN";
            case RIGHT: return "RIGHT";
            default: return "LEFT";
        }
    }
    
    public void setDirection(int direction) {
        this.direction = direction;
    }

    public Street getStreet() {
        return street;
    }

    public void setStreet(Street str) {
        this.street = str;
    }
    
    public double getX() {
        return this.currentPosition.getX();
    }
    
    public double getY() {
        return this.currentPosition.getY();
    }
    
    public double getZ() {
        return this.currentPosition.getZ();
    }

    public boolean isSemaphore() {
        return stoppedAtSemaphore;
    }

    public void setSemaphore(boolean semaphore) {
        this.stoppedAtSemaphore = semaphore;
    }

    public int getLane() {
        return lane;
    }

    public void setLane(int lane) {
        this.lane = lane;
    }
}
