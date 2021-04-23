/*
 * Simulation.java
 *
 * Created on 30 de junio de 2008, 12:20
 *
 * Represents a simulation, which includes all parameters and nodes
 */

package sim;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import config.MyConfig;
import nodes.Node;
import nodes.Position;

/**
 *
 * @author Manuel Fogué Cortés
 */
public class Simulation {

	// The simulator will print the nodes' locations once in intervalBetweenLocationUpdates. 
	private double intervalBetweenLocationUpdates = 1;

	public String outputFileName = "citymob.res";

    /*
     * Random numbers generator
     */
    private Random random;
    
    /*
     * Seed for random numbers
     */
    private long seed;
    
    /*
     * Points of interest
     */
    public ArrayList<PointOfInterest> pois = new ArrayList<PointOfInterest>();
    
    /*
     * Vertical streets
     */
    public HashMap<Double, Street> vStreets = new HashMap<Double, Street>();
    
    /*
     * Horizontal streets
     */
    public HashMap<Double, Street> hStreets = new HashMap<Double, Street>();
    
    /*
     * Array of nodes
     */
    public Node[] nodeSet;
    
    /*
     * Current simulation time
     */
    private double currentTime = 0;
    
    /*
     * Destination of the simulation output (default: standard output)
     */
    private PrintWriter out = new PrintWriter(System.out);
        
    
    /**
     * SIMULATION PARAMETERS (initialized at their default value)
     */
    
    /*
     * Number of nodes
     */
    public int NODES = 2;    
    
    /*
     * Width of map
     */
    public double MAX_X = 100;
    
    /*
     * Height of map
     */
    public double MAX_Y = 100;
    
    /*
     * Maximum simulation time
     */
    public double MAX_TIME = 2;
    
    /*
     * Distance between streets
     */
    public double STREET_DIST = 20;
    
    /*
     * Lanes per street
     */
    public int LANES = 1;
    
    /*
     * Number of accidents
     */
    public int ACCIDENTS = 5;    
    
    /*
     * Probability of visiting a point of interest
     */
    public double ALPHA = 0.5; 
    
    /*
     * Minimum distance between nodes
     */
    public double DELTA = 5;
    
    /*
     * Probability of semaphore pause
     */
    public double SEMAPHORE_PROB = 0.3;
    
    /*
     * Maximum semaphore pause duration
     */
    public double SEMAPHORE_PAUSE = 20;
    
    /*
     * Maximum speed in downtown (Km/h)
     */
    public double MAX_SPEED_DOWN = 50;
    
    /*
     * Minimum speed in downtown (Km/h)
     */
    public double MIN_SPEED_DOWN = 25;
    
    /*
     * Maximum speed (Km/h)
     */
    public double MAX_SPEED = 100;
    
    /*
     * Minimum speed (Km/h)
     */
    public double MIN_SPEED = 50;
    
    /*
     * Time interval between updates
     */
    public double TIME_INTERVAL = 1;
    
    
    /** Creates a new instance of Simulation */
    public Simulation() {
        
        seed = System.currentTimeMillis();
        random = new Random(seed);
        
    }
    
    /*
     * Iniatialize simulation with current parameters
     */
    public void initializeSimulation() {
        nodeSet = new Node[NODES];
                
        // Vertical streets creation
        for (int i = 1; i < (MAX_X/STREET_DIST); i++) {
            VStreet str = new VStreet(this, i*STREET_DIST, LANES);
            vStreets.put(new Double(i*STREET_DIST), str);
        }
        
        // Horizontal streets creation
        for (int i = 1; i < (MAX_Y/STREET_DIST); i++) {
            HStreet str = new HStreet(this, i*STREET_DIST, LANES);
            hStreets.put(new Double(i*STREET_DIST), str);
        }
    }
    
    /*
     * Determines the initial situation for the nodes
     */
    public void generateStaticScenario() throws ScenarioCreationException {
        
        println("#\n# MODEL 4: Enhanced Downtown Traffic Simulation Model\n#");
        println("#\n#\t SEED = " + seed + "\n#");

        for (int i = 0; i < NODES; i++) {
            Node node = null;
            
            // Node in downtown
            if (pois.size() > 0 && random(100) < ALPHA*100) {
                int nPoi = (int) random(pois.size());
                PointOfInterest poi = pois.get(nPoi);
                node = new Node(i, this, poi);
                
            // Node not in downtown
            } else {
                node = new Node(i, this);
            }
            
            node.setRandomInitialPosition();
                        
        }
    }
    
    /*
     * Starts simulation with current parameters
     */
    public void startSimulation() {
                
        println("#\n#-- END OF INITIAL POSITION CONFIGURATION --");
        println("#\n# Movements:\n#");
        
        // Initial Accidents
        for (int i = 0; i < NODES; i++) {
            Node node = nodeSet[i];
            if (i < ACCIDENTS) {
                node.accident();
            }
        }
        
        // Node movements
        currentTime = 0;
        while (getCurrentTime() < MAX_TIME) {
            currentTime += TIME_INTERVAL;
             
            updateNodesPositions();
            if (currentTime % intervalBetweenLocationUpdates == 0) {
            	MyConfig.writeStringToFile (outputFileName, String.format("time = %.0f\n", currentTime));
                for (int i = 0; i < NODES; i++) {
                    nodeSet[i].printNodePos(outputFileName);
                }
            }
            
            checkNewEvents();
        }
                       
        println("#\n#-- EXITING PROGRAM --#");
        if (out != null) {
            out.close();
        }
    }
    
    /*
     * Updates the position of all nodes
     */
    public void updateNodesPositions() {
        
        for (int i = 0; i < NODES; i++) {
            Node node = nodeSet[i];
            node.updatePosition(TIME_INTERVAL);
        }
    }
    
    /*
     * Checks events for all nodes
     */
    public void checkNewEvents() {
        
        for (int i = 0; i < NODES; i++) {
            Node node = nodeSet[i];
            node.checkEvents();
        }
    }
    
    /*
     * Updates the street where a node is set
     */
    public void setStreet(Node node) {
        if (node.getDirection() == Node.UP || node.getDirection() == Node.DOWN) {
            double nStreet = Math.round(node.getCurrentPosition().getX() / STREET_DIST) * STREET_DIST;
            Street str = vStreets.get(new Double(nStreet));
            str.add(node);
            node.setStreet(str);
        } else {
            double nStreet = Math.round(node.getCurrentPosition().getY() / STREET_DIST) * STREET_DIST;
            Street str = hStreets.get(new Double(nStreet));
            str.add(node);
            node.setStreet(str);
        }
    }
    
    /*
     * Returns the vertical street at a given position
     */
    public Street getVStreet(double x) {
        double nStreet = Math.round(x / STREET_DIST) * STREET_DIST;
        return vStreets.get(new Double(nStreet));
    }
    
    /*
     * Returns the horizontal street at a given position
     */
    public Street getHStreet(double y) {
        double nStreet = Math.round(y / STREET_DIST) * STREET_DIST;
        return hStreets.get(new Double(nStreet));
    }
    
    /*
     * Determines if a node is near a point of interest (in downtown)
     */
    public boolean isInDowntown(Node node) {
        Iterator it = pois.iterator();
        while (it.hasNext()) {
            PointOfInterest poi = (PointOfInterest) it.next();
            if (poi.isIn(node)) {
                return true;
            } 
        }
        return false;
    }
        
    /*
     * Adds a new node to the simulation
     */
    public void addNode(Node node) {
        nodeSet[node.getIndex()] = node;
    }
    
    /*
     * Determines if a node is colliding with another in the simulation
     */
    public boolean collision(Node n) {
        for (int i = 0; i < NODES; i++) {
            Node node = nodeSet[i];
            if (node != null && node.distanceTo(n) < DELTA) {
                if (node.getDirection() == n.getDirection() && node.getLane() == n.getLane()) {
                    return true;
                }
            }
        }
        return false;
    }
    
    public double street(double number) {
        int div = (int) (number / this.STREET_DIST);
        double sol = (int) div * this.STREET_DIST;
                
        if (sol >= this.MAX_X || sol >= this.MAX_Y)
            return sol - this.STREET_DIST;
        else if (sol == 0)
            return this.STREET_DIST;
        else
            return sol;
            
    }
    
    public double random() {
        return random.nextDouble();
    }
    
    public double random(double limit) {
        double value = random.nextDouble();
        double sol = value*limit;
        while (sol > limit) {
            value = random.nextDouble();
            sol = value*limit;
        }
        if (sol < 0) {
            sol = -sol;
        }
        return sol;
    }

    public double random(double limit1, double limit2) {
        if (limit1 > limit2) {
            double aux = limit1;
            limit1 = limit2;
            limit2 = aux;
        }
        double value = random.nextDouble();
        double sol = limit1 + value*(limit2 - limit1);
        while (sol > limit2 || sol < limit1) {
            value = random.nextDouble();
            sol = limit1 + value*(limit2 - limit1);
        }
        if (sol < 0) {
            sol = -sol;
        }
        return sol;
    }
    
    public long getSeed() {
        return seed;
    }    
    
    public String show() {
        try {
            int h = 60;
            int w = 60;
            char[][] matrix = new char[h][w];
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    matrix[i][j] = ' ';
                }
            }
            Iterator it = pois.iterator();
            while (it.hasNext()) {
                PointOfInterest poi = (PointOfInterest) it.next();
                int xMin = (int) (poi.getMinX()*w/MAX_X);
                int yMin = (int) (poi.getMinY()*h/MAX_Y);
                int xMax = (int) (poi.getMaxX()*w/MAX_X);
                int yMax = (int) (poi.getMaxY()*h/MAX_Y);
                for (int i = xMin; i < xMax; i++) {
                    for (int j = yMin; j < yMax; j++) {
                        matrix[i][j] = '·';
                    }
                }
            }
            for (int i = 0; i < NODES; i++) {
                Position pos = nodeSet[i].getCurrentPosition();
                int x = (int) (pos.getX()*w/MAX_X);
                int y = (int) (pos.getY()*h/MAX_Y);
                matrix[x][y] = 'X';
            }
            String res = "";
            for (int i = 0; i < h; i++) {
                res += new String(matrix[i]) + "\n";
            }
            return res;
        } catch (Exception e) {
            //System.out.println("Error");
            return null;
            
        }
    }

    /*
     * Return current simulation time
     */
    public double getCurrentTime() {
        return (Math.round(currentTime * 100)) / 100.0;
    }

    /*
     * Sets a new destination for the output
     */
    public void setOut(PrintWriter out) {
        this.out = out;
    }

    /*
     * Writes a String in the output
     */
    public void println(String str) {
        out.println(str);
    }
    
}
