/*
 * PointOfInterest.java
 *
 * Created on 1 de julio de 2008, 12:36
 *
 * Represents a point of interest of the map
 */

package sim;

import nodes.Node;

/**
 *
 * @author Manuel Fogué Cortés
 */
public class PointOfInterest {
    
    //private double x;
    //private double y;
    
    private double minX;
    private double minY;
    private double maxX;
    private double maxY;
    
    private Simulation sim;
    
    /*public PointOfInterest(double x, double y) {
        this.x = x;
        this.y = y;
    }
    
    public PointOfInterest(double x, double y, Simulation sim) {
        this.sim = sim;
        minX = x - sim.STREET_DIST*2;
        if (minX < 0) minX = 0;
        maxX = x + sim.STREET_DIST*2;
        if (maxX > sim.MAX_X) maxX = sim.MAX_X;
        minY = y - sim.STREET_DIST*2;
        if (minY < 0) minY = 0;
        maxY = y + sim.STREET_DIST*2;
        if (maxY > sim.MAX_Y) maxY = sim.MAX_Y;
    }*/
    
    /**
     * Creates a new instance of PointOfInterest
     */
    public PointOfInterest(double minX, double minY, double maxX, double maxY) {
        this.minX = minX;
        this.minY = minY;
        this.maxX = maxX;
        this.maxY = maxY;
    }
    
    /*public void setSim(Simulation sim) {
        this.sim = sim;
        minX = x - sim.STREET_DIST*2;
        if (minX < 0) minX = 0;
        maxX = x + sim.STREET_DIST*2;
        if (maxX > sim.MAX_X) maxX = sim.MAX_X;
        minY = y - sim.STREET_DIST*2;
        if (minY < 0) minY = 0;
        maxY = y + sim.STREET_DIST*2;
        if (maxY > sim.MAX_Y) maxY = sim.MAX_Y;
    }*/
    
    public boolean isIn(Node n) {
        double x = n.getCurrentPosition().getX();
        double y = n.getCurrentPosition().getY();
        return (x <= this.maxX && x >= this.minX && y <= this.maxY && y >= this.minY);
    }

    public double getMinX() {
        return minX;
    }

    public double getMinY() {
        return minY;
    }

    public double getMaxX() {
        return maxX;
    }

    public double getMaxY() {
        return maxY;
    }
    
    public boolean equals(Object o) {
        if (o != null && (o instanceof PointOfInterest)) {
            PointOfInterest poi = (PointOfInterest) o;
            return this.minX == poi.minX && this.minY == poi.minY &&
                    this.maxX == poi.maxX && this.maxY == poi.maxY;
        } else {
            return false;
        }
    }
    
    public String toString() {
        return "POI: (" + minX + "," + minY + "; " + maxX + "," + maxY + ")";
    }

    
}
