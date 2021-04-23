/*
 * VStreet.java
 *
 * Created on 4 de julio de 2008, 12:52
 *
 * Represents a vertical street
 */

package sim;

import java.util.ArrayList;
import java.util.List;
import nodes.Node;
import nodes.Position;

/**
 *
 * @author Manuel Fogué Cortés
 */
public class VStreet extends Street {
    
    private double x;
    
    /**
     * Creates a new instance of VStreet
     */
    public VStreet(Simulation sim, double x, int lanes) {
        super(sim, lanes);
        this.x = x;
    }
    
    public void add(Node node) {
        if (node.getDirection() == Node.DOWN) {
            nodesPos.add(node);
        } else if (node.getDirection() == Node.UP) {
            nodesNeg.add(node);
        }        
    }
    
    public void remove(Node node) {
        if (node.getDirection() == Node.DOWN) {
            nodesPos.remove(node);
        } else if (node.getDirection() == Node.UP) {
            nodesNeg.remove(node);
        }        
    }
    
    public Node nodeInFront(Node node) {
        if (node.getDirection() == Node.DOWN) {
            return searchNearestNode(node, nodesPos, false, true);
            
        } else if (node.getDirection() == Node.UP) {
            return searchNearestNode(node, nodesNeg, false, false);
                    
        } else {
            return null;
        }
    }
    
    public Node nodeBehind(Node node) {
        if (node.getDirection() == Node.DOWN) {
            return searchNearestNode(node, nodesPos, false, false);
            
        } else if (node.getDirection() == Node.UP) {
            return searchNearestNode(node, nodesNeg, false, true);
            
        } else {
            return null;
        }
    }
    
    public double getX() {
        return x;
    }
    
    public String toString() {
        return "VStreet: " + x;
    }
}
