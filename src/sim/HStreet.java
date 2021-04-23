/*
 * HStreet.java
 *
 * Created on 4 de julio de 2008, 12:43
 *
 * Represents a horizontal street
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
public class HStreet extends Street {
    
    public double y;
    
    /**
     * Creates a new instance of HStreet
     */
    public HStreet(Simulation sim, double y, int lanes) {
        super(sim, lanes);
        this.y = y;
    }
    
    public void add(Node node) {
        if (node.getDirection() == Node.RIGHT) {
            nodesPos.add(node);
        } else if (node.getDirection() == Node.LEFT) {
            nodesNeg.add(node);
        }   
    }
    
    public void remove(Node node) {
        if (node.getDirection() == Node.RIGHT) {
            nodesPos.remove(node);
        } else if (node.getDirection() == Node.LEFT) {
            nodesNeg.remove(node);
        }        
    }
    
    public Node nodeInFront(Node node) {
        if (node.getDirection() == Node.RIGHT) {
            return searchNearestNode(node, nodesPos, true, true);
            
        } else if (node.getDirection() == Node.LEFT) {
            return searchNearestNode(node, nodesNeg, true, false);
                        
        } else {
            return null;
        }
    }
    
    public Node nodeBehind(Node node) {
        if (node.getDirection() == Node.RIGHT) {
            return searchNearestNode(node, nodesPos, true, false);
            
        } else if (node.getDirection() == Node.LEFT) {
            return searchNearestNode(node, nodesNeg, true, true);
            
        } else {
            return null;
        }
    }
    
    public double getY() {
        return y;
    }
    
    public String toString() {
        return "HStreet: " + y;
    }
    
}
