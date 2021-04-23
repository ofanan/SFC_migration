/*
 * Street.java
 *
 * Created on 4 de julio de 2008, 12:34
 *
 * Represents a street of the map
 */

package sim;

import java.util.HashSet;
import java.util.List;
import nodes.Node;
import nodes.Position;

/**
 *
 * @author Manuel Fogué Cortés
 */
public abstract class Street {
        
    protected HashSet<Node> nodesPos;
    protected HashSet<Node> nodesNeg;
    protected Simulation sim;
    protected int lanes = 1;
    
    /** Creates a new instance of Street */
    public Street(Simulation sim) {
        nodesPos = new HashSet<Node>();
        nodesNeg = new HashSet<Node>();
        this.sim = sim;
    }
    
    public Street(Simulation sim, int lanes) {
        this(sim);
        this.lanes = lanes;
    }
    
    /*
     * Add a new node moving in the street
     */
    public abstract void add(Node node);
    
    /*
     * Remove a node which is not moving in the street anymore
     */
    public abstract void remove(Node node);
    
    /*
     * Indicates whether the street contains a node
     */
    public boolean contains(Node node) {
        return nodesPos.contains(node) || nodesNeg.contains(node);
    }
    
    /*
     * Return the node just in front of another node in the same street 
     * (if there is any)
     */
    public abstract Node nodeInFront(Node node);
    
    /*
     * Return the node just behind another node in the same street 
     * (if there is any)
     */
    public abstract Node nodeBehind(Node node);
        
    /*
     * Determines if a node is colliding with some node of the street which is
     * moving to the specified direction (UP, DOWN, LEFT, RIGHT)
     */
    public boolean collision(Node node, int direction) {
        if (direction == Node.DOWN || direction == Node.RIGHT) {
            for (Node n: nodesPos) {
                if (n.distanceTo(node) < 2*sim.DELTA) {
                    return true;
                }
            }
        } else {
            for (Node n: nodesNeg) {
                if (n.distanceTo(node) < 2*sim.DELTA) {
                    return true;
                }
            }
        }
        return false;
    }
    
    /*
     * Determines if a node is colliding with some node of the street which is
     * moving to the specified direction in a concrete lane (UP, DOWN, LEFT, RIGHT)
     */
    public boolean collision(Node node, int direction, int lane) {
        if (direction == Node.DOWN || direction == Node.RIGHT) {
            for (Node n: nodesPos) {
                if (node.getLane() == n.getLane() && n.distanceTo(node) < sim.DELTA) {
                    return true;
                }
            }
        } else {
            for (Node n: nodesNeg) {
                if (node.getLane() == n.getLane() && n.distanceTo(node) < sim.DELTA) {
                    return true;
                }
            }
        }
        return false;
    }
    
    public String toString() {
        return "Street";
    }
    
    public String nodesToString() {
        String res = "-->";
        for (Node n: nodesPos) {
            res += n.getCurrentPosition();
        }
        res += "\n<--";
        for (Node n: nodesNeg) {
            res += n.getCurrentPosition();
        }
        return res;
    }
    
    /*
     * Looks for the nearest node to another one (auxiliar function used in methods
     * 'nodeInFront' and 'nodeBehind')
     */
    protected Node searchNearestNode(Node node, HashSet<Node> streetNodes, boolean horizontal, boolean inFront) {
        Node nearestNode = null;
        double minDistance = 0;            
        for (Node stNode: streetNodes) {
            boolean foundNode;
            double stNodePos;
            double nodePos;
            if (horizontal) {
                stNodePos = stNode.getX();
                nodePos = node.getX();
            } else {
                stNodePos = stNode.getY();
                nodePos = node.getY();
            }
            if (inFront) {
                foundNode = stNodePos > nodePos;
            } else {
                foundNode = stNodePos < nodePos;
            }
            if (foundNode) {
                double dis = stNode.distanceTo(node);
                if (nearestNode == null || dis < minDistance) {
                    minDistance = dis;
                    nearestNode = stNode;
                }
            }
        }
        return nearestNode;
    }
    
}
