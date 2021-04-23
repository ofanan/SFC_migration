/*
 * Position.java
 *
 * Created on 30 de junio de 2008, 12:19
 *
 * Represents a position in the world
 */

package nodes;

/**
 *
 * @author Manuel Fogué Cortés
 */
public class Position {
    
    private double x;
    private double y;
    private double z;
    
    /** Creates a new instance of Position */
    public Position(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    
    public Position(double x, double y) {
        this.x = x;
        this.y = y;
        this.z = 0;
    }
    
    public double distanceTo(Position p) {
        double diffX = this.x - p.x;
        double diffY = this.y - p.y;
        double diffZ = this.z - p.z;
        if (diffX == 0) {
            return Math.abs(diffY);
        } else if (diffY == 0) {
            return Math.abs(diffX);
        } else {
            return Math.sqrt(diffX*diffX + diffY*diffY + diffZ*diffZ);
        }
    }

    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        if (!(o instanceof Position)) return false;
        Position pos = (Position) o;
        return this.x == pos.x && this.y == pos.y && this.z == pos.z;
    }
    
    public void copy(Position p) {
        this.x = p.x;
        this.y = p.y;
        this.z = p.z;
    }
    
    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getZ() {
        return z;
    }

    public void setZ(double z) {
        this.z = z;
    }
    
    public String toString() {
        return "(" + x + ", " + y + ")";
    }
}
