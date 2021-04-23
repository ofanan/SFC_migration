/*
 * ScenarioCreationException.java
 *
 * Created on 7 de julio de 2008, 10:47
 *
 * Represents an error ocurred during the creation of a new scenario
 */

package sim;

/**
 *
 * @author Manuel Fogué Cortés
 */
public class ScenarioCreationException extends Exception {
    
    private int index;
    
    /** Creates a new instance of ScenarioCreationException */
    public ScenarioCreationException(int index) {
        this.index = index;
    }
    
    public String getMessage() {
        return "Error during scenario creation, caused by node " + index;
    }
    
}
