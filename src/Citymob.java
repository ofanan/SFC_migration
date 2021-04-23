/*
 * Citymob.java
 *
 * Created on 13 de septiembre de 2008, 11:02
 *
 * Main class without GUI
 */

import java.util.ArrayList;

import sim.*;
import sim.PointOfInterest;
import sim.ScenarioCreationException;
import sim.Simulation;

//java -classpath C:\Users\ofanan\Documents\GitHub\SFC_migration\build\classes HelloWorld
/**
 *
 * @author Manuel Fogué Cortés
 */
public class Citymob {
    
    public static void main(String arg[]) {
        
        ArrayList<PointOfInterest> pois = new ArrayList<PointOfInterest>();
        Simulation sim = new Simulation();
        
        for (int i = 0; i < arg.length; i += 2) {
            
            if (arg[i].equals("-n")) {
                
                try {
                    int nodes = Integer.parseInt(arg[i+1]);
                    sim.NODES = nodes;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Nodes");
                    System.exit(1);
                }
                
                if (sim.NODES <= 0) {
                    System.err.println("'Nodes' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-w")) {
                
                try {
                    double maxX = Double.parseDouble(arg[i+1]);
                    sim.MAX_X = maxX;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Max X");
                    System.exit(1);
                }
                
                if (sim.MAX_X <= 0) {
                    System.err.println("'Max X' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-h")) {
                
                try {
                    double maxY = Double.parseDouble(arg[i+1]);
                    sim.MAX_Y = maxY;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Max Y");
                    System.exit(1);
                }
                
                if (sim.MAX_Y <= 0) {
                    System.err.println("'Max Y' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-t")) {
                
                try {
                    double maxTime = Double.parseDouble(arg[i+1]);
                    sim.MAX_TIME = maxTime;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Max Time");
                    System.exit(1);
                }
                
                if (sim.MAX_TIME <= 0) {
                    System.err.println("'Max Time' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-d")) {
                
                try {
                    double streetDist = Double.parseDouble(arg[i+1]);
                    sim.STREET_DIST = streetDist;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Street Distance");
                    System.exit(1);
                }
                
                if (sim.STREET_DIST <= 0) {
                    System.err.println("'Street Distance' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-l")) {
                
                try {
                    int lanes = Integer.parseInt(arg[i+1]);
                    sim.LANES = lanes;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Lanes");
                    System.exit(1);
                }
                
                if (sim.LANES < 1) {
                    System.err.println("'Lanes' must be greater than 0");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-a")) {
                
                try {
                    int accidents = Integer.parseInt(arg[i+1]);
                    sim.ACCIDENTS = accidents;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Accidents");
                    System.exit(1);
                }
                
                if (sim.ACCIDENTS < 0) {
                    System.err.println("'Accidents' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-A")) {
                
                try {
                    double alpha = Double.parseDouble(arg[i+1]);
                    sim.ALPHA = alpha;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Alpha");
                    System.exit(1);
                }
                
                if (sim.ALPHA < 0.0 || sim.ALPHA > 1.0) {
                    System.err.println("'Alpha' must take a value between 0.0 and 1.0");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-D")) {
                
                try {
                    double delta = Double.parseDouble(arg[i+1]);
                    sim.DELTA = delta;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Delta");
                    System.exit(1);
                }
                
                if (sim.DELTA < 0) {
                    System.err.println("'Delta' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-sprob")) {
                
                try {
                    double semaphoreProb = Double.parseDouble(arg[i+1]);
                    sim.SEMAPHORE_PROB = semaphoreProb;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Semaphore Probability");
                    System.exit(1);
                }
                
                if (sim.SEMAPHORE_PROB < 0.0 || sim.SEMAPHORE_PROB > 1.0) {
                    System.err.println("'Semaphore probability' must take a value between 0.0 and 1.0");
                    System.exit(1); 
                }
                
            } else if (arg[i].equals("-spause")) {
                
                try {
                    double semaphorePause = Double.parseDouble(arg[i+1]);
                    sim.SEMAPHORE_PAUSE = semaphorePause;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Semaphore Pause");
                    System.exit(1);
                }
                
                if (sim.SEMAPHORE_PAUSE < 0) {
                    System.err.println("'Semaphore Pause' must be positive or equals 0");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-maxs")) {
                
                try {
                    double maxSpeed = Double.parseDouble(arg[i+1]);
                    sim.MAX_SPEED = maxSpeed;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Max Speed");
                    System.exit(1);
                }
                
                if (sim.MAX_SPEED < 0) {
                    System.err.println("'Max Speed' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-mins")) {
                
                try {
                    double minSpeed = Double.parseDouble(arg[i+1]);
                    sim.MIN_SPEED = minSpeed;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Min Speed");
                    System.exit(1);
                }
                
                if (sim.MIN_SPEED < 0) {
                    System.err.println("'Min Speed' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-maxsd")) {
                
                try {
                    double maxSpeed = Double.parseDouble(arg[i+1]);
                    sim.MAX_SPEED_DOWN = maxSpeed;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Max Speed Downtown");
                    System.exit(1);
                }
                
                if (sim.MAX_SPEED_DOWN < 0) {
                    System.err.println("'Max Speed Downtown' must be positive");
                    System.exit(1);
                }
                
            } else if (arg[i].equals("-minsd")) {
                
                try {
                    double minSpeed = Double.parseDouble(arg[i+1]);
                    sim.MIN_SPEED_DOWN = minSpeed;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Min Speed Downtown");
                    System.exit(1);
                }
                
                if (sim.MIN_SPEED_DOWN < 0) {
                    System.err.println("'Min Speed Downtown' must be positive");
                    System.exit(1);
                }
                
            }  else if (arg[i].equals("-poi")) {
                
                try {
                    double minX = Double.parseDouble(arg[i+1]);
                    double minY = Double.parseDouble(arg[i+2]);
                    double maxX = Double.parseDouble(arg[i+3]);
                    double maxY = Double.parseDouble(arg[i+4]);
                    pois.add(new PointOfInterest(minX, minY, maxX, maxY));
                    i += 3;
                } catch (NumberFormatException ex) {
                    System.err.println("Error in Point of Interest");
                    System.exit(1);
                }
                
            }  else if (arg[i].equals("-help")) {
                
                help();
                usage();
                
            } else {
                
                System.err.println("Unsupported option: " + arg[i]);
                usage();
                
            }
            
        }
       
        
        sim.pois = pois;
        sim.initializeSimulation();
        try {
            sim.generateStaticScenario();
            sim.startSimulation();
        } catch (ScenarioCreationException ex) {
            System.err.println("Scenario creation failed");
        } 
        
    }
    
    public static void help() {
        
        System.err.println("/*****************************************************************************/");
        System.err.println("/* Citymob.java");
        System.err.println("/*");
        System.err.println("/*");
        System.err.println("/* Author:  Manuel Fogué Cortés");
        System.err.println("/*");
        System.err.println("/* This software is released into the public domain.");
        System.err.println("/* You are free to use it in any way you like.");
        System.err.println("/*");
        System.err.println("/* This software is provided \"as is\" with no expressed");
        System.err.println("/* or implied warranty.  I accept no liability for any");
        System.err.println("/* damage or loss of business that this software may cause.");
        System.err.println("/*");
        System.err.println("/*****************************************************************************/");
        
    }
    
    
    public static void usage() {
        
        System.err.println("Usage: java Citymob -n <nodes number> -t <simulation time> -w <map width> -h <map heigth>");
        System.err.println("\t -d <streets distance> -l <lanes per street> -a <accidents number>");
        System.err.println("\t -maxs <max speed> -mins <min speed> -maxsd <max speed downtown> -minsd <min speed downtown>");
        System.err.println("\t -A <alpha> -D <delta> -sprob <semaphore probability> -spause <semaphore pause>");
        System.err.println("\t {-poi <min X> <min Y> <max X> <max Y>, ...}");
        System.err.println("\nSpecial options:");
        System.err.println("\t -help \t Shows help message");
        
    }
    
}
