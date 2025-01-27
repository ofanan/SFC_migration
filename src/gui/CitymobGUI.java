/*
 * CitymobGUI.java
 *
 * Created on 6 de agosto de 2008, 22:33
 */

package gui;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import javax.swing.DefaultListModel;
import javax.swing.JOptionPane;
import sim.PointOfInterest;
import sim.ScenarioCreationException;
import sim.Simulation;

/**
 *
 * @author  Manuel Fogu� Cort�s
 */
public class CitymobGUI extends javax.swing.JFrame {
    
    private Simulation sim;
    private ArrayList<PointOfInterest> pois = new ArrayList<PointOfInterest>();
    
    /** Creates new form CitymobGUI */
    public CitymobGUI() {
        initComponents();
        getRootPane().setDefaultButton(b_startSimulation);
        setLocationByPlatform(true);
    }
    
    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    // <editor-fold defaultstate="collapsed" desc=" Generated Code ">//GEN-BEGIN:initComponents
    private void initComponents() {
        jMenuBar2 = new javax.swing.JMenuBar();
        jMenu1 = new javax.swing.JMenu();
        jPanel1 = new javax.swing.JPanel();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        jLabel4 = new javax.swing.JLabel();
        jLabel5 = new javax.swing.JLabel();
        jLabel6 = new javax.swing.JLabel();
        jLabel7 = new javax.swing.JLabel();
        tf_streetDist = new javax.swing.JTextField();
        tf_maxTime = new javax.swing.JTextField();
        tf_nodes = new javax.swing.JTextField();
        tf_maxX = new javax.swing.JTextField();
        tf_maxY = new javax.swing.JTextField();
        tf_lanes = new javax.swing.JTextField();
        tf_accidents = new javax.swing.JTextField();
        jLabel8 = new javax.swing.JLabel();
        jLabel9 = new javax.swing.JLabel();
        jLabel10 = new javax.swing.JLabel();
        jLabel11 = new javax.swing.JLabel();
        jLabel12 = new javax.swing.JLabel();
        jLabel13 = new javax.swing.JLabel();
        jLabel14 = new javax.swing.JLabel();
        jLabel15 = new javax.swing.JLabel();
        tf_semaphorePause = new javax.swing.JTextField();
        tf_semaphoreProb = new javax.swing.JTextField();
        tf_delta = new javax.swing.JTextField();
        tf_alpha = new javax.swing.JTextField();
        tf_minSpeedPOI = new javax.swing.JTextField();
        tf_maxSpeedPOI = new javax.swing.JTextField();
        tf_minSpeed = new javax.swing.JTextField();
        tf_maxSpeed = new javax.swing.JTextField();
        jPanel2 = new javax.swing.JPanel();
        b_addIP = new javax.swing.JButton();
        b_removeIP = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        jl_ips = new javax.swing.JList();
        jPanel4 = new javax.swing.JPanel();
        tf_outFile = new javax.swing.JTextField();
        b_startSimulation = new javax.swing.JButton();
        jMenuBar1 = new javax.swing.JMenuBar();
        tf_menuHelp = new javax.swing.JMenu();
        mCitymob = new javax.swing.JMenuItem();
        jSeparator1 = new javax.swing.JSeparator();
        mAbout = new javax.swing.JMenuItem();

        jMenu1.setText("Menu");
        jMenuBar2.add(jMenu1);

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Citymob 2.0");
        setResizable(false);
        jPanel1.setBorder(javax.swing.BorderFactory.createTitledBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED), "Simulation Parameters"));
        jPanel1.setName("Parameters");
        jLabel1.setText("Nodes");
        jLabel1.setToolTipText("Number of nodes");

        jLabel2.setText("Max X");
        jLabel2.setToolTipText("Width of the simulation area (m)");

        jLabel3.setText("Max Y");
        jLabel3.setToolTipText("Height of the simulation area (m)");

        jLabel4.setText("Max Time");
        jLabel4.setToolTipText("Maximum simulation time (s)");

        jLabel5.setText("Street Dist");
        jLabel5.setToolTipText("Distance between streets (m)");

        jLabel6.setText("Lanes");
        jLabel6.setToolTipText("Number of lanes per street and direction");

        jLabel7.setText("Accidents");
        jLabel7.setToolTipText("Number of accidents occured");

        tf_streetDist.setText("20");

        tf_maxTime.setText("60");

        tf_nodes.setText("50");
        tf_nodes.setName("tf_Nodes");

        tf_maxX.setText("100");

        tf_maxY.setText("100");

        tf_lanes.setText("1");

        tf_accidents.setText("5");

        jLabel8.setText("Alpha");
        jLabel8.setToolTipText("Probability of visiting a point of interest (0.0 - 1.0)");

        jLabel9.setText("Delta");
        jLabel9.setToolTipText("Minimum distance between nodes (m)");

        jLabel10.setText("Semaphore Prob");
        jLabel10.setToolTipText("Probabilty of stopping at a semaphore (0.0 - 1.0)");

        jLabel11.setText("Semaphore Pause");
        jLabel11.setToolTipText("Maximum waiting time at a semaphore (s)");

        jLabel12.setText("Max Speed");
        jLabel12.setToolTipText("Maximum speed (Km/h)");

        jLabel13.setText("Min Speed");
        jLabel13.setToolTipText("Minimum speed (Km/h)");

        jLabel14.setText("Max Speed POI");
        jLabel14.setToolTipText("Maximum speed near a point of interest (Km/h)");

        jLabel15.setText("Min Speed POI");
        jLabel15.setToolTipText("Minimum speed near a point of interest (Km/h)");

        tf_semaphorePause.setText("20");

        tf_semaphoreProb.setText("0.3");

        tf_delta.setText("5");

        tf_alpha.setText("0.5");

        tf_minSpeedPOI.setText("25");

        tf_maxSpeedPOI.setText("50");

        tf_minSpeed.setText("50");

        tf_maxSpeed.setText("100");

        jPanel2.setBorder(javax.swing.BorderFactory.createTitledBorder("Points of Interest"));
        jPanel2.setToolTipText("Points of interest");
        b_addIP.setText("Add");
        b_addIP.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                b_addIPMouseClicked(evt);
            }
        });

        b_removeIP.setText("Remove");
        b_removeIP.setEnabled(false);
        b_removeIP.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                b_removeIPMouseClicked(evt);
            }
        });

        jl_ips.addListSelectionListener(new javax.swing.event.ListSelectionListener() {
            public void valueChanged(javax.swing.event.ListSelectionEvent evt) {
                jl_ipsValueChanged(evt);
            }
        });

        jScrollPane1.setViewportView(jl_ips);

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                    .addComponent(b_addIP, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(b_removeIP, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 243, Short.MAX_VALUE)
                .addContainerGap())
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addComponent(b_addIP)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(b_removeIP))
                    .addComponent(jScrollPane1))
                .addContainerGap())
        );

        jPanel4.setBorder(javax.swing.BorderFactory.createTitledBorder("Output File"));
        jPanel4.setToolTipText("Name of ouput file");

        javax.swing.GroupLayout jPanel4Layout = new javax.swing.GroupLayout(jPanel4);
        jPanel4.setLayout(jPanel4Layout);
        jPanel4Layout.setHorizontalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel4Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(tf_outFile, javax.swing.GroupLayout.DEFAULT_SIZE, 320, Short.MAX_VALUE)
                .addContainerGap())
        );
        jPanel4Layout.setVerticalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel4Layout.createSequentialGroup()
                .addComponent(tf_outFile, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(jPanel4, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jPanel2, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(javax.swing.GroupLayout.Alignment.LEADING, jPanel1Layout.createSequentialGroup()
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel5)
                            .addComponent(jLabel4)
                            .addComponent(jLabel1)
                            .addComponent(jLabel2)
                            .addComponent(jLabel3)
                            .addComponent(jLabel6)
                            .addComponent(jLabel7))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(tf_accidents)
                            .addComponent(tf_lanes)
                            .addComponent(tf_maxY)
                            .addComponent(tf_maxX)
                            .addComponent(tf_maxTime)
                            .addComponent(tf_streetDist, javax.swing.GroupLayout.DEFAULT_SIZE, 72, Short.MAX_VALUE)
                            .addComponent(tf_nodes, javax.swing.GroupLayout.PREFERRED_SIZE, 93, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel11)
                            .addComponent(jLabel10)
                            .addComponent(jLabel9)
                            .addComponent(jLabel8)
                            .addComponent(jLabel15)
                            .addComponent(jLabel14)
                            .addComponent(jLabel13)
                            .addComponent(jLabel12))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(tf_maxSpeed)
                            .addComponent(tf_minSpeed)
                            .addComponent(tf_maxSpeedPOI)
                            .addComponent(tf_minSpeedPOI)
                            .addComponent(tf_alpha)
                            .addComponent(tf_delta)
                            .addComponent(tf_semaphoreProb)
                            .addComponent(tf_semaphorePause, javax.swing.GroupLayout.DEFAULT_SIZE, 98, Short.MAX_VALUE))))
                .addGap(22, 22, 22))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel1)
                    .addComponent(tf_nodes, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel12)
                    .addComponent(tf_maxSpeed, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel2)
                    .addComponent(tf_maxX, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel13)
                    .addComponent(tf_minSpeed, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel3)
                    .addComponent(tf_maxY, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel14)
                    .addComponent(tf_maxSpeedPOI, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel4)
                    .addComponent(tf_maxTime, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel15)
                    .addComponent(tf_minSpeedPOI, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel5)
                    .addComponent(tf_streetDist, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel8)
                    .addComponent(tf_alpha, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel6)
                    .addComponent(tf_lanes, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel9)
                    .addComponent(tf_delta, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel7)
                    .addComponent(tf_accidents, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel10)
                    .addComponent(tf_semaphoreProb, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel11)
                    .addComponent(tf_semaphorePause, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanel4, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        b_startSimulation.setFont(new java.awt.Font("Tahoma", 1, 18));
        b_startSimulation.setText("Start Simulation");
        b_startSimulation.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                b_startSimulationMouseClicked(evt);
            }
        });
        b_startSimulation.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                b_startSimulationActionPerformed(evt);
            }
        });

        tf_menuHelp.setText("Help");
        mCitymob.setText("Simulation");
        mCitymob.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                mCitymobActionPerformed(evt);
            }
        });

        tf_menuHelp.add(mCitymob);

        tf_menuHelp.add(jSeparator1);

        mAbout.setText("About");
        mAbout.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                mAboutActionPerformed(evt);
            }
        });

        tf_menuHelp.add(mAbout);

        jMenuBar1.add(tf_menuHelp);

        setJMenuBar(jMenuBar1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(b_startSimulation, javax.swing.GroupLayout.DEFAULT_SIZE, 396, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(b_startSimulation, javax.swing.GroupLayout.PREFERRED_SIZE, 47, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void mCitymobActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_mCitymobActionPerformed

        new HelpCitymob(this);
        
    }//GEN-LAST:event_mCitymobActionPerformed

    private void mAboutActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_mAboutActionPerformed

        new About(this);
        
    }//GEN-LAST:event_mAboutActionPerformed

    private void b_startSimulationActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_b_startSimulationActionPerformed
    
        this.b_startSimulationMouseClicked(null);
        
    }//GEN-LAST:event_b_startSimulationActionPerformed

    private void b_startSimulationMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_b_startSimulationMouseClicked
        
        String field = "";
        
        try {
            
            field = "Nodes";
            int nodes = Integer.parseInt(tf_nodes.getText());
            if (nodes <= 0) {
                throw new Exception("'Nodes' must be positive");
            }
            
            field = "Max X";
            double maxX = Double.parseDouble(tf_maxX.getText());
            if (maxX <= 0) {
                throw new Exception("'Max X' must be positive");
            }
            
            field = "Max Y";
            double maxY = Double.parseDouble(tf_maxY.getText());
            if (maxY <= 0) {
                throw new Exception("'Max Y' must be positive");
            }
            
            field = "Max Time";
            double maxTime = Double.parseDouble(tf_maxTime.getText());
            if (maxTime <= 0) {
                throw new Exception("'Max Time' must be positive");
            }
            
            field = "Street Distance";
            double streetDist = Double.parseDouble(tf_streetDist.getText());
            if (streetDist <= 0) {
                throw new Exception("'Street Distance' must be positive");
            }
            
            field = "Lanes";
            int lanes = Integer.parseInt(tf_lanes.getText());
            if (lanes < 1) {
                throw new Exception("'Lanes' must be greater than 1");
            }
            
            field = "Accidents";
            int accidents = Integer.parseInt(tf_accidents.getText());
            if (accidents < 0) {
                throw new Exception("'Accidents' must be positive or equals 0");
            }
            
            field = "Max Speed";
            double maxSpeed = Double.parseDouble(tf_maxSpeed.getText());
            if (maxSpeed <= 0) {
                throw new Exception("'Max Speed' must be positive");
            }
            
            field = "Min Speed";
            double minSpeed = Double.parseDouble(tf_minSpeed.getText());
            if (minSpeed > maxSpeed) {
                throw new Exception("'Max Speed' must be greater than 'Min Speed'");
            }
            
            field = "Max Speed POI";
            double maxSpeedPOI = Double.parseDouble(tf_maxSpeedPOI.getText());
            if (maxSpeedPOI <= 0) {
                throw new Exception("'Max Speed POI' must be positive");
            }
            
            field = "Min Speed POI";
            double minSpeedPOI = Double.parseDouble(tf_minSpeedPOI.getText());
            if (minSpeedPOI > maxSpeedPOI) {
                throw new Exception("'Max Speed POI' must be greater than 'Min Speed POI'");
            }
            
            field = "Alpha";
            double alpha = Double.parseDouble(tf_alpha.getText());
            if (alpha < 0.0 || alpha > 1.0) {
                throw new Exception("'Alpha' must take a value between 0.0 and 1.0");
            }
            
            field = "Delta";
            double delta = Double.parseDouble(tf_delta.getText());
            if (delta < 0) {
                throw new Exception("'Delta' must be positive or equals 0");
            }
            
            field = "Semaphore Probability";
            double semProb = Double.parseDouble(tf_semaphoreProb.getText());
            if (semProb < 0.0 || semProb > 1.0) {
                throw new Exception("'Semaphore probability' must take a value between 0.0 and 1.0");
            }
            
            field = "Semaphore Pause";
            double semPause = Double.parseDouble(tf_semaphorePause.getText());
            if (semPause < 0) {
                throw new Exception("'Semaphore pause' must be positive or equals 0");
            }
            
            
            String outputFile = tf_outFile.getText();
            if (outputFile == null || outputFile.trim().equals("")) {
                throw new Exception("You must indicate an output file name");
            } else {
                File output = new File(outputFile);
                if (output.exists()) {
                    int ans = JOptionPane.showConfirmDialog(this, "File '" + outputFile + "' already exists" +
                            "\nOverwrite it?", 
                            "Output file exists", JOptionPane.YES_NO_OPTION);
                    if (ans == JOptionPane.NO_OPTION) {
                        return;
                    }
                    
                } else {
                    output.createNewFile();
                }
                
                b_startSimulation.setEnabled(false);
                
                sim = new Simulation();
                                
                sim.ACCIDENTS = accidents;
                sim.ALPHA = alpha;
                sim.DELTA = delta;
                sim.LANES = lanes;
                sim.MAX_SPEED = maxSpeed;
                sim.MAX_SPEED_DOWN = maxSpeedPOI;
                sim.MIN_SPEED = minSpeed;
                sim.MIN_SPEED_DOWN = minSpeedPOI;
                sim.MAX_TIME = maxTime;
                sim.MAX_X = maxX;
                sim.MAX_Y = maxY;
                sim.NODES = nodes;
                sim.SEMAPHORE_PAUSE = semPause;
                sim.SEMAPHORE_PROB = semProb;
                sim.STREET_DIST = streetDist;
                
                sim.pois = pois;
                
                PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(outputFile)));
                sim.setOut(out);
                
                sim.initializeSimulation();
                sim.generateStaticScenario();
                sim.startSimulation();   
                
                JOptionPane.showMessageDialog(this, "Simulation finished", 
                    "Finish", JOptionPane.INFORMATION_MESSAGE);
            }
            
            
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "Field '" + field + "' contains a wrong value", 
                    "Error", JOptionPane.ERROR_MESSAGE);
            
        } catch (ScenarioCreationException ex) {
            JOptionPane.showMessageDialog(this, "Error during scenario creation\nPlease check parameters", 
                    "Simulation error", JOptionPane.ERROR_MESSAGE);
        
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, e.getMessage(), 
                    "Error", JOptionPane.ERROR_MESSAGE);
        
        } finally {
            b_startSimulation.setEnabled(true);
        }
        
                
    }//GEN-LAST:event_b_startSimulationMouseClicked

    private void jl_ipsValueChanged(javax.swing.event.ListSelectionEvent evt) {//GEN-FIRST:event_jl_ipsValueChanged

        int i = jl_ips.getSelectedIndex();
        
        if (i != -1) {
            b_removeIP.setEnabled(true);
        }
        
    }//GEN-LAST:event_jl_ipsValueChanged

    private void b_removeIPMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_b_removeIPMouseClicked

        int i = jl_ips.getSelectedIndex();
        
        if (i != -1) {
            pois.remove(i);
            refreshPois();
            b_removeIP.setEnabled(false);
        }
        
    }//GEN-LAST:event_b_removeIPMouseClicked

    private void b_addIPMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_b_addIPMouseClicked
        
        AddPoI add = new AddPoI(this);
             
        
    }//GEN-LAST:event_b_addIPMouseClicked
    
    
    public void addPoI(PointOfInterest poi) throws Exception {
        if (!pois.contains(poi)) {
            if (isValid(poi)) {
                pois.add(poi);
                refreshPois();
            } else {
                throw new Exception("Point of Interest not valid");
            }
        } else {
            throw new Exception("Point of Interest already exists");
        }
    }
    
    public boolean isValid(PointOfInterest poi) {
        try {
            
            double maxX = Double.parseDouble(tf_maxX.getText());
            double maxY = Double.parseDouble(tf_maxY.getText());
            
            return poi.getMaxX() > poi.getMinX() && poi.getMaxY() > poi.getMinY() &&
                    poi.getMaxX() <= maxX && poi.getMaxY() <= maxY;
            
        } catch (NumberFormatException ex) {
            return false;
        }
        
    }
    
    private void refreshPois() {
        DefaultListModel model = new DefaultListModel();
        for (PointOfInterest i: pois) {
            model.addElement(i);
        }
        jl_ips.setModel(model);
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new CitymobGUI().setVisible(true);
            }
        });
    }
    
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton b_addIP;
    private javax.swing.JButton b_removeIP;
    private javax.swing.JButton b_startSimulation;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel10;
    private javax.swing.JLabel jLabel11;
    private javax.swing.JLabel jLabel12;
    private javax.swing.JLabel jLabel13;
    private javax.swing.JLabel jLabel14;
    private javax.swing.JLabel jLabel15;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabel7;
    private javax.swing.JLabel jLabel8;
    private javax.swing.JLabel jLabel9;
    private javax.swing.JMenu jMenu1;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JMenuBar jMenuBar2;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JSeparator jSeparator1;
    private javax.swing.JList jl_ips;
    private javax.swing.JMenuItem mAbout;
    private javax.swing.JMenuItem mCitymob;
    private javax.swing.JTextField tf_accidents;
    private javax.swing.JTextField tf_alpha;
    private javax.swing.JTextField tf_delta;
    private javax.swing.JTextField tf_lanes;
    private javax.swing.JTextField tf_maxSpeed;
    private javax.swing.JTextField tf_maxSpeedPOI;
    private javax.swing.JTextField tf_maxTime;
    private javax.swing.JTextField tf_maxX;
    private javax.swing.JTextField tf_maxY;
    private javax.swing.JMenu tf_menuHelp;
    private javax.swing.JTextField tf_minSpeed;
    private javax.swing.JTextField tf_minSpeedPOI;
    private javax.swing.JTextField tf_nodes;
    private javax.swing.JTextField tf_outFile;
    private javax.swing.JTextField tf_semaphorePause;
    private javax.swing.JTextField tf_semaphoreProb;
    private javax.swing.JTextField tf_streetDist;
    // End of variables declaration//GEN-END:variables
    
}
