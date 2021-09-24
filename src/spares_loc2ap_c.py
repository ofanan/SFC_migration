    # my_loc2ap.time_period_str = '0730_0740' #'0829_0830' #'0730_0830'
    # my_loc2ap.parse_files (['0730_0830_16secs.loc']) #(['0730.loc', '0740.loc', '0750.loc', '0800.loc', '0810.loc', '0820.loc'])

    # gamad = [[1,2],[3,4]]
    # nanas = [[0]*4 for i in range(4)]
    # for row in range (4):
    #     for col in range (4):
    #         nanas[row][col] = gamad[int(row/2)][int(col/2)] 
    #     #nanas[row] = [[gamad[i][j]]*2 for i in range (2)] # for j in range (2)]
    #
    # print (nanas)
    # exit ()
    #
    # heatmap_vals = [1]
    # for cur_power_of_4 in range(1,max_power_of_4+1):     
    #     my_loc2ap       = loc2ap_c (cur_power_of_4 = cur_power_of_4, use_sq_cells = True, verbose = [VERBOSE_POST_PROCESSING])
    #     input_file_name = 'num_of_vehs_per_ap_{}aps.txt' .format (4**cur_power_of_4)
    #     my_loc2ap.rd_num_of_vehs_per_ap (input_file_name)
    #     # my_loc2ap.plot_num_of_vehs_heatmap ()
    #     np.array (my_loc2ap.plot_num_of_vehs_heatmap ())
    #     heatmap_for_this_lvl = np.array (my_loc2ap.plot_num_of_vehs_heatmap ()) # The original heatmap, of size 2x2, 4x4, and so on.
    #     n                    = 2**max_power_of_4 # The required size of the heatmaps
    #     mega_pixel_heatmap =  [[0]*(n) for i in range(n)]           # Will contain the "zoomed" heatamp, when repeating each pixel several times, for adopting it to 16x16 resolution
    #     for row in range (n):
    #         for col in range (n):
    #             mega_pixel_heatmap[row][col] = heatmap_for_this_lvl[int(row/2)][int(row/2)] 
    #
    #
    #     heatmap_vals.append (np.array (my_loc2ap.plot_num_of_vehs_heatmap ()) )
    #     heatmap_vals.append (np.tile (np.array (my_loc2ap.plot_num_of_vehs_heatmap ()), 2**(4-cur_power_of_4)))
0    #
    # print (heatmap_vals[1])
    
    # df = pd.DataFrame(np.random.random((10,10)))
    # df = []
    # for max_power_of_4 in range(4):
    #     df.append (pd.DataFrame(np.random.random((10,10))))
    # df[1] = heatmap_vals[1]
    
    # heatmap_vals
    # fig, axn = plt.subplots(2, 2, sharex=True, sharey=True)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    #
    # for i, ax in enumerate(axn.flat):
    #     my_heatmap = sns.heatmap(heatmap_vals[i+1], ax=ax,
    #                  cbar=i == 0,
    #                  vmin=0, vmax=100,
    #                  cbar_ax=None if i else cbar_ax,
    #                  cmap="YlGnBu", norm=LogNorm())
    #     my_heatmap.tick_params(left=False, bottom=False) ## other options are right and top
    #
    # fig.tight_layout(rect=[0, 0, .9, 1])
    # plt.show ()

    
    # # my_loc2ap.plot_tot_num_of_vehs_over_t_graph()
    # my_loc2ap.print_num_of_vehs_diffs ()
    # output_file_name = 'num_of_vehs_per_server{}.txt' .format (4**max_power_of_4)
    # my_loc2ap.plot_num_of_vehs_per_ap_graph ()
    # my_loc2ap.print_num_of_vehs_per_server (output_file_name)
    
    # For finding the maximum positional values of x and y in the .loc file(s), uncomment the line below 
    # my_loc2ap.find_max_X_max_Y ()    


    # def rd_num_of_vehs_per_ap (self, input_file_name):
    #     """
    #     Read the number of vehicles at each cell, as written in the input files. 
    #     """
    #     input_file  = open ('../res/' + input_file_name, "r")  
    #
    #     self.num_of_vehs_in_ap = []
    #     for line in input_file:
    #
    #         if (line == "\n" or line.split ("//")[0] == ""):
    #             continue
    #
    #         num_of_vehs_in_cur_ap = []
    #         line = line.split ("\n")[0]
    #         splitted_line = line.split (":")
    #         splitted_line = splitted_line[1].split('[')[1].split(']')[0].split(', ')
    #         for cur_num_of_vehs_in_this_ap in splitted_line:
    #             num_of_vehs_in_cur_ap.append (int(cur_num_of_vehs_in_this_ap))
    #
    #         self.num_of_vehs_in_ap.append (num_of_vehs_in_cur_ap)            
        

        # Generate a tikz heatmap
        # avg_num_of_vehs_in_ap =  np.array ([np.average(self.num_of_vehs_in_ap[ap]) for ap in range(self.num_of_APs)])
        # n = int (math.sqrt(len(avg_num_of_vehs_in_ap)))
        # heatmap_val = np.array ([avg_num_of_vehs_in_ap[self.tile_to_ap[i]] for i in range (len(self.tile_to_ap))]).reshape ( [n, n])

        # # Generate a tikz heatmap
        # self.heatmap_output_file = open ('../res/heatmap_num_vehs_{}.dat' .format (4**self.max_power_of_4), 'w')
        # for i in range (2**self.max_power_of_4):
        #     for j in range (2**self.max_power_of_4):
        #         printf (self.heatmap_output_file, '{} {} {}\n' .format (j, i, heatmap_val[i][j]))
        #     printf (self.heatmap_output_file, '\n')
        #
        # printf (self.heatmap_output_file, '\n\n{}' .format(self.vec_to_heatmap (np.array ([np.average(self.num_of_vehs_in_ap[ap]) for ap in range(self.num_of_APs)]))))
        
