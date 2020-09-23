# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:27:42 2020

@author: u15012639
"""

import numpy as np
import matplotlib.pyplot as plt
import definitions
import os


class Marc_Tables(object):
    def __init__(self, number_of_increments, plot = True, write = True,fig_save = False):
         self.number_of_increments = number_of_increments
         self.time = np.linspace(0,1,number_of_increments*2+1) # Add an additional increment for the intial touching
         
         #Compute the tables
         self.motion = self.gear_motion()
         self.moment = self.applied_moment()
         
         #Plot the tables
         if plot == True:
             self.plot_tables(fig_save = fig_save)
             
        #Save the tables in a format that Marc can read
         if write == True:
            motion_name = definitions.root + "\\models\\fem\\tables\\motion_" + str(self.number_of_increments) + ".txt"
            moment_name = definitions.root + "\\models\\fem\\tables\\moment_" + str(self.number_of_increments) + ".txt"
            time_name = definitions.root + "\\models\\fem\\tables\\time_" + str(self.number_of_increments) + ".txt"
            np.savetxt(motion_name,self.motion)
            np.savetxt(moment_name,self.moment)
            np.savetxt(time_name,self.time)

    def gear_motion(self):
        """
        writes out a table for use in Marc Mentat that dictates the position of a gear at a time increment
        """
        increase = 1/(self.number_of_increments-1) # The amount by which the angle is incremented each iteration
        
        motion = np.zeros((self.number_of_increments*2+1,2))
        motion[:,0] =self.time  # Set the time array in the main array to be exported to txt
        
        motion[0,1] = 0 # No motion whist the moment brings the gear into contact
        motion[1,1] = 0 # No motion whist moment is applied for first test
        
        counter = 2
        loading_cycle = True
        for i in range(self.number_of_increments*2-1):
            if loading_cycle == True:
                motion[counter,1] = motion[counter-1,1] # When in a loading cycle, keep rotation constant
            else:
                motion[counter,1] = motion[counter-1,1] + increase
            
            loading_cycle = not loading_cycle
            counter += 1
        return motion
    
    def applied_moment(self):
        """
        writes out a table for use in Marc Mentat that dictates the moment on te gear for a time increment
        """
        self.small_moment = 0.01 #Set the contact maintain moment to be 1/100 of actual full moment
        
        moment = np.zeros((self.number_of_increments*2+1,2))
        moment[:,0] =self.time  # Set the time array in the main array to be exported to txt
        
        moment[0,1] = 0 # Brings the gear into contact
        moment[1,1] = self.small_moment # End of bringing into contact, start loading
        
        counter = 2
        loading_cycle = True
        for i in range(self.number_of_increments*2-1):
            if loading_cycle == True:
                moment[counter,1] = 1 # When in a loading cycle, keep rotation constant
            else:
                moment[counter,1] = self.small_moment
            
            loading_cycle = not loading_cycle
            counter += 1
        return moment
    
    def plot_tables(self,fig_save = False):
        plt.figure()

        # Plot the angular displacement over time
        A = self.motion
        plt.plot(A[:,0],A[:,1],"-k", label =  "% of total ring angular displacement")

        # Plot the moment over time
        B = self.moment
        plt.plot(B[:,0], B[:,1], "--k",label = "% of maximum applied moment")

        # Plot the lower torque range
        #plt.plot(B[:,0], np.ones(len(B[:,0]))*self.small_moment)


        plt.grid(which="both")
        plt.xlabel("Simulation time")
        plt.ylabel("%")
        plt.xticks(np.arange(0, 1, step=0.1))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))

        if fig_save:
            repos = os.path.abspath(os.path.join(definitions.root, os.pardir))
            fig_save_path = repos + "\\fem_images\\tables_5.pdf"
            plt.savefig(fig_save_path, bbox_inches = "tight")
        #plt.scatter(self.time,self.time)
        return
        
    
# table_generator = Marc_Tables(5, plot=True, fig_save=True, write=False)
table_generator = Marc_Tables(1000, plot=False, fig_save=False, write=True)



