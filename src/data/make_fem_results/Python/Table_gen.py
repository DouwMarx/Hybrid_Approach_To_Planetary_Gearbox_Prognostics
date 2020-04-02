# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:27:42 2020

@author: u15012639
"""

import numpy as np
import matplotlib.pyplot as plt

#a = np.random.rand(9,2)

#print(a)

#np.savetxt("numpy_test.txt",a)


class Marc_Tables(object):
    def __init__(self, number_of_increments, plot = True, write = True):
         self.number_of_increments = number_of_increments
         self.time = np.linspace(0,1,number_of_increments*2+1) # Add an additional increment for the intial touching
         
         #Compute the tables
         self.motion = self.gear_motion()
         self.moment = self.applied_moment()
         
         #Plot the tables
         if plot == True:
             self.plot_tables()
             
        #Save the tables in a format that Marc can read
         if write == True:
            motion_name = "..\\Tables\\motion_" + str(self.number_of_increments) + ".txt"
            moment_name = "..\\Tables\\moment_" + str(self.number_of_increments) + ".txt"
            time_name = "..\\Tables\\time_" + str(self.number_of_increments) + ".txt"
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
        small_moment = 0.01 #Set the contact maintain moment to be 1/100 of actual full moment
        
        moment = np.zeros((self.number_of_increments*2+1,2))
        moment[:,0] =self.time  # Set the time array in the main array to be exported to txt
        
        moment[0,1] = 0 # Brings the gear into contact
        moment[1,1] = small_moment # End of bringing into contact, start loading
        
        counter = 2
        loading_cycle = True
        for i in range(self.number_of_increments*2-1):
            if loading_cycle == True:
                moment[counter,1] = 1 # When in a loading cycle, keep rotation constant
            else:
                moment[counter,1] = small_moment
            
            loading_cycle = not loading_cycle
            counter += 1
        return moment
    
    def plot_tables(self):
        plt.figure()
        A = self.motion
        plt.plot(A[:,0],A[:,1],label = "Displacement")
        
        B = self.moment
        plt.plot(B[:,0],B[:,1],label = "Moment")
        
        plt.legend()
        plt.xlabel("time")
        
        plt.scatter(self.time,self.time)
        return
        
    
table_generator = Marc_Tables(200, plot=True, write=True)



