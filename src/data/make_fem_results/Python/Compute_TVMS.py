# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 08:09:07 2020

@author: u15012639
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

class TVMS(object):
    """Class that creates TVMS object in order to compute time varying mesh stiffness"""
    
    def __init__(self, Planet_Axle_Angle):
        """ Initializes the class by assigning the text files that contain angular displacements of the planet carrier axle rigid geaometry"""
        self.Planet_Axle_Angle = self.read_text_file(Planet_Axle_Angle)
        self.angle = self.Planet_Axle_Angle[:,1]
        return
    
    def read_text_file(self,file):
        a = np.loadtxt(file, skiprows =9)
        return a
    
    def plot_angle_measurements(self):
        plt.figure()
        #plt.scatter(self.Planet_Axle_Angle[:,0], self.Planet_Axle_Angle[:,1], label="Planet Axle angle")
        plt.plot(self.Planet_Axle_Angle[:, 0], self.Planet_Axle_Angle[:, 1], label="Planet Axle angle")
        plt.xlabel("Time [s]")
        plt.ylabel("Angular Position [rad]")
        plt.legend()
        return
    
    def measure_stiffness(self, Plot_Test = False):
        """Finds the indexes that will be used to compute the stiffness"""

        l = len(self.Planet_Axle_Angle[:,0])
        i_even = np.arange(0,l-1,2).astype(int)
        i_odd = np.arange(1,l,2).astype(int)
        
        ideal_angle = self.angle[i_odd][1:] #The actaul angle, toutching under no load
        deflected_angle = self.angle[i_even][1:]

        deflection_angle = deflected_angle - ideal_angle # The first value from no load to initial contact is discarded

        if Plot_Test ==True:
            plt.figure()
            plt.scatter(ideal_angle, deflection_angle)
            plt.xlabel("Ideal, Infinite stiffness angle")
            plt.ylabel("Angle of deflection")

        return ideal_angle,deflection_angle

n_increments = 10
l = []
for crack in range(1,4):
    mesh_name = "m1_a" + str(crack) + "mm"
    name = "Mr_1" + "..." + "Mp_" + mesh_name + "..." + "a_" + str(2.5) + "n_" + str(n_increments) + "_job1_angle"



    obj = TVMS("..\\Results\\" + name + ".txt")

    #obj.plot_angle_measurements()
    ideal,deflect = obj.measure_stiffness(Plot_Test=True)
    l.append(deflect)

#print(np.array(l))

plt.figure()
plt.contour(np.array(l))