# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 08:09:07 2020

@author: u15012639
"""

import numpy as np
import matplotlib.pyplot as plt
import definitions
import os

plt.close("all")


class TVMS(object):
    """
    Class that creates TVMS object in order to compute time varying mesh stiffness
    SI units, radians
    """

    def __init__(self, Planet_Axle_Angle, Applied_Moment, Pitch_Diameter):
        """ Initializes the class by assigning the text files that contain angular displacements of the planet carrier axle rigid geaometry"""
        self.Planet_Axle_Angle = self.read_text_file(Planet_Axle_Angle)
        self.angle = self.Planet_Axle_Angle[:, 1]
        self.M_app = Applied_Moment
        self.Dp = Pitch_Diameter

        self.ideal, self.deflected = self.measure_deflection()

        self.linear_stiffness = self.compute_linear_stiffness()
        return

    def read_text_file(self, file):
        a = np.loadtxt(file, skiprows=9)
        return a

    def plot_angle_measurements(self):
        plt.figure()
        # plt.scatter(self.Planet_Axle_Angle[:,0], self.Planet_Axle_Angle[:,1], label="Planet Axle angle")
        plt.plot(self.Planet_Axle_Angle[:, 0], self.Planet_Axle_Angle[:, 1], label="Planet Axle angle")
        plt.xlabel("Time [s]")
        plt.ylabel("Angular Position [rad]")
        plt.legend()
        return

    def plot_linear_stiffness(self):
        plt.figure()
        plt.scatter(self.ideal, self.linear_stiffness)
        plt.ylabel("Stiffness [N/m]")
        return

    def compute_linear_stiffness(self):
        r = 0.5 * self.Dp

        torsional_stiffness = self.M_app / self.deflected
        linear_stiffness = torsional_stiffness / r ** 2
        return -linear_stiffness

    def measure_deflection(self, Plot_Test=False):
        """Finds the indexes that will be used to compute the stiffness"""

        l = len(self.Planet_Axle_Angle[:, 0])
        i_even = np.arange(0, l - 1, 2).astype(int)
        i_odd = np.arange(1, l, 2).astype(int)

        ideal_angle = self.angle[i_odd][1:]  # The actual angle, touching under no load
        deflected_angle = self.angle[i_even][1:]

        deflection_angle = deflected_angle - ideal_angle  # The first value from no load to initial contact is discarded

        if Plot_Test == True:
            plt.figure()
            plt.scatter(ideal_angle, deflection_angle)
            # plt.plot(ideal_angle, deflection_angle)
            plt.xlabel("Ideal, Infinite stiffness angle")
            plt.ylabel("Angle of deflection")

        return ideal_angle, deflection_angle


data_dir = definitions.root + "\\data\\external\\fem\\raw"
#for crack_length_result in ["run_6_0.0mm_planet_angle.txt"]:  # os.listdir(data_dir):
for crack_length_result in ["run_2.json_planet_coarse_correct_geom_planet_angle.txt"]:  # os.listdir(data_dir):
    tvms_obj = TVMS(data_dir + "\\" + crack_length_result, 4, 0.054)  # Still have to sort out stiffness for sun planet interaction. Also perhaps interface with .json files from simulation

    tvms_obj.plot_angle_measurements()
    tvms_obj.plot_linear_stiffness()
    #ideal, deflect = tvms_obj.measure_stiffness(Plot_Test=True)

    #print(tvms_obj.angular_deflection_to_linear_stiffnes(deflect))
