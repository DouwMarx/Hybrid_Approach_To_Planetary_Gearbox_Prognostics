# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 08:09:07 2020

@author: u15012639
"""

import numpy as np
import matplotlib.pyplot as plt
import definitions
import os
import json

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
        plt.ylabel("Planet Angle [rad]")
        plt.legend()
        return

    def plot_linear_stiffness(self,save_path = None):
        plt.figure()
        print(self.ideal)
        print(self.linear_stiffness)
        plt.scatter(self.ideal[1:], self.linear_stiffness,c="k",marker=".")
        #plt.plot(self.ideal[1:], self.linear_stiffness) # Not sure if first or last element should be excluded
        plt.ylabel("Stiffness [N/m]")
        plt.xlabel("Rotation angle [rad]")

        if save_path:
            plt.savefig(save_path)
        return

    def compute_linear_stiffness(self):
        r = 0.5 * self.Dp

        torsional_stiffness = self.M_app / self.deflected
        linear_stiffness = torsional_stiffness / r ** 2
        return linear_stiffness

    def measure_deflection(self, Plot_Test=False):
        """Finds the indexes that will be used to compute the stiffness"""

        l = len(self.Planet_Axle_Angle[:, 0])
        i_even = np.arange(0, l - 1, 2).astype(int)
        i_odd = np.arange(1, l, 2).astype(int)

        ideal_angle = self.angle[i_odd]#[1:]  # The actual angle, touching under no load
        deflected_angle = self.angle[i_even][1:]

        deflection_angle = deflected_angle - ideal_angle[0:-1]  # The first value from no load to initial contact is discarded

        if Plot_Test == True:
            plt.figure()
            plt.scatter(ideal_angle[0:-1], deflection_angle*1000)
            plt.plot(ideal_angle[0:-1], deflection_angle*1000)
            plt.xlabel("Ideal, Infinite stiffness angle ")
            plt.ylabel("Angle of deflection x 1000")

        return ideal_angle, np.abs(deflection_angle)


data_dir = definitions.root + "\\data\\external\\fem\\raw"


repos = os.path.abspath(os.path.join(definitions.root, os.pardir))

def get_parameters_from_json(run_file_name):
    run_file_path = definitions.root + "\\models\\fem\\run_input_files\\" + run_file_name + ".json"
    print(run_file_path)
    with open(run_file_path) as json_file:
        input_file = json.load(json_file)
    return input_file

        # # Loadcase
        # total_rotation = input["TVMS Properties"]["Load Case"]["Total Rotation"]  # The total angular distance rotated [rad]
        # n_increments = input["TVMS Properties"]["Load Case"]["Number Of Loadsteps"]  # int
        # applied_moment = input["TVMS Properties"]["Load Case"]["Applied Moment"]  # Moment on planet gear [Nm]
        #
        # # Contact
        # friction_coefficient = input["TVMS Properties"]["Contact"][
        #     "Friction Coefficient"]  # Dynamic friction coefficient for lubricated Cast iron on Cast iron https://www.engineeringtoolbox.com/friction-coefficients-d_778.html
        #
        # # Geometry
        # gear_thickness = input["Geometry"]["Gear Thickness"]  # [mm]
        #
        # move_planet_up = input["TVMS Properties"]["Position Adjustment"][
        #     "Move Planet Up"]  # -1.65  # Distance the planet gear should be moved up [mm]
        # rotate_planet = input["TVMS Properties"]["Position Adjustment"][
        #     "Rotate Planet"]  # -1.72 - (360 / 24) * 2  # Angle the planet gear should be rotated [degrees]
        # planet_carrier_pcr = input["Geometry"][
        #     "Planet Carrier Pitch Centre Radius"]  # Pitch Centre Radius of planet carrier axle
        #
        # ring_gear_external_radius = input["Geometry"]["Ring Gear External Radius"]  # External Radius of Ring gear [mm]
        # ring_gear_rotation = input["TVMS Properties"]["Position Adjustment"][
        #     "Ring Gear Rotation"]  # -(360 / 62) * 2  # Angle the ring gear should be rotated
        #
        # planet_axle_radius = input["Geometry"]["Planet Axle Radius"]  # Internal radius of the planet gear [mm]
        #
        # # Material Parameters
        # ##############################################################################################################
        # E = input["Material"]["Young's Modulus"]  # MPa
        # v = input["Material"]["Poisson Ratio"]

#for crack_length_result in ["run_8_1.4mm_planet_angle.txt"]:  # os.listdir(data_dir):
#for crack_length_result in ["run_9_0.0mm_planet_angle.txt"]:  # os.listdir(data_dir):
#for crack_length_result in ["beam_to_beam_half_force_no_fric.txt"]:
# for crack_length_result in ["beam_to_beam_double_force.txt"]:
#for crack_length_result in ["displacement_and_angle_convergence_criteria.txt"]:

# run_name = "run_17"
# crack_length = "3.2mm"
# result_file_name = run_name + "_" + crack_length + "_planet_angle.txt"

# run_name = "run_16"
# crack_length = "health"
# result_file_name = run_name + "_" + crack_length + "_planet_angle.txt"

# run_name = "run_18"
# crack_length = "health"
# result_file_name = run_name + "_" + crack_length + "_planet_angle.txt"

#run_20_planet_0.5mm-global_0.05-edge_20201016_tr_planet_angle.txt

# run_name = "run_20"
# crack_length = "planet_0.5mm-global_0.05-edge_20201016_tr"
# result_file_name = run_name + "_" + crack_length + "_planet_angle.txt"

# Healthy planet
run_name = "run_21"
crack_length = "health_20201016_tr"
result_file_name = run_name + "_" + crack_length + "_planet_angle.txt"

# Cracked planet
# run_name = "run_22"
# crack_length = "3.0mm"
# result_file_name = run_name + "_" + crack_length + "_planet_angle.txt"

# Sun planet
run_name = "run_23"
crack_length = "health_20201016_tr"
result_file_name = run_name + "_" + crack_length + "_planet_angle.txt"

#run_23_health_20201016_tr_planet_angle.txt
params = get_parameters_from_json(run_name)
applied_moment = params["TVMS Properties"]["Load Case"]["Applied Moment"]  # Moment on planet gear [Nm]
print("moment", applied_moment)
pitch_diameter = params["Geometry"]["Pitch Diameter"]  # Internal radius of the planet gear [mm]
print("PD", pitch_diameter)

fig_save_path = repos + "\\fem_images\\linear_tvms_" + run_name + "_" + crack_length + ".pdf"
for crack_length_result in [result_file_name]:
# run_11_1.0mm_planet_angle
#for crack_length_result in ["run_7_1.4mm_planet_angle.txt"]:  # os.listdir(data_dir):
#for crack_length_result in ["run_6_0.0mm_planet_angle.txt"]:  # os.listdir(data_dir):
#for crack_length_result in ["run_2.json_planet_coarse_correct_geom_planet_angle.txt"]:  # os.listdir(data_dir):
    tvms_obj = TVMS(data_dir + "\\" + crack_length_result, applied_moment, pitch_diameter/1000)  # Still have to sort out stiffness for sun planet interaction. Also perhaps interface with .json files from simulation

    tvms_obj.plot_angle_measurements()
    tvms_obj.plot_linear_stiffness(save_path=fig_save_path)
    #ideal, deflect = tvms_obj.measure_stiffness(Plot_Test=True)
    #tvms_obj.measure_deflection(Plot_Test=True)
    #print(tvms_obj.angular_deflection_to_linear_stiffnes(deflect))
