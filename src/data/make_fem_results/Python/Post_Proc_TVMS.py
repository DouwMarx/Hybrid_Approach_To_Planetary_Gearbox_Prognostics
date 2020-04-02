# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:27:28 2020

@author: u15012639
"""

from py_mentat import *
import math

#name = 'Mr_1...Mp1...a_2.5n_100_job1'


def open_file(name):
    py_send("*post_open ..\\Run_Dir\\" + name + ".t16")
    py_send("*post_next")
    py_send("*fill_view")

def angle_pos_history_plot(name):
    "Makes a history plot of the angular displacement of the planet axle rigid body"
    py_send("*history_collect 0 999999999 1")
    py_send("*history_clear")
    py_send("*prog_option history_plot:data_carrier_type_x:global")
    py_send("*set_history_global_variable_x Time")

    py_send("*prog_option history_plot:data_carrier_type_y:cbody")  # This refers to y axis
    py_send("*set_history_data_carrier_cbody_y Carrier_Axle")
    py_send("*set_history_cbody_variable_y Angle Pos")

    py_send("*history_add_curve")
    py_send("*history_fit")

    py_send("*history_write ..\\Results\\" + name + "_angle" + ".txt yes")
    return


def main():
    #py_send("*post_close")
    for crack in range(1,4):

        mesh_name = "m1_a" + str(crack) + "mm"
        n_increments = 10
        name = "Mr_1" + "..." + "Mp_" + mesh_name + "..." + "a_" + str(2.5) + "n_" + str(n_increments) + "_job1"
        open_file(name)


        angle_pos_history_plot(name)

    return

if __name__ == "__main__":
    py_connect('',40007)
    main()
    py_disconnect()