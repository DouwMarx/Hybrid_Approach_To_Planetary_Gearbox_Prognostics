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

def read_text_file(file):
    a = np.loadtxt(file, skiprows=9)
    return a


data_dir = definitions.root + "\\data\\external\\fem\\raw"
# sim_name = "run_10_planet_3mm-global-0.1mm-edge_20200904"
#sim_name = "run_12_planet_2mm-global-0.05mm-edge_20200918"
sim_name = "run_16_planet_2mm-global_0.05-edge_20201007_crack_refine_0.03"

measured_dict = {}
for measured_variable in ["_crack_length", "_SIF_mode1", "_fatigue_count"]:
# measured_variable = "_crack_length"
    measured_file = read_text_file(data_dir + "\\" + sim_name + measured_variable + ".txt")
    time = measured_file[:,0]
    measured_data = measured_file[:, 1]
    measured_dict.update({measured_variable: measured_data})
    measured_dict.update({"_time": time})

def plot_all_gathered_data():
    plt.figure()
    plt.plot(measured_dict["_crack_length"], measured_dict["_SIF_mode1"])


    plt.figure()
    plt.plot(measured_dict["_time"], measured_dict["_crack_length"])

    plt.figure()
    uneven_ids = range(1,len(measured_dict["_time"]),2)
    plt.plot(measured_dict["_time"][uneven_ids], measured_dict["_SIF_mode1"][uneven_ids])

    plt.figure()
    plt.plot(measured_dict["_time"], measured_dict["_fatigue_count"])
    return

def save_figure(name):
    """Used to save a figure to a single directory"""
    repos = os.path.abspath(os.path.join(definitions.root, os.pardir))
    fig_save_path = repos + "\\fem_images\\" + name + ".pdf"  # Path where these plots are saved
    plt.savefig(fig_save_path)
    return


def plot_sifs(measured_crack_length, measured_SIF):
    """ inputs include intermediate steps"""
    plt.figure()
    # Use only the uneven steps where values are updated
    # uneven_ids = range(1, len(measured_dict["_time"]),2)[0:-1]
    # crack = measured_crack_length[uneven_ids]
    # SIF = measured_SIF[uneven_ids]

    crack = measured_crack_length[1:]
    SIF = measured_SIF[1:]
    # Make a polynomial fit

    order =6
    pfit = np.polyfit(crack,SIF,order) # TODO: Save this model for use in particle filter
    pmod = np.poly1d(pfit)
    print(pmod)
    crack_fine = np.linspace(np.min(crack),np.max(crack))
    model_predictions = pmod(crack_fine)

    #plt.title("SIF vs a")
    plt.scatter(crack, SIF, label="data points from FEM", c="k", marker="x")
    plt.plot(crack_fine, model_predictions, label = "Polynomial model")
    plt.xlabel("Crack length, a [mm]")
    plt.ylabel(r"$\Delta K_I$ [$MPa.mm^{0.5}$]")
    x = np.median(crack)*0.3
    y = 0 #np.min(SIF)*0.9
    text = r"$\Delta K_{I} = "
    for i in range(order):
        text = text + str(round(pmod.coeffs[i],2)) + "a^" + str(order-i) + " + "

    text+= str(round(pmod.coeffs[-1],2)) + "$"
    plt.text(x,y,text,fontsize=7)
    save_figure("SIF_vs_a")
    plt.legend()
    return

#plot_sifs(measured_dict["_crack_length"],measured_dict["_SIF_mode1"])
# plot_all_gathered_data()


measured_file = read_text_file(data_dir + "\\20201008sifs.txt")
crack_length = measured_file[:,0][0:-5] #
sifs = measured_file[:, 1][0:-5] # Discard the last 4 datapoints

plot_sifs(crack_length,sifs)
