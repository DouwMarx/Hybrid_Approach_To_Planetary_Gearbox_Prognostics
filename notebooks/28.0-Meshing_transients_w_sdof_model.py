import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

import src.features.proc_lib as proc
plt.close("all")


#voltage_list = ["9_0","8_8","8_6","8_4"]
voltage_list = ["8_6"]
#voltage_list = ["9_8","9_6","9_4","9_2","9_0","8_8","8_6","8_4","8_2","8_0"]

#g1stats = np.zeros((len(voltage_list),5))
#g1tstats = np.zeros((len(voltage_list),5))


for gear_width in [""]:#,"t"]:
    for voltage,i in zip(voltage_list,range(len(voltage_list))):
#        try:

        filename = "g1" + gear_width + "_p0_v" + voltage + ".pgdata"

        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)

        winds = data.derived_attributes["extracted_windows"][0:10,0:1000]  # Use only a few windows for now

        tobj = proc.TransientAnalysis()
        tobj.info = data.info
        tobj.derived_attributes = data.derived_attributes
        tobj.dataset = data.dataset
        tobj.dataset_name = data.dataset_name

        model_param_stats = tobj.get_sdof_stats_over_all_windows(winds, plot_results=True, plot_checks=True)

        # if gear_width == "":
        #     g1stats[i,0] = rpm_sun_ave
        #     g1stats[i,1:3] = peak_stats
        #     g1stats[i,3:] = freq_stats
        # if gear_width == "t":
        #     g1tstats[i,0] = rpm_sun_ave
        #     g1tstats[i,1:3] = peak_stats
        #     g1tstats[i,3:] = freq_stats
      #  except:
      #      print(gear_width,voltage)

#print(g1stats)
#print(g1tstats)

def plot_stats(g1stats,g1tstats,i):
    if i==1:
        name = "Peak magnitude"

    if i==3:
        name = "Prominent Frequency"

    plt.figure(name)
    for thickness,label in zip([g1stats,g1tstats],["full thickness","half thickness"]):
        plt.errorbar(thickness[:,0],thickness[:,i],thickness[:,i+1],fmt='o',solid_capstyle='projecting', capsize=5,label = label)
    plt.xlabel("Rotational speed")
    plt.ylabel(name)
    plt.legend()
    return

#plot_stats(g1stats,g1tstats,1)
#plot_stats(g1stats,g1tstats,3)
