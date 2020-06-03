import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

import src.features.proc_lib as proc
from tqdm import tqdm
plt.close("all")

#  Load the dataset object
#filename = "cycle_2_end.pgdata"
#filename = "g1_p0_v9_0.pgdata"
#filename = "cycle_6_end.pgdata"
#directory = definitions.root + "\\data\\processed\\" + filename
#with open(directory, 'rb') as filename:
#    data = pickle.load(filename)

#plt.figure()
#data.plot_time_series("Acc_Sun")

#data.plot_trigger_times_test()


#RPM = data.info["rpm_carrier_ave"]
#print("Average RPM of motor over the course of the test as based on 1XPR magnetic encoder", data.info["rpm_sun_ave"])
#fp_ave = data.PG.f_p(data.info["rpm_carrier_ave"]/60)
#print("Average planet gear rotational speed", fp_ave, "Rev/s")

#print("Planet pass period", 1/data.derived_attributes["PPF_ave"])
#print("Gear mesh period", 1/data.derived_attributes["GMF_ave"])

#tsa = data.Compute_TSA(0, plot=True)

#data.plot_rpm_over_time()

#plt.figure()
#data.plot_time_series_4("Acc_Sun")

#plt.figure()
#data.plot_time_series("Acc_Sun")

#plt.vlines(data.derived_attributes["trigger_time_mag"],-150,150,"r")
#plt.figure()
#data.plot_time_series("Acc_Carrier")
#plt.vlines(data.derived_attributes["trigger_time_mag"],-150,150,"r")
# filename = "cycle_2_end.pgdata"
# filename = "g1_p0_v9_0.pgdata"
filename = "g1t_p0_v9_0.pgdata"

#voltage_list = ["9_0","8_8","8_6","8_4"]
#voltage_list = ["8_6"]
voltage_list = ["9_8","9_6","9_4","9_2","9_0","8_8","8_6","8_4","8_2","8_0"]

g1stats = np.zeros((len(voltage_list),19))
g1tstats = np.zeros((len(voltage_list),19))


for gear_width in ["","t"]:
    for voltage,i in tqdm(zip(voltage_list,range(len(voltage_list))),total=len(voltage_list)):
#        try:

        filename = "g1" + gear_width + "_p0_v" + voltage + ".pgdata"

        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)

        print_stats = False
        if print_stats == True:
            print("Average RPM of motor over the course of the test as based on 1XPR magnetic encoder", data.info["rpm_sun_ave"])
            fp_ave = data.PG.f_p(data.info["rpm_carrier_ave"] / 60)
            print("Average planet gear rotational speed", fp_ave, "Rev/s")
            print("Planet pass period", 1 / data.derived_attributes["PPF_ave"])
            print("Gear mesh period", 1 / data.derived_attributes["GMF_ave"])


        winds = data.derived_attributes["extracted_windows"]#[0:10,0:1000]

        tobj = proc.TransientAnalysis()
        tobj.info = data.info
        tobj.derived_attributes = data.derived_attributes
        tobj.dataset = data.dataset
        tobj.dataset_name = data.dataset_name

        peak_stats, freq_stats = tobj.get_peak_freq_stats_over_all_windows(winds, plot_results=True, plot_checks=False)
        model_param_stats = tobj.get_sdof_stats_over_all_windows(winds, plot_results=False, plot_checks=False)

        rpm_sun_ave = data.info["rpm_sun_ave"]

        if gear_width == "":
            g1stats[i,0] = rpm_sun_ave
            g1stats[i,1:4] = peak_stats
            g1stats[i,4:7] = freq_stats
            g1stats[i,7:10] = model_param_stats[:,0] #zeta
            g1stats[i,10:13] = model_param_stats[:,1] # omega_n
            g1stats[i,13:16] = model_param_stats[:,2] # d0
            g1stats[i,16:] = model_param_stats[:,3] # v0

        if gear_width == "t":
            g1tstats[i,0] = rpm_sun_ave
            g1tstats[i,1:4] = peak_stats
            g1tstats[i,4:7] = freq_stats
            g1tstats[i, 7:10] = model_param_stats[:, 0]  # zeta
            g1tstats[i, 10:13] = model_param_stats[:, 1]  # omega_n
            g1tstats[i, 13:16] = model_param_stats[:, 2]  # d0
            g1tstats[i, 16:] = model_param_stats[:, 3]  # v0
    #  except:
    #      print(gear_width,voltage)

#print(g1stats)
#print(g1tstats)

def plot_stats(g1stats,g1tstats,i):
    if i==1:
        name = "Peak magnitude [mg]"

    if i==4:
        name = "Prominent Frequency"

    if i==7:
        name = "zeta"

    if i==10:
        name = "omega_n [rads/s]"

    if i==13:
        name = "d0 [m]"

    if i==16:
        name = "v0 [m/s]"

    plt.figure(name)
    for thickness,label in zip([g1stats,g1tstats],["full thickness","half thickness"]):
        plt.errorbar(thickness[:,0],thickness[:,i],thickness[:,i+1],fmt='o',solid_capstyle='projecting', capsize=5,label = label)
    plt.xlabel("Rotational speed")
    plt.ylabel(name)
    plt.legend()

    plt.figure(name + "median")
    for thickness,label in zip([g1stats,g1tstats],["full thickness","half thickness"]):
        plt.scatter(thickness[:,0],thickness[:,i+2], label = label)
    plt.xlabel("Rotational speed")
    plt.ylabel(name + "median")
    plt.legend()
    return

plot_stats(g1stats,g1tstats,1)
plot_stats(g1stats,g1tstats,4)
plot_stats(g1stats,g1tstats,7)
plot_stats(g1stats,g1tstats,10)
plot_stats(g1stats,g1tstats,13)
plot_stats(g1stats,g1tstats,16)
