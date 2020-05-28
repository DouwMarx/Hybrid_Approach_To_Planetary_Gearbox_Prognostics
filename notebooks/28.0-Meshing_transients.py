import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

import src.features.proc_lib as proc
plt.close("all")

#  Load the dataset object
#filename = "cycle_2_end.pgdata"
filename = "g1_p0_v9_0.pgdata"
#filename = "cycle_6_end.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

#plt.figure()
#data.plot_time_series("Acc_Sun")

#data.plot_trigger_times_test()


#RPM = data.info["rpm_carrier_ave"]
print("Average RPM of motor over the course of the test as based on 1XPR magnetic encoder", data.info["rpm_sun_ave"])
fp_ave = data.PG.f_p(data.info["rpm_carrier_ave"]/60)
print("Average planet gear rotational speed", fp_ave, "Rev/s")

print("Planet pass period", 1/data.derived_attributes["PPF_ave"])
print("Gear mesh period", 1/data.derived_attributes["GMF_ave"])

#tsa = data.Compute_TSA(0, plot=True)

data.plot_rpm_over_time()

plt.figure()
data.plot_time_series_4("Acc_Sun")

plt.figure()
data.plot_time_series("Acc_Sun")

plt.vlines(data.derived_attributes["trigger_time_mag"],-150,150,"r")
plt.figure()
data.plot_time_series("Acc_Carrier")
plt.vlines(data.derived_attributes["trigger_time_mag"],-150,150,"r")
