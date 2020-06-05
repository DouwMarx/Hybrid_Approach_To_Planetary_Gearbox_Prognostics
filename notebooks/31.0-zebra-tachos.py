import matplotlib.pyplot as plt
import numpy as np
import src.features.proc_lib as proc
import definitions
import pickle

# Load the dataset object
filename = "g1_p0_v9_0.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
   data = pickle.load(filename)


trig_lvl = 5
slope = 1
PPR = 100
New_fs = data.info["f_s"]

fine_res_rpm, t_rpm, ave_rpm = data.getrpm("Tacho_Carrier",trig_lvl,slope,PPR,New_fs)
#tsa = data.Compute_TSA(0, plot=True)

data.plot_rpm_over_time()
plt.plot(t_rpm,fine_res_rpm)

plt.figure()
plt.vlines(data.derived_attributes["trigger_time_mag"],0,6)
plt.plot(data.dataset["Time"].values,data.dataset["Tacho_Carrier"].values)
