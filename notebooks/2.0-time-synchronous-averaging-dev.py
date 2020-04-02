import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

import src.features.proc_lib as proc
plt.close("all")

#  Load the dataset object
filename = "g1_p0_v10_0.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)




time = data.dataset["Time"].values
mag = data.dataset["1PR_Mag_Pickup"].values
acc = data.dataset["Acc_Sun"].values

indexes = data.derived_attributes["trigger_index_mag"]


RPM = data.info["rpm_carrier_ave"]
print("Average RPM of motor over the course of the test as based on 1XPR magnetic encoder", np.average(RPM)*data.PG.GR)
fc_ave = 1/(RPM/60)

fp_ave = proc.Bonfiglioli.f_p(fc_ave)

print("Average planet gear speed",fp_ave,"Rev/s")

#winds = data.Window_extract()

#aves = proc.Time_Synchronous_Averaging.Window_average(winds,12)

#mesh_seq = list(np.ndarray.astype(np.array(proc.Bonfiglioli.Meshing_sequence())/2,int))
#arranged, all_together = proc.Time_Synchronous_Averaging.Aranged_averaged_windows(aves,mesh_seq)

#plt.figure()
#plt.plot(all_together)
#plt.vlines(np.arange(12)*4963, -4000, 4000, zorder = 10)

