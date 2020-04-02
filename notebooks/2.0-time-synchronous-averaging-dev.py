import sys


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import Proc_Lib
import importlib
importlib.reload(Proc_Lib)

import Proc_Lib as proc




plt.close("all")
# Pump Fan
#filename = "Pump_Only"
filename = "Cycle_4_end"

dir = 'D:\h5_datasets' + "\\" + filename + ".h5"

df = pd.read_hdf(dir)
fs = proc.sampling_rate(df["Time"].values)


time = df["Time"].values
mag = df["1PR_Mag_Pickup"].values
acc = df["Acc_Sun"].values

indexes, timepoints = proc.Planet_Pass_Time(time, mag, Plot_checks=False)


RPM, t_rpm = proc.getrpm(mag,fs,0.5,1,1,fs)
print("Average RPM of motor over the course of the test as based on 1XPR magnetic encoder", np.average(RPM)*5.77)
fc_ave = 1/np.average(RPM/60)

Zp = proc.Bonfiglioli.Z_p
fp_ave = proc.Bonfiglioli.f_p(fc_ave)

print("Average planet gear speed",fp_ave,"Rev/s")

winds = proc.Window_extract(acc,indexes,fp_ave,fs,Zp/2,555)

aves = proc.Window_average(winds,12)

mesh_seq = list(np.ndarray.astype(np.array(proc.Bonfiglioli.Meshing_sequence())/2,int))
arranged, all = proc.Aranged_averaged_windows(aves,mesh_seq)

plt.figure()
plt.plot(all)
plt.vlines(np.arange(12)*4963, -4000, 4000, zorder = 10)

#for pl in range(12):
#    plt.figure()
#    plt.plot(arranged[pl,:])


#plt.figure()
#plt.plot(df["Time"],df["1PR_Mag_Pickup"])
#plt.plot(df["Time"],df["Acc_Sun"])

#proc.fftplot(df["Acc_Carrier"].values, fs)
#plt.title("Pump cooling Fan")
#plt.xlim(0,3000)

# Motor Fan
