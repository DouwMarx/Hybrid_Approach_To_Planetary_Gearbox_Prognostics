import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import Proc_Lib
import importlib
import importlib
import Proc_Lib as proc

importlib.reload(Proc_Lib)



path = r"C:\Users\douwm\Google Drive\Meesters\Meeting_Preparations\Date_Here"



plt.close("all")
# Pump Fan
filename = "Pump_Only"
dir = 'D:\h5_datasets' + "\\" + filename + ".h5"

df = pd.read_hdf(dir)
fs = proc.sampling_rate(df["Time"].values)

plt.figure()
proc.fftplot(df["Acc_Carrier"].values, fs)
proc.fftplot(df["Torque"].values, fs)

plt.title("Pump cooling Fan")
plt.xlim(0,3000)



# Motor Fan
filename = "Fan_No_Rotation_2"
dir = 'D:\h5_datasets' + "\\" + filename + ".h5"

df = pd.read_hdf(dir)

plt.figure()
proc.fftplot(df["Acc_Carrier"].values, fs)
plt.title("Motor Cooling Fan")
plt.xlim(0,600)

# Everything without motor rotating
filename = "Fan_Pump_Motor_fan on"
dir = 'D:\h5_datasets' + "\\" + filename + ".h5"

df = pd.read_hdf(dir)

plt.figure()
proc.fftplot(df["Acc_Carrier"].values, fs)
plt.title("Pump Fan, Motor Fan, Cooling Fan")
plt.xlim(0,2000)
full_path = path + "\\" + "non_gear_components.pdf"
plt.savefig(full_path)

