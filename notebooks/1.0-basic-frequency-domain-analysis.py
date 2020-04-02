import Proc_Lib as proc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import Proc_Lib
import Proc_Lib as proc



importlib.reload(Proc_Lib)

Zr = 62
Zs = 13
Zp = 24

Bonfiglioli = proc.PG(Zr,Zs,Zp)






# Load the appropriate dataset
filename = "Cycle_1_end"
dir = 'D:\h5_datasets' + "\\" + filename + ".h5"
df = pd.read_hdf(dir)

fs = 32000

mag_pickup = df["1PR_Mag_Pickup"].values


#rpm, t = proc.getrpm(mag_pickup,fs,1,1,1,fs)  # Compute the RPM for the input shaft based on the shaft encoder
ave_rpm = np.mean(rpm)  # The average input (low speed side) over the tested interval
print("Average low side speed: ", ave_rpm," RPM")
print("Average motor input high side speed: ", ave_rpm*5.77, "RPM")


# Compute the frequency of rotation of the sun gear shaft (This is the same as the motor speed)
fsun = proc.PG.RPM_to_f(ave_rpm*Bonfiglioli.GR) # Frequency of rotation of sun gear (fast side)
GMF = Bonfiglioli.GMF(fsun)
FF1 = Bonfiglioli.FF1(fsun)
print(FF1)


max_height = 5

plt.figure()
proc.fftplot(df["Acc_Sun"].values, fs)
#plt.title("Freq plot")
plt.vlines(np.arange(1,5)*GMF,0,max_height,'r',zorder=10,label="GMF and Harmonics")
plt.vlines(np.arange(1,4)*FF1,0,max_height,'g',zorder=10,label="FF1 and Harmonics")
plt.xlim(0,6000)
plt.ylim(0,1.5)

plt.figure()
plt.plot(t, rpm)
plt.xlabel("Time [s]")
plt.ylabel("RPM")
plt.title("Rotational speed based on 1PR magnetic encoder")
path = proc.image_save_path + "\\" + "Speed_example.pdf"
plt.savefig(path)