import pickle
import definitions
import src.features.proc_lib as proc
import numpy as np
import matplotlib.pyplot as plt

#  Load the dataset object
# =====================================================================================================================
# filename = "g1_p7_8.8.pgdata"
# filename = "g1_fc_1000.pgdata"
filename = "g1_fc_1000_long.pgdata"

directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

signal = data.dataset["Acc_Carrier"].values
fs = data.info["f_s"]

# t_range = np.linspace(0,10,10000)
# fs = 1/np.average(np.diff(t_range))
#
# f1 = 1; A1 = np.random.randint(0,10)
# f2 = 2*f1; A2 = np.random.randint(0,10)
# f3 = 3*f1; A3 = np.random.randint(0,10)
# f4 = 4*f1; A4 = np.random.randint(0,10)
# signal = A1*np.sin(2*np.pi*f1*t_range) + A2*np.sin(2*np.pi*f2*t_range) + A3*np.sin(2*np.pi*f3*t_range) + A4*np.sin(2*np.pi*f4*t_range)

plt.figure()
plt.plot(signal, label="label")
plt.show()

# FTx = np.fft.fft(signal)  # /length Usually you would have to normalize by length but now it would cancel anyway
# x_cpw = np.fft.ifft(FTx / np.abs(FTx))
x_cpw = data.cepstrum_pre_whitening(signal)

plt.figure()
plt.plot(x_cpw, label="label")
plt.show()
