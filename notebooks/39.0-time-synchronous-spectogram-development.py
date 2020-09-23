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
nperseg = 50
f, t, sxx = data.spectrogram(signal, fs,nperseg,plot=False)
specwinds = data.spec_window_extract(0,2/62,sxx)
ave, allperteeth= data.spec_window_average(specwinds)
data.spec_aranged_averaged_windows(ave,t,f,plot=True)

# tooth1 = ave[1]
# plt.figure()
# plt.pcolormesh(t[0:np.shape(tooth1)[1]], f, tooth1)#, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim([0,6000])
# plt.show()
