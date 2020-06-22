import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

import src.features.proc_lib as proc
plt.close("all")

#  Load the dataset object
filename = "g1t_p0_v8_0.pgdata"
#filename = "cycle_5_end.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

data.plot_rpm_over_time()

# Band pass filter the signal
sigprocobj = proc.Signal_Processing()
sigprocobj.info = data.info
sigprocobj.dataset = data.dataset
#sigprocobj.filter_column("Acc_Carrier", 38400/13, 38400/9)
tsa_obj = proc.Time_Synchronous_Averaging()

# Create a TSA object
tsa_obj.info = data.info
tsa_obj.derived_attributes = data.derived_attributes
tsa_obj.dataset = sigprocobj.dataset # Notice that the dataset is exchanged for filtered dataset
tsa_obj.dataset_name = data.dataset_name
tsa_obj.PG = data.PG

offset_frac = (1/62)*(0.0)
#winds = tsa_obj.window_extract(offset_frac, 2*1/62, "Filtered_Acc_Carrier", plot=False)
winds = tsa_obj.window_extract(offset_frac, 2*1/62, "Acc_Carrier", plot=False)
wind_ave,all_p_teeth = tsa_obj.window_average(winds,plot=True)

#np.save("Synchronized_samples",all_p_teeth[:,4,60:216].T)
#tsa = data.Compute_TSA(0,3/62, plot=True)
#data.plot_rpm_over_time()

#plt.figure()
#plt.plot(winds.T)

