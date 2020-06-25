import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

import src.features.proc_lib as proc
plt.close("all")

#  Load the dataset object
#filename = "g1t_p0_v8_0.pgdata"
#filename = "cycle_2_end.pgdata"
filename = "g1_fc_1000_long.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

#Plot rpm over time
#data.plot_rpm_over_time()

# Band pass filter the signal
sigprocobj = proc.Signal_Processing()
sigprocobj.info = data.info
sigprocobj.dataset = data.dataset

# filt_params = {"type": "band",
#                "low_cut": 1000,
#                "high_cut": 2000}

#sigprocobj.filter_column("Acc_Carrier",400,2000)
tsa_obj = proc.Time_Synchronous_Averaging()

# Create a TSA object
tsa_obj.info = data.info
tsa_obj.derived_attributes = data.derived_attributes
tsa_obj.dataset = sigprocobj.dataset # Notice that the dataset is exchanged for filtered dataset
tsa_obj.dataset_name = data.dataset_name
tsa_obj.PG = data.PG

filtered = False

offset_frac = (1/62)*(0.0)

if filtered:
    winds = tsa_obj.window_extract(offset_frac, 2*1/62, "Filtered_Acc_Carrier",order_track=True, plot=False)
else:
    winds = tsa_obj.window_extract(offset_frac, 2*1/62, "Acc_Carrier",order_track=True, plot=False)

wind_ave,all_p_teeth = tsa_obj.window_average(winds,plot=True)
ave_in_order,planet_gear_rev = tsa_obj.aranged_averaged_windows(wind_ave,plot=True)


