import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions
from PyEMD import EMD

import src.features.proc_lib as proc
plt.close("all")

#  Load the dataset object
# =====================================================================================================================
filename = "g1_p7_8.8.pgdata"


directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

#Plot rpm over time
# ======================================================================================================================
# data.plot_rpm_over_time()

# Do signal processing
# ======================================================================================================================
sigprocobj = proc.Signal_Processing()
sigprocobj.info = data.info
sigprocobj.dataset = data.dataset

# Filter
filt_params = {"type": "band",
               "low_cut": 1000,
               "high_cut": 2500}
sigprocobj.filter_column("Acc_Carrier",filt_params)

# Square
sigprocobj.square_column("Acc_Carrier")


# =====================================================================================================================
def get_ot_arranged_tsa(signame):
    order_t, order_sig, samples_per_rev = data.order_track(signame)
    data.derived_attributes.update({"order_track_time": order_t, "order_track_" + signame: order_sig,
                                    "order_track_samples_per_rev": samples_per_rev})

    winds = tsa_obj.window_extract(offset_frac, 2 * 1 / 62, "order_track_" + signame, order_track=True,
                                         plot=False)
    wind_ave,all_p_teeth = tsa_obj.window_average(winds,plot=False)
    ave_in_order,planet_gear_rev = tsa_obj.aranged_averaged_windows(wind_ave,plot=True)
    plt.suptitle("order_track_" + signame)
    return ave_in_order

def compute_and_update_ordertrack(signame):
    order_t, order_sig, samples_per_rev = data.order_track(signame)
    data.derived_attributes.update({"order_track_time": order_t, "order_track_" + signame: order_sig,
                                    "order_track_samples_per_rev": samples_per_rev})
    return

# Create a Signal Processing Object
# =====================================================================================================================
tsa_obj = proc.Time_Synchronous_Averaging()

# Create a TSA object
# =====================================================================================================================
tsa_obj.info = data.info
tsa_obj.derived_attributes = data.derived_attributes
tsa_obj.dataset = sigprocobj.dataset # Notice that the dataset is exchanged for processed dataset
tsa_obj.dataset_name = data.dataset_name
tsa_obj.PG = data.PG

# Order track
# =====================================================================================================================
# compute_and_update_ordertrack("Acc_Carrier")
# compute_and_update_ordertrack("filtered_bp_Acc_Carrier")
# compute_and_update_ordertrack("squared_Acc_Carrier")

print(sigprocobj.dataset.keys())

# Set mag pickup offset
# =====================================================================================================================
offset_frac = (1/62) * 0.0  # Set the offset of the TSA in fractions of a revolution

# For tsa only,  no order track
baseline_winds = tsa_obj.window_extract(offset_frac, 2*1/62, "Acc_Carrier", order_track=False, plot=False)
wind_ave, all_p_teeth = tsa_obj.window_average(baseline_winds, plot=False)
ave_in_order, planet_gear_rev = tsa_obj.aranged_averaged_windows(wind_ave, plot=True)
plt.suptitle("no_order_track_Acc_Carrier")

odtave = get_ot_arranged_tsa("Acc_Carrier")
get_ot_arranged_tsa("filtered_bp_Acc_Carrier")
get_ot_arranged_tsa("squared_Acc_Carrier")



for i in range(6):
    plt.figure("squared spectrum")
    freq, magnitude, phase =sigprocobj.fft(odtave[i,:], data.info["f_s"])
    # freq, magnitude, phase = sigprocobj.fft(odtave.T, data.info["f_s"])
    plt.plot(freq, magnitude**2, label=str(i*2))
    plt.legend()
    plt.title("Squared Spectrum")


    emd = EMD()
    imfs = emd(odtave[i, :])
    for j in range(5):
        plt.figure("EMD " + str(j))
        plt.plot(imfs[j], label=str(i*2))
        plt.legend()
