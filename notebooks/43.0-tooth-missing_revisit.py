import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np
import pywt

filename = "cycle_2_end.pgdata"
# filename = "g1_p0_9.0.pgdata"

directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

# data.plot_rpm_over_time()
print("Average GMF:", data.derived_attributes["GMF_ave"], "Hz")
print("Average FF:", data.PG.FF1(data.info["rpm_sun_ave"] / 60), "Hz")

offset_frac = 0
window_frac = 2 / 62
# Check that order tracking and TSA is in fact working
# data.compute_tsa(offset_frac,window_frac,data.dataset["Acc_Carrier"],plot=True,ordertrack = False)
# data.compute_tsa(offset_frac,window_frac,data.dataset["Acc_Carrier"],plot=True,ordertrack = True)

channel = "Acc_Carrier"


def plot_unfiltered_fft():
    # plt.figure()
    squared = data.dataset[channel].values ** 2
    # plt.plot(squared)
    # plt.vlines(data.derived_attributes["trigger_index_mag"],0,1500000)
    data.plot_fft(squared, data.info["f_s"], plot_gmf=True, plot_ff=True)

odt_sig = data.order
gmf = data.derived_attributes["GMF_ave"]
fault_freq = data.PG.FF1(data.info["rpm_sun_ave"] / 60)
bandwidth = 60
for gear_mesh_harmonic in [1, 2, 3]:
    mid_freq = gmf * gear_mesh_harmonic
    print(mid_freq)

    filter_params = {"type": "band",
                     "low_cut": mid_freq - 0.5 * bandwidth,
                     "high_cut": mid_freq + 0.5 * bandwidth}

    data.filter_column(channel, filter_params)

    # squared = data.dataset["filtered_bp_" + channel].values**2
    squared = data.dataset["filtered_bp_" + channel].values

    # data.plot_fft(squared, data.info["f_s"])#, plot_gmf=True, plot_ff=True)
    freq, mag, phase = data.fft(squared, data.info["f_s"])

    plt.figure()
    plt.plot(freq, mag ** 2, "k")
    maxi = 1
    # plt.vlines(2*mid_freq,0, maxi,"g")
    plt.vlines(mid_freq, 0, maxi, "g")

    # plt.vlines(2*mid_freq + fault_freq,0, maxi,"r")
    # plt.vlines(2*mid_freq - fault_freq,0, maxi,"r")
    plt.vlines(mid_freq + fault_freq, 0, maxi, "r")
    plt.vlines(mid_freq - fault_freq, 0, maxi, "r")

    # plt.ylim(0,200)
    # plt.xlim(2*mid_freq-7*fault_freq,2*mid_freq + 7*fault_freq)
