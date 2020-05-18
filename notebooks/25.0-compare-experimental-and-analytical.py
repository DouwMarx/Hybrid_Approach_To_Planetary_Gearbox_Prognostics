import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

#import src.features.proc_lib as proc
plt.close("all")

#  Load the dataset object
filename = "cycle_2_end.pgdata"
#filename = "cycle_6_end.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

#plt.figure()
#data.plot_time_series("Acc_Sun")

#data.plot_trigger_times_test()


RPM = data.info["rpm_carrier_ave"]
print("Average RPM of motor over the course of the test as based on 1XPR magnetic encoder", data.info["rpm_sun_ave"])
fp_ave = data.PG.f_p(data.info["rpm_carrier_ave"]/60)
print("Average planet gear rotational speed", fp_ave, "Rev/s")

tsa = data.Compute_TSA(0, plot=False)

#data.plot_rpm_over_time()

#plt.figure()
#plt.plot(data.derived_attributes["order_track_time"], data.derived_attributes["order_track_signal"])


data.plot_fft(data.dataset["Acc_Sun"],data.info["f_s"], plot_gmf=True)
plt.title("FFT")

data.plot_order_spectrum(data.derived_attributes["order_track_signal"], data.info["f_s"],data.derived_attributes["order_track_samples_per_rev"],plot_gmf=True)
plt.title("Order Spectrum")

data.plot_order_spectrum(tsa, data.info["f_s"], data.derived_attributes["order_track_samples_per_rev"])#, plot_gmf=True)
plt.title("TSA order specturm")
