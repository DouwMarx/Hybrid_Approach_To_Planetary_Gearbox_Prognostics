import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np
import pywt

channel = "Acc_Carrier"
"""
for g2_p5_9.0 it appears that the interesting order band is between 100 and 300 carrier orders.
Since the average carrier period is 0.3354s, the frequency band is around 300-900Hz  

it appears as if the fault is only detectable at high rotational speeds? 
"""
for gear in ["g2_p5_"]:
    for voltage in ["9.0"]:
        filename = gear + voltage + ".pgdata"
        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)

# filename = "g1_fc_1000_long.pgdata"
# directory = definitions.root + "\\data\\processed\\" + filename
# with open(directory, 'rb') as filename:
#     data = pickle.load(filename)


signal = data.dataset["Acc_Carrier"].values
fs = data.info["f_s"]
time = data.dataset["Time"]
signal = data.dataset["Acc_Carrier"]

scales = np.logspace(1.6, 2.3, num=10, dtype=np.int32) # Interesting low frequencies
#scales = np.logspace(0.3, 2.4, num=50, dtype=np.int32) # Full spectrum
#scales = np.logspace(1, 2.4, num=50, dtype=np.int32) # Fullish spectrum
#scales = np.logspace(1, 1.6, num=50, dtype=np.int32) # High Band

#scales = np.linspace(8,350, num=30, dtype=np.int32)
#scales = 1/np.linspace(0.03,0.003, num=50)

tnew, interp_sig, samples_per_rev = data.order_track(signal)
fs = samples_per_rev

waveletname='cmor1.5-1.0'
coefficients, frequencies = pywt.cwt(interp_sig, scales, waveletname, 1/fs)
print("Carrier orders used", frequencies)

power = (abs(coefficients))**2
#power = np.log(power)
# Plot the full scaleogram
# plt.contourf(tnew,frequencies,power)
# plt.vlines(data.derived_attributes["trigger_time_mag"],0,3000)

specto_info = {"interp_spec": power,
               "samples_per_rev": samples_per_rev,
               "total_samples": len(tnew)}

specwinds = data.spec_window_extract(0,  10/ 62, specto_info, order_track=True)
ave, allperteeth = data.spec_window_average(specwinds)
data.spec_aranged_averaged_windows(ave, tnew, frequencies, plot=True, plot_title_addition="order_tracked " + channel)