import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions
import src.features.proc_lib as proc
import scipy.signal as scisig
import scipy.interpolate as interpolate

plt.close("all")

#  Load the dataset object
# filename = "cycle_2_end.pgdata"
#filename = "g1_p0_v9_0.pgdata"
filename = "g1t_p0_v9_0.pgdata"
# filename = "cycle_6_end.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

print("Average RPM of motor over the course of the test as based on 1XPR magnetic encoder", data.info["rpm_sun_ave"])
fp_ave = data.PG.f_p(data.info["rpm_carrier_ave"] / 60)
print("Average planet gear rotational speed", fp_ave, "Rev/s")

print("Planet pass period", 1 / data.derived_attributes["PPF_ave"])
print("Gear mesh period", 1 / data.derived_attributes["GMF_ave"])

# Plot RPM over time
# data.plot_rpm_over_time()

# plt.figure()
# data.plot_time_series_4("Acc_Sun")

# plt.figure()
# data.plot_time_series("Acc_Sun")
# plt.vlines(data.derived_attributes["trigger_time_mag"], -150, 150, "r")

# plt.figure()
# data.plot_time_series("Acc_Carrier")
# plt.vlines(data.derived_attributes["trigger_time_mag"], -150, 150, "r")

winds = data.Window_extract(-3600)
print(np.shape(winds))


def get_peaks(signal, plot=False):
    samples_per_gearmesh = data.info["f_s"] * (1 / data.derived_attributes["GMF_ave"])

    indices, properties = scisig.find_peaks(signal,
                                            height=30,
                                            distance=samples_per_gearmesh * 0.7)

    peaks = signal[indices]

    if plot:
        wind_len = np.shape(winds)[1]
        ts = 1 / data.info["f_s"]
        time_end = wind_len * ts
        trange = np.linspace(0, time_end, wind_len)
        plt.figure()
        plt.plot(trange, signal)
        ind, peaks, prop = get_peaks(signal)
        plt.scatter(ind * ts, peaks, marker="x", c="black")

        plt.figure("Time between peaks")
        plt.hist(np.diff(ind) / data.info["f_s"])
        plt.xlabel("Time between extracted mesh transient peaks [s]")
        plt.ylabel("Frequency of occurrence")

        plt.figure("Peak value")
        plt.hist(peaks)
        plt.xlabel("Vibration magnitude [mg]")
        plt.ylabel("Frequency of occurrence")

    return indices, peaks, properties


def get_transients(signal, plot=False):
    """
    Extracts the gear mesh transients for a given signal
    Parameters
    ----------
    signal

    Returns
    -------

    """
    #samples_before = int(0.001 * 38600)
    #samples_after = int(0.001 * 38600)
    samples_before = int(0.0002 * 38600)
    samples_after = int(0.0008 * 38600)

    ind, peaks, prop = get_peaks(signal)

    t_gm = np.diff(ind) / data.info["f_s"]

    ind = ind[1:-1]  # eliminate first and final peak
    ind_start = ind - samples_before
    ind_end = ind + samples_after

    transients_store = np.zeros((len(ind), samples_after + samples_before))
    for i in range(len(ind)):
        transient = signal[ind_start[i]:ind_end[i]]
        transients_store[i, :] = transient

    if plot:
        ts = 1 / data.info["f_s"]
        time_end = (samples_before + samples_after) * ts
        time_range = np.linspace(0, time_end, samples_before + samples_after)
        plt.figure()
        plt.plot(time_range, transients_store.T)

    return transients_store, peaks, t_gm


def get_stats_over_all_windows(windows, plot=False):
    all_peaks = np.array([])
    all_t_gm = np.array([])
    all_prom_freqs = np.array([])
    for window in windows:
        trans, peaks, t_gm = get_transients(window)

        prom_freqs = get_prominent_freqs_over_all_transients(trans)

        all_peaks = np.hstack((all_peaks, peaks))
        all_t_gm = np.hstack((all_t_gm, t_gm))
        all_prom_freqs = np.hstack((all_prom_freqs,prom_freqs))

    if plot:
        plt.figure("Time between peaks")
        plt.hist(all_t_gm)
        plt.xlabel("Time between extracted mesh transient peaks [s]")
        plt.ylabel("Frequency of occurrence")

        plt.figure("Transient peak value")
        plt.hist(all_peaks)
        plt.xlabel("Transient Peak Vibration Magnitude [mg]")
        plt.ylabel("Frequency of occurrence")

        plt.figure("Most prominent frequency")
        plt.hist(all_prom_freqs)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Frequency of occurrence")
    return


def get_prominent_freqs_over_all_transients(transients, plot=False):
    """
    Get the most prominent frequency of vibration of the transient
    Parameters
    ----------
    sig

    Returns
    -------

    """

    # Remove peaks in frequency spectrum with period longer than transient length
    length = np.shape(transients)[1]
    t_transient = length / data.info["f_s"]
    f_lower_cutoff = 2.5 * 1 / t_transient



    #length = 1000
    fs = data.info["f_s"]
    #fs = 0.0001
    freq = np.fft.fftfreq(length, 1 / fs)[0:int(length / 2)]

    cut_off_index = np.argmax(np.diff(np.sign(freq-f_lower_cutoff))) +1

    fft_mag_store = np.zeros((np.shape(transients)[0],int(length/2)))


    for sig,i in zip(transients,range(len(transients))):
        # Do FFT on signal segment (transient)
        d = sig

      #  interp_func = interpolate.interp1d(np.linspace(0,1,len(sig)), sig, kind="cubic")
     #   d = interp_func(np.linspace(0,1,length))

        Y = np.fft.fft(d) / length
        magnitude = np.abs(Y)[0:int(length / 2)]
        #phase = np.angle(Y)[0:int(length / 2)]

        fft_mag_store[i, :] = magnitude

    spectra_peak_locs = np.argmax(fft_mag_store[:, cut_off_index:], axis=1)
    most_promiment_freqs = freq[cut_off_index:][spectra_peak_locs]


    if plot:
        plt.figure("Frequency spectra of transients")
        plt.plot(freq, fft_mag_store.T)
        plt.scatter(freq, fft_mag_store.T[:,0])
        plt.vlines(f_lower_cutoff, np.min(fft_mag_store), np.max(fft_mag_store)
                   )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        plt.figure("Most prominent frequency in transient")
        plt.hist(most_promiment_freqs)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Frequency of occurrence")

    return most_promiment_freqs


n = np.random.randint(0, 50)
sig = winds[n, :]
# ind, peak, prop = get_peaks(sig, plot=False)
trans, peaks, t_gm = get_transients(sig, plot=True)

get_prominent_freqs_over_all_transients(trans, plot=True)

get_stats_over_all_windows(winds, plot=True)
