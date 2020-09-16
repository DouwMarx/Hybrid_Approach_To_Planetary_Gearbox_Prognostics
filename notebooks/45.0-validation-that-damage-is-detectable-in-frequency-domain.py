import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import pywt

# damaged_filename = "cycle_2_end.pgdata"
# healthy_filename = "g1_p0_v8_8.pgdata"

# damaged_filename = "g1_p7_8.8.pgdata"
# d_mid_filename = "g1_p5_8.8.pgdata"
# h_mid_filename = "g1_p3_9.0.pgdata"
# healthy_filename = "g1_p0_9.0.pgdata"


def get_gearmesh(data, channel, gear_mesh_harmonics, bandwidth_as_fraction_of_fundamental):
    """
    Order track and filter around gearmesh frequencies
    Parameters
    ----------
    data
    channel
    gear_mesh_harmonics
    bandwidth_as_fraction_of_fundamental

    Returns
    -------

    """
    # Start by order tracking the data
    signal = data.dataset[channel].values
    odt_time, odt_sig, samples_per_rev = data.order_track(signal)

    gear_mesh_bands = np.zeros((len(gear_mesh_harmonics), len(odt_sig)))
    for i, harmonic in enumerate(gear_mesh_harmonics):
        mid_order = harmonic * data.PG.Z_r  # GMF = rotation speed * n tooth ring

        # Filter around gearmesh
        band_top = mid_order + 0.5 * bandwidth_as_fraction_of_fundamental * data.PG.Z_r
        band_bot = mid_order - 0.5 * bandwidth_as_fraction_of_fundamental * data.PG.Z_r

        fsig = data.filter_band_pass_order(odt_sig, band_bot, band_top, samples_per_rev)

        gear_mesh_bands[i, :] = fsig

    return gear_mesh_bands, samples_per_rev


def squared_spectrum_at_harmonics(dataset_list, gearmesh_harmonics):
    fig, axs = plt.subplots(len(gearmesh_harmonics))
    fig.text(0.5, 0.04, 'n x GMF', ha='center')
    fig.text(0.04, 0.5, 'Squared magnitude', va='center', rotation='vertical')

    band_width_measure = 0.5
    border = 0.4
    cpw = False
    scaling_options = [None, "relative_to_max", "log", "log_de_mean_squared", "poly"]
    scaling = scaling_options[0]
    hcount = 0
    dcount = 0
    color_d = ["lightcoral", "red", "darkred", "tomato", "chocolate", "saddlebrown"]
    color_h = ["forestgreen", "limegreen", "springgreen","olive","greenyellow","seagreen"]

    for data in dataset_list:
        try:
            heath = int(data.dataset_name[4])
        except:
            heath = "string"
        if heath == 0:
            color = color_h[hcount]
            hcount += 1
        else:
            color = color_d[dcount]
            dcount += 1

        bands, spr = get_gearmesh(data, "Acc_Carrier", gearmesh_harmonics, band_width_measure)

        for i, band in enumerate(bands):
            if cpw:
                freq, mag, phase = data.fft(data.cepstrum_pre_whitening(band), spr / 62)
            else:
                freq, mag, phase = data.fft(band, spr / 62)

            if scaling == None:
                mag = mag**2
            if scaling == "relative_to_max":
                mag = (mag / np.max(mag)) ** 2
            if scaling == "log":
                mag =np.log((mag) ** 2)
            if scaling == "log_de_mean_squared":
                log_of_squared = np.log((mag) ** 2)
                de_meaned = log_of_squared - np.mean(log_of_squared)
                mag = de_meaned
            if scaling == "poly":
                log_of_squared = np.log((mag) ** 2)
                x = np.linspace(0, 100, len(log_of_squared))
                model = np.polyfit(x, log_of_squared, 9)
                predicted = np.polyval(model, x)
                de_meaned = log_of_squared - predicted
                mag = de_meaned

            if heath == 0:
                axs[i].plot(freq, mag/np.max(mag), color=color)
            else:
                axs[i].plot(freq, -mag/np.max(mag) , color=color)
            axs[i].set_xlim(gearmesh_harmonics[i] - border, gearmesh_harmonics[i] + border)
    fig.legend(["sun rpm: " + str(int(x.info["rpm_sun_ave"])) + " RPM" for x in dataset_list])  # Label data by rpm
    fig.show()
    return


def squared_envelope_spectrum(dataset_list, gearmesh_harmonics):
    fig, axs = plt.subplots(len(gearmesh_harmonics))
    fig.text(0.5, 0.04, 'n x FF', ha='center')
    fig.text(0.04, 0.5, 'Squared magnitude', va='center', rotation='vertical')

    band_width_measure = 0.5
    border = 0.4
    cpw = False
    scaling_options = [None, "relative_to_max", "log", "log_de_mean_squared", "poly"]
    scaling = scaling_options[0]
    hcount = 0
    dcount = 0
    color_d = ["lightcoral", "red", "darkred", "tomato", "chocolate", "saddlebrown"]
    color_h = ["forestgreen", "limegreen", "springgreen","olive","greenyellow","seagreen"]

    for data in dataset_list:
        try:
            heath = int(data.dataset_name[4])
        except:
            heath = "string"

        if heath == 0:
            color = color_h[hcount]
            hcount += 1
        else:
            color = color_d[dcount]
            dcount += 1

        bands, spr = get_gearmesh(data, "Acc_Carrier", gearmesh_harmonics, band_width_measure)

        for i, band in enumerate(bands):
            if cpw:
                freq, mag, phase = data.fft(data.cepstrum_pre_whitening(band**2), 24*spr / 62)
            else:
                freq, mag, phase = data.fft(band**2, 24*spr / 62)

            if scaling == None:
                mag = mag
            if scaling == "relative_to_max":
                mag = (mag / np.max(mag)) ** 2
            if scaling == "log":
                mag =np.log((mag) ** 2)
            if scaling == "log_de_mean_squared":
                log_of_squared = np.log((mag) ** 2)
                de_meaned = log_of_squared - np.mean(log_of_squared)
                mag = de_meaned
            if scaling == "poly":
                log_of_squared = np.log((mag) ** 2)
                x = np.linspace(0, 100, len(log_of_squared))
                model = np.polyfit(x, log_of_squared, 9)
                predicted = np.polyval(model, x)
                de_meaned = log_of_squared - predicted
                mag = de_meaned

            if heath == 0:
                axs[i].plot(freq, mag/np.max(mag), color=color)
            else:
                axs[i].plot(freq, -mag/np.max(mag) , color=color)
            axs[i].set_xlim(0,5)
            axs[i].set_ylabel(str(gear_mesh_harmonics[i]))
    fig.legend(["sun rpm: " + str(int(x.info["rpm_sun_ave"])) + " RPM" for x in dataset_list])  # Label data by rpm
    fig.show()
    return




# d_list = []
# for gear in ["g1_p7_", "g1_p0_"]:
#     for voltage in ["9.0", "9.2", "9.4", "9.6", "9.8", "8.8"]:
#     #for voltage in ["9.0" ,"9.2","9.4"]:
#         filename = gear + voltage + ".pgdata"
#         directory = definitions.root + "\\data\\processed\\" + filename
#         with open(directory, 'rb') as filename:
#             data = pickle.load(filename)
#             d_list.append(data)

d_list = []
for filename in ["cycle_2_end.pgdata", "g1_p0_v8_8.pgdata"]:
        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)
            d_list.append(data)

# gear_mesh_harmonics = np.array([1, 2, 3, 4, 5])
for gear_mesh_harmonics in [np.array([i,i+1]) for i in range(1,10,2)]:
    squared_spectrum_at_harmonics(d_list, gear_mesh_harmonics)
    #squared_envelope_spectrum(d_list, gear_mesh_harmonics)

