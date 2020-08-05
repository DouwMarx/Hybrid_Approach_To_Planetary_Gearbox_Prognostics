import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np
import pywt

#damaged_filename = "cycle_2_end.pgdata"
#healthy_filename = "g1_p0_v8_8.pgdata"

damaged_filename = "g1_p7_8.8.pgdata"
d_mid_filename = "g1_p5_8.8.pgdata"
h_mid_filename = "g1_p3_9.0.pgdata"
healthy_filename = "g1_p0_9.0.pgdata"

harmonics = [1,3, 9, 11,16]

healthy_directory = definitions.root + "\\data\\processed\\" + healthy_filename
with open(healthy_directory, 'rb') as filename:
    healthy = pickle.load(filename)

damaged_directory = definitions.root + "\\data\\processed\\" + damaged_filename
with open(damaged_directory, 'rb') as filename:
    damaged = pickle.load(filename)

d_mid_directory = definitions.root + "\\data\\processed\\" + d_mid_filename
with open(d_mid_directory, 'rb') as filename:
    d_mid = pickle.load(filename)

h_mid_directory = definitions.root + "\\data\\processed\\" + h_mid_filename
with open(h_mid_directory, 'rb') as filename:
    h_mid = pickle.load(filename)

def compare_sidebands(damaged, healthy, n_harmonics=15, bandwidth_as_fraction_of_gmf=0.1, channel="Acc_Carrier"):
    fig, axs = plt.subplots(n_harmonics)
    for harmonic in range(n_harmonics):
        axs[harmonic].grid()
        fig.text(0.5, 0.04, "n * gear mesh frequency", ha='center')

        for data, name in zip([damaged, healthy], ["damaged", "healthy"]):
            mid_freq = data.derived_attributes["GMF_ave"] * (harmonic + 1)
            bandwidth = bandwidth_as_fraction_of_gmf * data.derived_attributes[
                "GMF_ave"]  # window bandwidth of 20% of gmf

            # Filter around gearmesh
            top_freq = mid_freq + 0.5 * bandwidth
            bot_freq = mid_freq - 0.5 * bandwidth

            filter_params = {"type": "band",
                             "low_cut": bot_freq,
                             "high_cut": top_freq}

            # Filter the data around the gearmesh frequency
            data.filter_column(channel, filter_params)
            odt_time, odt_sig, samples_per_rev = data.order_track(data.dataset["filtered_bp_Acc_Carrier"])
            freq, mag, phase = data.fft(odt_sig,
                                        samples_per_rev / data.PG.Z_r)  # Sampling rate scales frequency to be in terms
            # of synchronous gear mesh frequency

            axs[harmonic].plot(freq, (mag / np.max(mag)) ** 2, label=name)
            axs[harmonic].set_ylabel(str(harmonic + 1))  # "Magnitude^2")
            axs[harmonic].set_xlim((harmonic + 1) - 0.8, (harmonic + 1) + 0.8)
            axs[harmonic].legend(loc=1)
    return


def get_gearmesh(data, channel, gear_mesh_harmonics, bandwidth_as_fraction_of_fundamental):
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


def compare_nat_freqs(data, bandwidth_as_fraction_of_gmf=0.5, channel="Acc_Carrier", n_harmonics=4, plot=False):
    signal = data.dataset[channel].values

    bandwidth = bandwidth_as_fraction_of_gmf * data.derived_attributes["GMF_ave"]

    for harmonic in range(n_harmonics):
        mid_freq = data.derived_attributes["GMF_ave"] * (harmonic + 1)

        # Filter around gearmesh
        top_freq = mid_freq + 0.5 * bandwidth
        bot_freq = mid_freq - 0.5 * bandwidth

        signal = data.filter_band_stop(signal, bot_freq, top_freq)

    # Order track the filtered signal
    odt_time, odt_sig, samples_per_rev = data.order_track(signal)

    if plot:
        freq, mag, phase = data.fft(odt_sig,
                                    samples_per_rev / data.PG.Z_r)  # Sampling rate scales frequency to be in terms
        plt.figure()
        plt.plot(freq, mag ** 2)
        # # Filter the data around the gearmesh frequency
        # data.filter_column(channel, filter_params)
        # odt_time, odt_sig, samples_per_rev = data.order_track(data.dataset["filtered_bp_Acc_Carrier"])
        # freq, mag, phase = data.fft(odt_sig,
        #                             samples_per_rev / data.PG.Z_r)  # Sampling rate scales frequency to be in terms
        # # of synchronous gear mesh frequency
        #
        # axs[harmonic].plot(freq, (mag / np.max(mag)) ** 2, label=name)
        # axs[harmonic].set_ylabel(str(harmonic + 1))#"Magnitude^2")
        # axs[harmonic].set_xlim((harmonic + 1) - 0.8 ,(harmonic +1) + 0.8)
        # axs[harmonic].legend(loc=1)
    return odt_time, odt_sig, samples_per_rev


def scalogram_of_non_gearmesh(d_list):
    for data in d_list:
        #plt.xlim(0, 20)
        t,odt,spr = compare_nat_freqs(data,n_harmonics=7)
        freq, mag, phase = data.fft(odt,spr / data.PG.Z_r)  # Sampling rate scales frequency to be in terms
        plt.figure(1)
        plt.plot(freq, mag ** 2, label = data.dataset_name)
        plt.legend(loc=1)

        sig_for_tsa = {"interp_sig":odt,"samples_per_rev":spr}
        winds = data.window_extract(0,2/62,sig_for_tsa,order_track="precomputed")
        windave,all_per_teeth = data.window_average(winds)
        aaw = data.aranged_averaged_windows(windave,plot=True)


        scales = np.logspace(1.6, 2.3, num=50, dtype=np.int32)  # Interesting low frequencies
        #scales = np.logspace(0.005, 2.5, num=100, dtype=np.int32)  # Interesting low frequencies

        fs = spr

        waveletname = 'cmor1.5-1.0'
        coefficients, frequencies = pywt.cwt(odt, scales, waveletname, 1 / fs)

        power = (abs(coefficients)) ** 2

        specto_info = {"interp_spec": power,
                       "samples_per_rev": spr,
                       "total_samples": len(t)}

        offset_frac = 0
        rev_frac = 4/62
        specwinds = data.spec_window_extract(offset_frac, rev_frac, specto_info, order_track=True)
        ave, allperteeth = data.spec_window_average(specwinds)
        data.spec_aranged_averaged_windows(ave, t, frequencies, plot=True,plot_title_addition=data.dataset_name)
    return

def squared_spectrum_at_harmonics(dataset_list,gearmesh_harmonics):

    fig, axs = plt.subplots(len(gearmesh_harmonics))
    fig.text(0.5, 0.04, 'n x GMF', ha='center')
    fig.text(0.04, 0.5, 'Squared magnitude', va='center', rotation='vertical')

    band_width_measure = 0.5
    border = 0.4
    cpw = False
    for data in dataset_list:
        bands, spr = get_gearmesh(data, "Acc_Carrier", gearmesh_harmonics, band_width_measure)

        for i, band in enumerate(bands):
            if cpw:
                freq, mag, phase = data.fft(data.cepstrum_pre_whitening(band), spr / 62)
            else:
                freq, mag, phase = data.fft(band, spr / 62)

            axs[i].plot(freq, (mag/np.max(mag)) ** 2,label=data.dataset_name)
            #axs[i].plot(freq, (mag) ** 2, label=data.dataset_name)
            #axs[i].plot(freq, (mag/np.mean(mag)) ** 2,label=data.dataset_name)

            axs[i].set_xlim(harmonics[i] - border, harmonics[i] + border)
    fig.legend([x.dataset_name for x in dataset_list])
    fig.show()
    return

def spectrum_of_squared_band_limited_signal(dataset_list,gearmesh_harmonics):

    fig, axs = plt.subplots(len(gearmesh_harmonics))
    fig.text(0.5, 0.04, 'n x GMF', ha='center')
    fig.text(0.04, 0.5, 'spectrum(band limited^2)', va='center', rotation='vertical')
    cpw = False

    band_width_measure = 0.5
    for data in dataset_list:
        bands, spr = get_gearmesh(data, "Acc_Carrier", gearmesh_harmonics, band_width_measure)

        for i, band in enumerate(bands):
            if cpw:
                freq, mag, phase = data.fft(data.cepstrum_pre_whitening(band)**2, spr / 62)
            else:
                freq, mag, phase = data.fft(band**2, spr / 62)
            # axs[i].plot(freq, np.log(mag), label=data.dataset_name)
            axs[i].plot(freq, mag/np.mean(mag), label=data.dataset_name)
            axs[i].set_xlim(0, 0.2)
            axs[i].set_ylabel("h" + str(gearmesh_harmonics[i]))
    fig.legend([x.dataset_name for x in dataset_list])
    fig.show()
    return

def tsa_of_bandlimited(dataset_list,gearmesh_harmonics):

    band_width_measure = 0.5

    for data in dataset_list:
        bands, spr = get_gearmesh(data, "Acc_Carrier", gearmesh_harmonics, band_width_measure)

        for i, band in enumerate(bands):
            sig_for_tsa = {"interp_sig":band**2,"samples_per_rev":spr}
            winds = data.window_extract(0,
                                        2/62,
                                        sig_for_tsa,
                                        order_track="precomputed")
            windave,all_per_teeth = data.window_average(winds)
            aaw = data.aranged_averaged_windows(windave, plot=True,
                                        plot_title_addition = data.dataset_name + "band_limited(x^2)" + str(gearmesh_harmonics[i]) + "*gmf")
    return

#dlist = [damaged, healthy]
dlist = [damaged, d_mid, h_mid, healthy]

squared_spectrum_at_harmonics(dlist, harmonics)
spectrum_of_squared_band_limited_signal(dlist, harmonics)
#tsa_of_bandlimited(dlist,harmonics)
#scalogram_of_non_gearmesh(dlist)

print("Dataset ave RPM's ", [x.info["rpm_sun_ave"] for x in dlist])