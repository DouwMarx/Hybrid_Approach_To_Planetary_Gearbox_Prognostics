import pickle
import definitions
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


#  Load the dataset object
# =====================================================================================================================
# filename = "g1_p7_8.8.pgdata"
#filename = "g1_fc_1000.pgdata"
filename = "g1_fc_1000_long.pgdata"

directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

# Set the TSA parameters
offset_frac = 0
rev_frac = 2 / 62

# to_apply = ["rpm",
#             "no_order_track",
#             "order_track",
#             "wiener_filtered_signal",
#             "bp_filtered_signal",
#             "squared_signal",
#             "EEMD",
#             "tsa_spectrogram",
#             "tsa_odt_spectrogram",
#             "spectrogram",
#             "squared_signal_spectrum",
#             "abs_odt_cpw_tsa"
#             ]


to_apply = ["tsa_odt_spectrogram"]  # "bp_filtered_signal"]

# for channel in ["Acc_Sun", "Acc_Carrier"]:
for channel in ["Acc_Carrier"]:
    # TSA no order track
    if "no_order_track" in to_apply:
        data.compute_tsa(offset_frac,
                         rev_frac,
                         data.dataset[channel].values,
                         ordertrack=False,
                         plot=True,
                         plot_title_addition=channel + " not order tracked")

    # TSA order track
    if "order_track" in to_apply:
        tsa_odt = data.compute_tsa(offset_frac,
                                   rev_frac,
                                   data.dataset[channel].values,
                                   ordertrack=True,
                                   plot=True,
                                   plot_title_addition=channel + " order tracked")

    # Squared signal order tracked
    if "squared_signal" in to_apply:
        data.compute_tsa(offset_frac,
                         rev_frac,
                         data.dataset[channel].values ** 2,
                         ordertrack=True,
                         plot=True,
                         plot_title_addition=channel + ", squared signal")

    if "wiener_filtered_signal" in to_apply:
        data.filter_column(channel, {"type": "wiener"})
        r = data.compute_tsa(offset_frac,
                             rev_frac,
                             data.dataset["filtered_wiener_" + channel].values,
                             ordertrack=True,
                             plot=True,
                             plot_title_addition=channel + ", Wiener filtered signal")

        print(np.linalg.norm(r))
        print(np.linalg.norm(tsa_odt))

    # Filtered signal at different frequency ranges
    if "bp_filtered_signal" in to_apply:
        data.filter_at_range_of_freqs(channel, "band", data)

    # Empirical Mode decomposition
    if "EEMD" in to_apply:
        data.eemd(tsa_odt)

    if "spectrogram" in to_apply:
        data.full_spectrogram(data.dataset[channel], data.info["f_s"], plot=True, plot_title_addition=channel)

    if "tsa_spectrogram" in to_apply:
        fs = data.info["f_s"]
        nperseg = 100
        f, t, sxx = data.spectrogram(data.dataset[channel].values, fs, nperseg, plot=False)
        specwinds = data.spec_window_extract(0, 2 / 62, sxx)
        ave, allperteeth = data.spec_window_average(specwinds)
        data.spec_aranged_averaged_windows(ave, t, f, plot=True, plot_title_addition=channel)

    if "tsa_odt_spectrogram" in to_apply:
        tnew, interp_sig, samples_per_rev = data.order_track(data.dataset[channel].values)

        fs = samples_per_rev
        print(samples_per_rev)

        nperseg = 80
        f, t, sxx = data.spectrogram(interp_sig, fs, nperseg, plot=False)

        specto_info = {"interp_spec":sxx,
                        "samples_per_rev":samples_per_rev,
                        "total_samples": len(tnew)}

        specwinds = data.spec_window_extract(0, 2 / 62, specto_info, order_track=True)
        ave, allperteeth = data.spec_window_average(specwinds)
        data.spec_aranged_averaged_windows(ave, t, f, plot=True, plot_title_addition= "order_tracked " +channel)

    if "rpm" in to_apply:
        data.plot_rpm_over_time()

    if "squared_signal_spectrum" in to_apply:
        tnew, interp_sig, samples_per_rev = data.order_track(data.dataset[channel])
        data.plot_order_spectrum(data.dataset[channel].values ** 2, samples_per_rev)
        plt.xlim([0,800])
        plt.ylim([0,60])
        plt.title("Order tracked squared signal envelope spectrum: " + channel)

    if "abs_odt_cpw_tsa" in to_apply:
        abs_sig = np.abs(data.dataset[channel].values)
        tnew, interp_sig, samples_per_rev = data.order_track(abs_sig)
        cpw = data.cepstrum_pre_whitening(interp_sig)
        signal = {"interp_sig":interp_sig,
                  "samples_per_rev":samples_per_rev}

        winds = data.window_extract(offset_frac, rev_frac, signal, order_track="precomputed",plot=False)

        aves, apt = data.window_average(winds)
        arranged, together = data.aranged_averaged_windows(aves, plot=True, plot_title_addition=channel + " cpw(odt(abs(x)))")


#multipage("ACC_carrier.pdf")
# plt.close("all")
