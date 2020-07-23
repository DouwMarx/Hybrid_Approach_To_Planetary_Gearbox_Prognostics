import pickle
import definitions
from PyEMD import EMD, EEMD
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


plt.close("all")

#  Load the dataset object
# =====================================================================================================================
# filename = "g1_p7_8.8.pgdata"
# filename = "g1_fc_1000.pgdata"
filename = "g1_fc_1000_long.pgdata"

directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

# Set the TSA parameters
offset_frac = 0
rev_frac = 2 / 62

to_apply = ["rpm","no_order_track","order_track","wiener_filtered_signal","bp_filtered_signal","squared_signal","EEMD","tsa_spectrogram","spectrogram"]
#to_apply = ["tsa_spectrogram"]  # bp_filtered_signal"]

#for channel in ["Acc_Sun", "Acc_Carrier"]:
for channel in ["Acc_Carrier"]:
    # Plot rpm over time


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
        n_tsa_sig = np.shape(tsa_odt)[0]
        fig1, axs1 = plt.subplots(n_tsa_sig, 1)
        fig2, axs2 = plt.subplots(n_tsa_sig, 1)
        fig3, axs3 = plt.subplots(n_tsa_sig, 1)
        fig4, axs4 = plt.subplots(n_tsa_sig, 1)

        for tooth_pair in range(n_tsa_sig):
            # emd = EMD() # EMD is apparently not very robust and therefore we use the more expensive EEMD
            # emd.FIXE = 4

            eemd = EEMD(DTYPE=np.float16, trials=20, max_imfs=4, parallel=False)

            tsa_sig = tsa_odt[tooth_pair, :]
            # imfs = emd(tsa_sig)
            imfs = eemd(tsa_sig)

            axs1[tooth_pair].plot(imfs[0])
            axs1[tooth_pair].set_ylabel(str(tooth_pair * 2))
            # axs1[tooth_pair].set_xlabel("Samples @ 38400Hz")
            fig1.suptitle("EEMD of TSA: imf1")

            axs2[tooth_pair].plot(imfs[1])
            axs2[tooth_pair].set_ylabel(str(tooth_pair * 2))
            # axs2[tooth_pair].set_xlabel("Samples @ 38400Hz")
            fig2.suptitle("EEMD of TSA: imf2")

            axs3[tooth_pair].plot(imfs[2])
            axs3[tooth_pair].set_ylabel(str(tooth_pair * 2))
            # axs3[tooth_pair].set_xlabel("Samples @ 38400Hz")
            fig3.suptitle("EEMD of TSA: imf3")

            axs4[tooth_pair].plot(imfs[3])
            axs4[tooth_pair].set_ylabel(str(tooth_pair * 2))
            # axs4[tooth_pair].set_xlabel("Samples @ 38400Hz")
            fig4.suptitle("EEMD of TSA: imf4")

            plt.xlabel("Samples @ 38400Hz")

    if "spectrogram" in to_apply:
        data.full_spectrogram(data.dataset[channel], data.info["f_s"], plot=True, plot_title_addition=channel)

    if "tsa_spectrogram" in to_apply:
        fs = data.info["f_s"]
        nperseg = 100
        f, t, sxx = data.spectrogram(data.dataset[channel].values, fs, nperseg, plot=False)
        specwinds = data.spec_window_extract(0, 2 / 62, sxx)
        ave, allperteeth = data.spec_window_average(specwinds)
        data.spec_aranged_averaged_windows(ave, t, f, plot=True,plot_title_addition=channel)

    if "rpm" in to_apply:
        data.plot_rpm_over_time()
multipage("ACC_carrier.pdf")
plt.close("all")
