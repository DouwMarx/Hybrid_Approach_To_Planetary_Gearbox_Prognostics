import pickle
import definitions
from PyEMD import EMD
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


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

# Plot rpm over time
# ======================================================================================================================
data.plot_rpm_over_time()

# Set the TSA parameters
offset_frac = 0
rev_frac = 2 / 62

for channel in ["Acc_Sun", "Acc_Carrier"]:
    # # TSA no order track
    # data.compute_tsa(offset_frac,
    #                  rev_frac,
    #                  data.dataset[channel].values,
    #                  ordertrack=False,
    #                  plot=True,
    #                  plot_title_addition=channel + " not order tracked")

    # TSA order track
    tsa_odt = data.compute_tsa(offset_frac,
                     rev_frac,
                     data.dataset[channel].values,
                     ordertrack=True,
                     plot=True,
                     plot_title_addition=channel + " order tracked")

    # Squared signal order tracked
    # data.compute_tsa(offset_frac,
    #                  rev_frac,
    #                  data.dataset[channel].values ** 2,
    #                  ordertrack=True,
    #                  plot=True,
    #                  plot_title_addition=channel + ", squared signal")

    # Empirical Mode decomposition

    emd = EMD()
    imfs = emd(tsa_odt[:,1])
    plt.figure()
    plt.plot(imfs)

multipage("multipage.pdf")
plt.close("all")
