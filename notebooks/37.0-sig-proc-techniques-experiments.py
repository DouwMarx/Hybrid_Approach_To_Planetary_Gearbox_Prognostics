import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np

#  Load the dataset object
# =====================================================================================================================
# filename = "g1_p7_8.8.pgdata"
# filename = "g1_fc_1000.pgdata"
# filename = "g1_fc_1000_long.pgdata"


to_apply = [
            "squared_signal"
            ]


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
#             "abs_odt_cpw_tsa",
#             "mean_variance_skewness_kurtosis",
#             "tsa_odt_scaleogram"
#             ]

channel = "Acc_Carrier"

#for gear in ["g1_p7_", "g2_p5_"]:
#for gear in ["g2_p5_"]:
for gear in ["g2_p5_"]:
    for voltage in ["9.0", "9.2", "9.4", "9.6", "9.8", "8.8"]:
    #for voltage in ["9.0"]:
        filename = gear + voltage + ".pgdata"
        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)
            data.sigproc_dset(to_apply, channel)
