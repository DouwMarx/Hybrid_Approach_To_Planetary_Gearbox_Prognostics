import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np
import src.features.proc_lib as proc
from src.visualization.multipage_pdf import multipage

#damaged_filename = "cycle_2_end.pgdata"
#healthy_filename = "g1_p0_v8_8.pgdata"

# damaged_filename = "g1_p7_8.8.pgdata"
# d_mid_filename = "g1_p5_8.8.pgdata"
# h_mid_filename = "g1_p3_9.0.pgdata"
# healthy_filename = "g1_p0_9.0.pgdata"

# damaged_filename = "g1_p7_9.4.pgdata"
# d_mid_filename = "g1_p7_9.2.pgdata"
# h_mid_filename = "g1_p7_9.0.pgdata"
# healthy_filename = "g1_p7_8.8.pgdata"
#
# healthy_directory = definitions.root + "\\data\\processed\\" + healthy_filename
# with open(healthy_directory, 'rb') as filename:
#     healthy = pickle.load(filename)
#
# damaged_directory = definitions.root + "\\data\\processed\\" + damaged_filename
# with open(damaged_directory, 'rb') as filename:
#     damaged = pickle.load(filename)
#
# d_mid_directory = definitions.root + "\\data\\processed\\" + d_mid_filename
# with open(d_mid_directory, 'rb') as filename:
#     d_mid = pickle.load(filename)
#
# h_mid_directory = definitions.root + "\\data\\processed\\" + h_mid_filename
# with open(h_mid_directory, 'rb') as filename:
#     h_mid = pickle.load(filename)


# Get tsa when filtered around a natural freq


def filter_around_nat_freqs(channel, pgdata,centre_freqs,band_width,plot_title_addition):
    type = "band"
    bot_freqs = centre_freqs - 0.5*band_width

    for bot_freq in bot_freqs:
        sigprocobj = proc.Signal_Processing()
        sigprocobj.info = pgdata.info
        sigprocobj.dataset = pgdata.dataset
        # Filter the signal
        if type == "band":
            filter_params = {"type": "band",
                             "low_cut": bot_freq,
                             "high_cut": bot_freq + band_width}
        else:
            filter_params = {"type": "low",
                             "cut": bot_freq + band_width}

        sigprocobj.filter_column(channel, filter_params)
        tsa_obj = proc.Time_Synchronous_Averaging()

        # Create a TSA object
        tsa_obj.info = pgdata.info
        tsa_obj.derived_attributes = pgdata.derived_attributes
        tsa_obj.dataset = sigprocobj.dataset  # Notice that the dataset is exchanged for filtered dataset
        tsa_obj.dataset_name = pgdata.dataset_name
        tsa_obj.PG = pgdata.PG

        offset_frac = (1 / 62) * (0.0)
        win_frac = 4/62
        if type == "band":
            winds = tsa_obj.window_extract(offset_frac, win_frac,
                                           tsa_obj.dataset["filtered_bp_" + channel].values, plot=False)
        else:
            winds = tsa_obj.window_extract(offset_frac, win_frac,
                                           tsa_obj.dataset["filtered_lp_" + channel].values, plot=False)

        wind_ave, all_p_teeth = tsa_obj.window_average(winds, plot=False)

        ave_in_order, planet_gear_rev = tsa_obj.aranged_averaged_windows(wind_ave**2, plot="together") # Using the squared signal
        plt.title(
            "Filter (order tracked) " + channel + " " + str(bot_freq) + " -> " + str(bot_freq + band_width) + " Hz_" + plot_title_addition)
        #plt.show()
    return

# Study from frequencies obtained by modal analysis
# =====================================================================================================================
# nat_freqs = {"ch4": np.array([100,106,280,475,664,730,1162,2400,8062,8773,10841]),
#              "ch3": np.array([322,541,576,620,696,756,997,1057,1185,1253,2625,3051]),
#              "ch2": np.array([266,385,564,664,733,821,1070,1219,2062,2199,2642,6313])}
#

# for mod_ch in ["ch2","ch3","ch4"]:
#     for bandwidth in [50,100,200,400,800]:
#
#         try:
#             filter_around_nat_freqs("Acc_Carrier",damaged,nat_freqs[mod_ch],bandwidth)
#         except:
#             pass
#
#
#         result_path = definitions.root + "\\reports\\signal_processing_results\\"
#         result_file_name = "Full_crack_at_nat_freqs_mod-"+ mod_ch + "_bandwidth-" + str(bandwidth) + ".pdf"
#         multipage(result_path + result_file_name)  # Save all open figures
#         plt.close("all")  # Close all open figures

# Study for frequencies from response after removing gearmesh
# =====================================================================================================================
# bandwidth = 5
# nat_freqs = np.array([289,297,301,610.2,1476,   1798])  #610.2

# bandwidth = 50
# nat_freqs = np.array([424,612,676,745,892,1171,1309,1478,1567,1669,1724])  #676,1171,1724

# bandwidth = 50
# nat_freqs = np.array([2948, 3095, 3136, 3450])  # 3136

bandwidth = 800
nat_freqs = np.array([3450])

for gear in ["g2_p5_"]:
    for voltage in ["9.0", "9.2", "9.4", "9.6", "9.8"]:
    #for voltage in ["9.0", "9.2", "9.4", "9.6", "9.8", "8.8"]:
        filename = gear + voltage + ".pgdata"
        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)
            filter_around_nat_freqs("Acc_Carrier", data, nat_freqs, bandwidth,plot_title_addition = "@" + voltage + "V")

    result_path = definitions.root + "\\reports\\signal_processing_results\\"
    result_file_name = "full_crack_at_nat_freq-"+ str(nat_freqs[0])  + "Hz_bandwidth-" + str(bandwidth) + ".pdf"
    multipage(result_path + result_file_name)  # Save all open figures
    plt.close("all")  # Close all open figures

#bandwidth = 200
#nat_freqs = np.array([1171])
# 2948 is the centre of the natural frequency band with bandwidth of 1400

# for dset in [healthy,h_mid,d_mid,damaged]:
#     filter_around_nat_freqs("Acc_Carrier",dset,nat_freqs,bandwidth)


# result_path = definitions.root + "\\reports\\signal_processing_results\\"
# result_file_name = "Full_crack_at_nat_freqs_mod-"+ mod_ch + "_bandwidth-" + str(bandwidth) + ".pdf"
# multipage(result_path + result_file_name)  # Save all open figures
# plt.close("all")  # Close all open figures
