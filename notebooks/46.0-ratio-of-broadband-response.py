import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np
import src.features.proc_lib as proc
from src.visualization.multipage_pdf import multipage

# Used to determine if there is a relationship between the amplitude of the tsa of the damaged gear and the change in
# its mesh stiffness

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

        ave_in_order, planet_gear_rev = tsa_obj.aranged_averaged_windows(wind_ave, plot=True) # Using the squared signal
        plt.title(
            "Filter (order tracked) " + channel + " " + str(bot_freq) + " -> " + str(bot_freq + band_width) + " Hz_" + plot_title_addition)
        #plt.show()
    return

damaged_filename = "cycle_4_end.pgdata"

damaged_directory = definitions.root + "\\data\\processed\\" + damaged_filename
with open(damaged_directory, 'rb') as filename:
    damaged = pickle.load(filename)

data = damaged

offset_frac = 0
win_frac = 4/62

#winds = data.window_extract(offset_frac, win_frac,
#                               data.dataset["Acc_Carrier"].values, plot=False)

#wind_ave, all_p_teeth = data.window_average(winds, plot=False)
#ave_in_order, planet_gear_rev = data.aranged_averaged_windows(wind_ave, plot= True)  # Using the squared signal

# freqs = np.array([289,297,301,610.2,1476,   1798])  #610.2
# freqs = np.array([424,612,676,745,892,1171,1309,1478,1567,1669,1724])
freqs = np.array([2948, 3095, 3136, 3450])  # 3136
bw = 800 # bandwidth  hz
filter_around_nat_freqs("Acc_Carrier",data,freqs,bw,"")
