import pickle
import definitions

to_apply = [
            "order_track"
            ]


channel = "Acc_Carrier"

#for gear in ["g1_p7_", "g2_p5_"]:
#for gear in ["g2_p5_"]:
mid_freq = 3450
band_width = 800
bot_freq = mid_freq - 0.5*band_width
top_freq = mid_freq + 0.5*band_width

channel = "Acc_Carrier"

for gear in ["g2_p5_"]:
    for voltage in ["9.0", "9.2", "9.4", "9.6", "9.8", "8.8"]:
    #for voltage in ["9.0"]:
        filename = gear + voltage + ".pgdata"
        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)
            signal = data.dataset[channel].values
            filter_params = {"type": "band",
                             "low_cut": bot_freq,
                             "high_cut": bot_freq + band_width}

            data.filter_column(channel,filter_params)

            data.compute_tsa(0, 2/62, data.dataset["filtered_bp_Acc_Carrier"]**2, plot=True)
