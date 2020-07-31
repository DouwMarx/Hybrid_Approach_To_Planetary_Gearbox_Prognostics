import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as sig

channel = "Acc_Carrier"

for gear in ["g2_p5_"]:
    for voltage in ["9.0"]:
        filename = gear + voltage + ".pgdata"
        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)


d = data.dataset["Acc_Carrier"].values
dcpw = data.cepstrum_pre_whitening(np.abs(d))
winds = data.window_extract(0,4/62,dcpw)

ws = data.window_stats(winds, "skewness", relative = True, filter_stat = True)
asw = data.arranged_window_stats(ws, plot=True,plot_title_addition="Stuff")

