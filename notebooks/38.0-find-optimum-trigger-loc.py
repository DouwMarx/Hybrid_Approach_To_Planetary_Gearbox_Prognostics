import definitions
import pickle
import numpy as np
import matplotlib.pyplot as plt
#  Load the dataset object
# =====================================================================================================================
# filename = "find_acc_centre_vs_magpickup_t.pgdata"
filename = "find_acc_centre_vs_magpickup_t.pgdata"

directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

"""
Data acquired by turning shaft large amount to determine whether trigger should be on positive or negative slope
and then small amount of which the average should be the centre
"""

# From plot, use data from 15s to 17.5s to compute average
fs = data.info["f_s"]
trigpoint = np.average(data.dataset["1PR_Mag_Pickup"].values[int(15*fs):int(17.5*fs)])
print(trigpoint)

t = data.dataset["Time"].values
plt.figure()
data.plot_time_series("1PR_Mag_Pickup")
plt.hlines(trigpoint,min(t),max(t),"r")

"""
Results:
Use trigger with negative slope and trigger val of 3.897
"""

