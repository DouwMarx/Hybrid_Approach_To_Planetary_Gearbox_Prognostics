import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

import src.features.proc_lib as proc
plt.close("all")

#  Load the dataset object
filename = "g1_p0.pgdata"
#filename = "cycle_5_end.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

#data.plot_rpm_over_time()



