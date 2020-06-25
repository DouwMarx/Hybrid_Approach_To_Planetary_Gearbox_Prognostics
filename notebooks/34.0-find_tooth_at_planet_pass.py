import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import definitions

import src.features.proc_lib as proc

plt.close("all")

#  Load the dataset object
filename = "g1_fc_1000_long.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename

with open(directory, 'rb') as filename:
    data = pickle.load(filename)

# data.plot_rpm_over_time()

m_seq = data.derived_attributes["mesh_sequence_at_planet_pass"]
