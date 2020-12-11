import pickle
import definitions
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import pywt
import src.features.compare_datasets as cd

# damaged_filename = "cycle_2_end.pgdata"
# healthy_filename = "g1_p0_v8_8.pgdata"

# damaged_filename = "g1_p7_8.8.pgdata"
# d_mid_filename = "g1_p5_8.8.pgdata"
# h_mid_filename = "g1_p3_9.0.pgdata"
# healthy_filename = "g1_p0_9.0.pgdata"




# d_list = []
# for gear in ["g1_p7_", "g1_p0_"]:
#     for voltage in ["9.0", "9.2", "9.4", "9.6", "9.8", "8.8"]:
#     #for voltage in ["9.0" ,"9.2","9.4"]:
#         filename = gear + voltage + ".pgdata"
#         directory = definitions.root + "\\data\\processed\\" + filename
#         with open(directory, 'rb') as filename:
#             data = pickle.load(filename)
#             d_list.append(data)

d_list = []
for filename in ["cycle_2_end.pgdata", "g1_p0_v8_8.pgdata"]:
        directory = definitions.root + "\\data\\processed\\" + filename
        with open(directory, 'rb') as filename:
            data = pickle.load(filename)
            d_list.append(data)

# gear_mesh_harmonics = np.array([1, 2, 3, 4, 5])
# for gear_mesh_harmonics in [np.array([i,i+1]) for i in range(1,5,2)]:
    # cd.squared_spectrum_at_harmonics(d_list, gear_mesh_harmonics)
    # cd.squared_envelope_spectrum(d_list, gear_mesh_harmonics)

