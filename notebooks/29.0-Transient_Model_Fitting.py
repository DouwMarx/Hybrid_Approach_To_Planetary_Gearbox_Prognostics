import numpy as np
import matplotlib.pyplot as plt
import pickle
import dill
import definitions
import scipy.integrate as inter
import scipy.optimize as opt
import scipy.signal as scisig
import src.models.analytical_sdof_model as an_sdof

import src.features.proc_lib as proc

plt.close("all")

#  Load the dataset object
filename = "g1_p0_v9_0.pgdata"
directory = definitions.root + "\\data\\processed\\" + filename
with open(directory, 'rb') as filename:
    data = pickle.load(filename)

# Make a transients object and assign some data attributes for rapid development
tobj = proc.TransientAnalysis()
tobj.info = data.info
tobj.derived_attributes = data.derived_attributes
tobj.dataset = data.dataset
tobj.dataset_name = data.dataset_name

windows = data.derived_attributes["extracted_windows"]
n = np.random.randint(0, np.shape(windows)[0])
sig = windows[n, :]  # Get a single signal
trans, peak, tgm = tobj.get_transients(sig,0.0001,0.0008)  # Extract the transients from this signal

i = np.random.randint(0, np.shape(trans)[0])
acc = trans[i, :]  # Select one of the transients randomly





ods = an_sdof.one_dof_sys(acc, data.info['f_s'])
#print(ods.run_optimisation_ana_sol(plot=True))
# print(ods.run_optimisation())
s = ods.do_least_squares(plot=True)
print(s)