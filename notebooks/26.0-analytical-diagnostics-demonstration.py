import definitions
import numpy as np
import models.lumped_mas_model.llm_models as lmm_models
from src.models.diagnostics import Diagnostics
import pickle

opim_for = {
            "m_r": np.array([0.588*0.5, 0.588*1.5]),
            "delta_k": np.array([1E8*0.5, 1E8*1.5]),
            "m_p": np.array([0.1*0.5, 0.1*1.5]),
            }

# Load "actual" generated data
d = definitions.root + "\\" + "data\\external\\lmm\\"
measured = np.load(d + "transducer_vib_diagnostics1.npy")

# # Load experimental order tracked TSA data
# filename = "cycle_2_end.pgdata"
# directory = definitions.root + "\\data\\processed\\" + filename
# with open(directory, 'rb') as filename:
#     data = pickle.load(filename)
#     tsa = data.Compute_TSA(0)
#     measured = tsa[0:10000]


# Create the diagnostics object
diag_obj = Diagnostics(measured, opim_for, lmm_models.make_chaari_2006_model_w_dict())

# Run diagnostics
diag_obj.do_optimisation()

