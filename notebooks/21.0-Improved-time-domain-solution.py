import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import models.lumped_mas_model.llm_models as lmm_models
import importlib  #This allows me to reload my own module every time
import definitions
import src.features.second_order_solvers as s
import time



plt.close("all")

#PG = lmm_models.make_chaari_2006_1planet()
#PG = lmm_models.make_lin_1999_model()
#PG = lmm_models.make_liang_2015()
PG = lmm_models.make_chaari_2006_model()


def fft(data, fs):
    """

    Parameters
    ----------
    data: String
        The heading name for the dataframe

    Returns
    -------
    freq: Frequency range
    magnitude:
    phase:
    """

    d = data
    length = len(d)
    Y = np.fft.fft(d) / length
    magnitude = np.abs(Y)[0:int(length / 2)]
    phase = np.angle(Y)[0:int(length / 2)]
    freq = np.fft.fftfreq(length, 1 / fs)[0:int(length / 2)]
    return freq, magnitude, phase

#PG.get_solution()
#PG.plot_solution("Displacement")

#PG.get_natural_freqs()
PG.get_solution()
PG.plot_solution("Displacement")


#d = definitions.root + "\\" + "data\\external\\lmm\\"
#np.save(d + "Response_4_0.npy", s)