import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import models.lumped_mas_model.llm_models as lmm_models
import definitions
import time



plt.close("all")

#PG = lmm_models.make_chaari_2006_1planet()
#PG = lmm_models.make_lin_1999_model()
#PG = lmm_models.make_liang_2015()
#PG = lmm_models.make_chaari_2006_model()
PG_info_dict = lmm_models.make_chaari_2006_model_w_dict()

PG = pglmm.Planetary_Gear(PG_info_dict)
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
#PG.plot_solution("Displacement")
#sol = PG.get_transducer_vibration()

#d = definitions.root + "\\" + "data\\external\\lmm\\"
#np.save(d + "transducer_vib_diagnostics1.npy", sol)

#t_range = np.linspace(0,0.1,1000)
#Keobj = pglmm.K_e(PG)
#ss = Keobj.smooth_square(t_range,100,1/100)

#plt.figure()
#plt.plot(ss)