import src.models.lumped_mas_model as pglmm
import numpy as np
import matplotlib.pyplot as plt
import models.lumped_mas_model.llm_models as lmm_models
import definitions
import time



plt.close("all")

#PG_info_dict = lmm_models.make_chaari_2006_model_w_dict()
PG_info_dict = lmm_models.make_no_torque_no_detak_rand_init()
#PG_info_dict = lmm_models.make_torque_no_detak_rand_init_base_exitation()

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


#PG.get_natural_freqs()
PG.get_solution()
PG.plot_solution("Displacement", labels=True)
PG.plot_solution("Velocity", labels=True)
sol = PG.get_transducer_vibration()

plt.figure()
plt.plot(sol)
#d = definitions.root + "\\" + "data\\external\\lmm\\"
#np.save(d + "transducer_vib_diagnostics2.npy", sol)

#PG.plot_stiffness_and_damping_mat()
