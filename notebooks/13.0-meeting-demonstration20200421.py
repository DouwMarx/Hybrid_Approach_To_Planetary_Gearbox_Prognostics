
import models.lumped_mas_model.llm_models as lmm
import numpy as np
import matplotlib.pyplot as plt
import definitions
import src.models.lumped_mas_model as pglmm

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

plt.close("all")

#PG = lmm.make_chaari_2006_model()
PG = lmm.make_chaari_2006_1planet()


t = np.linspace(0, 1, 100000)
#d1 = pglmm.DE_Integration(PG)
d = definitions.root + "\\" + "data\\external\\lmm\\"


xdd = np.load(d + "Acceleration_3_0.npy")
plt.figure("Accelerations")
plt.ylabel("Acceleration [m]")
plt.xlabel("Time [s]")
#p = plt.plot(t, sol[:,0:half_sol_shape])
#plt.ylim([-1e-6,1e-6])
#plt.legend(iter(p),lables)
plt.plot(t, xdd[:, 4])


plt.figure("FFT rot ref")
plt.ylabel("Acceleration [m/s^2]")
plt.xlabel("Frequency [Hz]")
#p = plt.plot(t, sol[:,0:half_sol_shape])
#plt.ylim([-1e-6,1e-6])
#plt.legend(iter(p),lables)
freq, mag, phase = fft(xdd[:, 4],100000)
plt.plot(freq, mag)



