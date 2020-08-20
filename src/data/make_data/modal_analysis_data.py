import numpy as np
import matplotlib.pyplot as plt

# 432  : 1000 Hz hard tip
# 433  : 1000 Hz soft tip
# 434  : 11000 Hz hard tip

# Accelerometer? SN179969

for channel in ["ch2","ch3","ch4"]:
    plt.figure()
    plt.title("0->11000 Hz hard, channel: " + channel)
    directory = "D:\\M_Data\\raw\\modal_analysis\\" + "SIG0434_H(" + channel + ",ch1).csv"
    d = np.genfromtxt(directory, skip_header=31, delimiter=",")

    freq = d[:, 0]
    real = d[:, 1]
    imag = d[:, 2]

    mag = np.sqrt(real ** 2 + imag ** 2)

    plt.plot(freq, mag**2)
    plt.xlabel("frequency[Hz]")
    plt.ylabel("squared magnitude")

for channel in ["ch2","ch3","ch4"]:
    plt.figure()
    plt.title("0-1000Hz, channel: " + channel)
    for hardness,number in zip(["hard","soft"],["432","433"]):
        directory = "D:\\M_Data\\raw\\modal_analysis\\" + "SIG0" + number + "_H(" + channel + ",ch1).csv"
        d = np.genfromtxt(directory, skip_header=31, delimiter=",")

        freq = d[:, 0]
        real = d[:, 1]
        imag = d[:, 2]

        mag = np.sqrt(real ** 2 + imag ** 2)

        plt.plot(freq, mag**2,label = hardness)
        plt.xlabel("frequency[Hz]")
        plt.ylabel("squared magnitude")
        plt.legend()



nat_freqs = {"ch4": np.arary([100,106,280,475,664,730,1162,2400,8062,8773,10841]),
             "ch3": np.arary([322,541,576,620,696,756,997,1057,1185,1253,2625,3051]),
             "ch2": np.array([266,385,564,664,733,821,1070,1219,2062,2199,2642,6313])}

# Bandwith around 80Hz?
# For higher frequencies like 2400Hz we can have up to 1000Hz bandwidth