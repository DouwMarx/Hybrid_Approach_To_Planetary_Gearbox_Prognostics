import numpy as np
import scipy.signal as s
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import src.models.analytical_sdof_model as an_sdof
import scipy.interpolate as interp


class Dataset_Plotting(object):
    """This function generates plots from certain attributes that a dataset might have"""

    def plot_trigger_times_test(self):
        """Plots to tests Tachos_and_Triggers.trigger_times function for the magnetic pickup"""
        trigger_points = self.derived_attributes["trigger_time_mag"]
        time = self.dataset["Time"]
        magnetic_pickup = self.dataset["1PR_Mag_Pickup"]

        plt.figure("Evaluate if trigger is occuring at the correct time")
        plt.vlines(trigger_points, 0, 10)
        plt.plot(time, magnetic_pickup)

        plt.figure("Should form a straight line for constant speed")
        plt.scatter(np.arange(trigger_points.shape[0]), trigger_points)

    def plot_rpm_over_time(self):
        """
        Plots the rpm of the low speed side based on the magnetic pickup readings
        Returns
        -------

        """
        plt.figure()
        rpm = self.derived_attributes["rpm_mag"]
        plt.plot(self.derived_attributes["t_rpm_mag"], rpm)
        plt.xlabel("Time [s]")
        plt.ylabel("Input RPM")
        plt.text(0, max(rpm), "Average motor speed: " + str(int(self.info["rpm_sun_ave"])) + " RPM")

    def plot_fft(self, data, fs, plot_gmf=False):
        """
        Computes and plots the FFT for a given signal
        Parameters
        ----------
        data: String
            Name of the heading of the dataset to be FFTed

        Returns
        -------

        """
        freq, mag, phase = self.fft(data, fs)

        plt.figure()
        # max_height = 2
        plt.plot(freq, mag, "k")
        plt.ylabel("Magnitude")
        plt.xlabel("Frequency [Hz]")

        if plot_gmf == True:
            GMF = self.PG.GMF(self.info["rpm_sun_ave"] / 60)
            # FF1 = GMF/self.PG.Z_p
            FF = self.PG.FF1(self.info["rpm_sun_ave"] / 60)
            max_height = np.max(mag)
            plt.vlines(np.arange(1, 5) * GMF, 0, max_height, 'r', zorder=10, label="GMF and Harmonics")
            # plt.vlines(np.arange(1, 3) * FF, max_height, 'c', zorder=10, label="GMF and Harmonics")
            # plt.vlines(np.arange(1, 4) * FF1, 0, max_height, 'g', zorder=10, label="FF1 and Harmonics")

        # plt.xlim(0, 6000)
        # plt.show()

        return

    def plot_order_spectrum(self, data, samples_per_rev, plot_gmf=False):
        """
        Computes and plots the FFT for a given signal
        Parameters
        ----------
        data: String
            Name of the heading of the dataset to be FFTed

        Returns
        -------

        """
        freq, mag, phase = self.fft(data, samples_per_rev)

        plt.figure()
        # max_height = 2
        plt.plot(freq, mag, "k")
        plt.ylabel("Magnitude")
        plt.xlabel("Carrier orders")

        if plot_gmf == True:
            GMF = self.PG.GMF(self.info["rpm_sun_ave"] / 60)

            FF = self.PG.FF1(self.info["rpm_sun_ave"] / 60)
            # FF1 = GMF/self.PG.Z_p
            max_height = np.max(mag)
            plt.vlines(np.arange(1, 8) * GMF / samples_per_rev, 0, max_height, 'r', zorder=10,
                       label="GMF and Harmonics")
            plt.vlines(np.arange(1, 3) * FF / samples_per_rev, 0, max_height, 'c', zorder=10,
                       label="Fault frequency and Harmonics")
            # plt.vlines(np.arange(1, 4) * FF1, 0, max_height, 'g', zorder=10, label="FF1 and Harmonics")

        # plt.xlim(0, 6000)
        # plt.show()

        return

    def plot_TSA(self):
        """
       Plots the TSA
        Parameters
        ----------


        Returns
        -------

        """

        plt.plot(self.derived_attributes["TSA_Sun"], "k")
        plt.ylabel("TSA")
        plt.xlabel("Time")

    def plot_time_series(self, data_channel):
        """
        plots one of the dataframe attributes
        ----------
        data: String
            Name of the heading of the dataset to be FFTed

        Returns
        -------

        """

        plt.plot(self.dataset["Time"].values, self.dataset[data_channel], "k")
        plt.ylabel(data_channel)
        plt.xlabel("time [s]")

    def plot_time_series_4(self, data_channel):
        """
        plots one of the dataframe attributes
        ----------
        data: String
            Name of the heading of the dataset to be FFTed

        Returns
        -------

        """

        plt.plot(self.dataset["Time"].values, self.dataset[data_channel] ** 4, "k")
        plt.ylabel(data_channel)
        plt.xlabel("time [s]")


class Tachos_And_Triggers(object):
    """This class is used to work with magnetic switch and tacho data"""

    def trigger_times(self, tacho, trig_level):
        """Calculates the times at which the damaged planet gear passes the magnetic switch
        Parameters
        ----------
        trig_level
        tacho: string
             the tacho to be used for calculation ie. 'Tacho_Carrier', 'Tacho_Sun','1PR_Mag_Pickup'

        magnetic_pickup: array
                        magnetic pickup reading

        Returns
        -------
        time_points: Float
            Times at which the planet gear passes the accelerometer
            """
        time = self.dataset["Time"].values
        tach = self.dataset[tacho].values

        y = np.sign(tach - trig_level)
        dy = np.diff(y)
        indexes = np.where(dy > 0.8)

        time_points = time[indexes]

        if len(time_points) < 10:
            raise ValueError("Tacho trigger indexes less than 10")

        return indexes[0], time_points

    def getrpm(self, tacho, TrigLevel, Slope, PPRM, NewSampleFreq):
        """
        1. tacho = Tachometer Signal name
        3. triglevel =  trigger level defined by author for a pulse
        4. Slope = Positive or negative value for positive or negative pulses
        5. pprm = Tachometer pulses per revolution
        6. NewSampleFreq = Reinterpolation sampling frequency

        NOTE! The trig function is very simple and basic which requires a
        clean tacho signal. In some cases, a filtered tacho may work better
        than the original one.

        See also SMOOTHRPM

        A simple smoothing is performed on the rpm signal. A harder smoothing
        may in some circumstances be required.

        Returns:
            TimeRPM, RPM

        Copyright (c) 2003-2006, Axiom EduTech AB, Sweden. All rights reserved.
        URL: http://www.vibratools.com Email: support@vibratools.com
        Revision: 1.1  Date: 2003-08-06
        Revision history
        2006-05-03      Added extrapolation in the interp1 call was added to
                        avoid NaN's.
        Converted to Python by RC Balshaw
        """

        Fs = self.info["f_s"]
        Tacho = self.dataset[tacho]

        if type(Tacho) == list:
            Tacho = np.array(Tacho)

        y = np.sign(Tacho - TrigLevel)
        dy = np.diff(y)

        tt = self.maketime(dy, Fs)
        Pos = []
        cnt = 0
        if Slope > 0:
            for i in (dy > 0.8):
                if i == True:
                    Pos.append(cnt)
                cnt += 1

        if Slope < 0:
            for i in (dy < 0.8):
                if i == True:
                    Pos.append(cnt)

        if len(Pos) < 3:
            raise ValueError("Threshold for RPM measurement is probably too high")

        yt = tt[Pos]

        dt = np.diff(yt)
        dt = np.hstack([dt, np.array([dt[-1]])])

        Spacing = 2 * np.pi * np.ones(len(dt)) / PPRM  # Basic Spacing - radians

        rpm = (60 / (2 * np.pi)) * ((Spacing) / dt)
        b = [0.25, 0.5, 0.25]
        a = 1
        rpm = sig.filtfilt(b, a, rpm)
        # print(rpm.shape)
        N = np.max(tt) * (NewSampleFreq) + 1

        trpm = np.linspace(0, np.max(tt), N)

        rpm = np.interp(trpm, yt, rpm)
        Pos = []
        cnt = 0
        for i in np.isnan(rpm):
            if i == False:
                Pos.append(cnt)
            cnt += 1

        RPM = rpm[Pos]
        t_rpm = trpm[Pos]
        average_rpm = np.average(RPM)

        return (RPM, t_rpm, average_rpm)

    def maketime(self, X, Fs):
        """
        X = Signal
        Fs = sampling frequency

        Returns:
            time
        """
        t0 = 0
        t1 = len(X) / Fs
        t = np.arange(t0, t1 + 1 / Fs, 1 / Fs)
        return t


class Signal_Processing(object):
    """This class contains signal processing related functions"""

    def sampling_rate(self):
        """Compute the sampling rate from a timeseries"""
        timeseries = self.dataset["Time"].values
        # print(self.dataset["Time"])
        return np.ceil(1 / np.average(timeseries[1:] - timeseries[0:-1]))

    def fft(self, data, fs):
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

        #        d = self.dataset[data].values
        #        fs = self.info["f_s"]
        length = len(d)
        Y = np.fft.fft(d) / length
        magnitude = np.abs(Y)[0:int(length / 2)]
        phase = np.angle(Y)[0:int(length / 2)]
        freq = np.fft.fftfreq(length, 1 / fs)[0:int(length / 2)]
        return freq, magnitude, phase

    def order_track(self, data):
        d = self.dataset[data].values
        t = self.dataset["Time"]
        fs = self.info["f_s"]

        trigger_times = self.derived_attributes["trigger_time_mag"]

        tnew = np.array([])
        # tnew = np.array([0])
        ave_rot_time = np.average(np.diff(trigger_times))
        # print(ave_rot_time)
        samples_per_rev = int(fs * ave_rot_time)
        # print(samples_per_rev)
        for index in range(len(trigger_times) - 1):
            section = np.linspace(trigger_times[index], trigger_times[index + 1], samples_per_rev)
            tnew = np.hstack((tnew, section))

        interp_obj = interp.interp1d(t, d, kind='cubic')
        interp_sig = interp_obj(tnew)

        return tnew, interp_sig, samples_per_rev

    def filter(self,signal,lowcut,highcut):
        nyq = self.info["f_s"]/2
        low = lowcut/nyq
        high = highcut/nyq

        order = 5
        b,a = sig.butter(order, [low,high], btype="band")
        return sig.filtfilt(b,a,signal)

    def filter_column(self,signame,lowcut,highcut):
        sig = self.dataset[signame].values
        fsig = self.filter(sig,lowcut,highcut)
        key_name = "Filtered_" + signame
        self.dataset[key_name] = fsig
        return


class Time_Synchronous_Averaging(object):
    """
    Used to compute the time synchronous average for a planetary gearbox.
    See A TECHNIQUE FOR CALCULATING THE TIME DOMAIN AVERAGES OF THE VIBRATION OF THE INDIVIDUAL PLANET GEARS AND THE SUN GEAR IN AN EPICYCLIC GEARBOX, McFadden 1994

    Take note that the sun gear accelerometer is used
    """

    def window_extract(self, sample_offset_fraction, fraction_of_revolution, signal_name, plot=False):

        """Extracts a rectangular window depending on the average frequency of rotation of the planet gear
        ----------
        acc: array
             Accelerometer samples over time

        window_centre: array
             array of indexes where planet passes accelerometer as calculated by Planet Pass Time Function

        f_p_ave: float
                        average frequency of rotation of planet gear

        fs: Float
            Sampling Frequency

        Z_p: int
            Number of planet gear teeth. In the case of the Bonfiglioli gearbox, this should be Z_p/2 seeing that only even numbered gear planet gear teeth mesh with a given ring gear tooth.

        fraction_of_revolution: Used to determine the length of time for witch the window should be extracted
        Returns
        -------
        windows: nxm Array
            n windows each with m samples
            """

        acc = self.dataset[signal_name].values
        #acc = self.derived_attributes["order_track_signal"]

        # Notice that the average carrier period is used to ensure equal window lengths.
        # The assumption is that the RPM varies little enough that this is allowable
        # Also, the natural frequency of the transients is expected to be time independent
        carrier_period = self.info["carrier_period_ave"]

        # Number of samples for fraction of revolution at average speed
        window_length = int(self.info["f_s"] * carrier_period * fraction_of_revolution)

        if window_length % 2 == 0:  # Make sure that the window length is uneven
            window_length += 1

        # Takes fraction of a revolution
        offset_length = int(self.info["f_s"] * carrier_period * sample_offset_fraction)
        window_half_length = int((window_length - 1) / 2)
        window_center_index = self.derived_attributes["trigger_index_mag"] + offset_length

        # Exclude the first and last revolution to prevent errors with insufficient window length
        n_revs = np.shape(window_center_index)[0] - 2

        windows = np.zeros((n_revs, window_length))  # Initialize an empty array that will hold the extracted windows

        window_count = 0

        # Exclude the first and last revolution to prevent errors with insufficient window length
        for index in window_center_index[1:-1]:
            windows[window_count, :] = acc[index - window_half_length:index + window_half_length + 1]
            window_count += 1

        if plot:
            plt.figure("All Extracted")
            plt.plot(windows.T)

            plt.figure("Average of all Extracted")
            plt.plot(np.average(windows, axis=0))
        return windows

    def window_average(self, window_array, plot=False):
        """ Computes the average of windows extracted from the extract_windows function
        ----------
        window_array: array
             List of all extracted windows as calculated by extract_windows function

        rotations_to_repeat: int
             number of rotations (extracted windows) before an identical tooth meshing occurs to the first meshing configuration

        Returns
        -------
        Averages: nxm Array
            n gear teeth of the planet gear each with an averaged window of m samples
            """
        rotations_to_repeat = len(self.PG.Mesh_Sequence)
        n_samples_for_average = int(np.floor(np.shape(window_array)[0] / rotations_to_repeat))
        sig_len = np.shape(window_array)[1]

        averages = np.zeros((rotations_to_repeat, sig_len))
        all_per_teeth = np.zeros((n_samples_for_average, rotations_to_repeat, sig_len))
        print("n samples per average ",n_samples_for_average)
        for sample in range(n_samples_for_average):
            averages += window_array[sample * rotations_to_repeat:(sample + 1) * rotations_to_repeat, :]
            all_per_teeth[sample, :, :] = window_array[sample * rotations_to_repeat:(sample + 1) * rotations_to_repeat, :]

        # int_wind = window_array[0:n_samples_for_average*rotations_to_repeat, :]
        # r = np.reshape(int_wind.T, (n_samples_for_average, rotations_to_repeat, sig_len))

        if plot:
            fig, axs = plt.subplots(rotations_to_repeat, 2)

            for tooth_pair in range(rotations_to_repeat):
                sigs = all_per_teeth[:, tooth_pair, :].T
                axs[tooth_pair, 0].plot(sigs)
                axs[tooth_pair, 1].plot(np.average(sigs, axis=1))
                #axs[tooth_pair,0].set_ylim(-250,250)
                #axs[tooth_pair,1].set_ylim(-50,50)

        return averages,all_per_teeth

    def aranged_averaged_windows(self, window_averages, meshing_sequence):
        """ Takes the computed averages of the extracted windows and arranges them in order as determined by the meshing sequence.
        ----------
        window_averages: array
             Array of averaged windows obtained from the Window_average function

        meshing_sequence: Array
                         Meshing sequence of a planetary gearbox (For the Bonfiglioli gearbox, the meshing sequence array should be divided by 2 seeing that only the even numbered planet gear teeth with mesh with a given ring gear tooth.

        Returns
        -------
        averages_in_order: nxm Array
                n gear teeth of the planet gear each with an averaged window of m samples in order of increasing geartooth number
        planet_gear_revolution: Array of length n*m
                n gear teeth, m samples  in a window, all samples in the correct order concatenated together.
            """

        averages_in_order = window_averages[
            meshing_sequence]  # Order the array of averages according to the meshing sequence
        planet_gear_revolution = averages_in_order.reshape(-1)

        return averages_in_order, planet_gear_revolution

    def compute_tsa(self, fraction_offset, fraction_of_revolution, signal_name, plot=False):

        winds = self.window_extract(fraction_offset, fraction_of_revolution, signal_name)

        aves = self.window_average(winds)

        mesh_seq = list(np.ndarray.astype(np.array(self.PG.Meshing_sequence()) / 2, int))
        arranged, together = self.aranged_averaged_windows(aves, mesh_seq)

        if plot:
            plt.figure()
            minimum = np.min(together)
            maximum = np.max(together)
            plt.ylabel("Response_squared")
            plt.xlabel("Planet Gear Angle")
            angles = np.linspace(0, 360, len(together))
            plt.plot(angles, together)
            for line in range(np.shape(aves)[0]):
                plt.vlines(line * np.shape(aves)[1] * np.average(np.diff(angles)), minimum, maximum)

        return together


class Callibration(object):
    """
    Sets the relationships between a read voltage and some physical quantity.

    Callibration for torque performed with moment arm
    Callibration for temperature performed with boiling water
    """

    def Torque(self, Voltage):
        """

        Parameters
        ----------
        Voltage: The recorded voltage signal

        Returns
        -------
        T: The torque in Nm

        Note that this callibration was performed by fitting a straight line though 7 datapoints: See Torque_Callibration.xlsx
        """

        return 8.9067 * Voltage + 0.1318

    def Temperature(self, Voltage):
        """
        This callibration was performed using boiling water

        Parameters
        ----------
        Voltage: Recorded voltage signal

        Returns
        -------
        T: The torque in Nm
        """

        return 23.77 * Voltage - 25.009

    def change_df_column(self, Column_name, Callibration_function_to_apply):
        """
        Perform the scaling of a signal to read the actual values and not voltages as originally measured

        Parameters
        ----------
        Column_name: The name of the column of the dataframe to be changed. For instance "Torque"
        Callibration_function_to_apply: A function from the class Callibration that desribes the scaling between a voltage and the actaul value.

        Returns
        -------
        df: The same dataframe with changed column
        """
        self.dataset[Column_name] = Callibration_function_to_apply(self.dataset[Column_name])
        return


class TransientAnalysis(object):
    """
    Class for investigating the gear mesh transients
    """

    def get_peaks(self, signal, plot=False):
        samples_per_gearmesh = self.info["f_s"] * (1 / self.derived_attributes["GMF_ave"])

        indices, properties = sig.find_peaks(signal,
                                             height=1 * self.info["Acc_Carrier_SD"],
                                             distance=samples_per_gearmesh * 0.8)

        peaks = signal[indices]

        if plot:
            wind_len = len(signal)
            ts = 1 / self.info["f_s"]
            time_end = wind_len * ts
            trange = np.linspace(0, time_end, wind_len)
            plt.figure()
            plt.plot(trange, signal)
            # ind, peaks, prop = self.get_peaks(signal)
            plt.scatter(indices * ts, peaks, marker="x", c="black")

            plt.figure()
            plt.hist(np.diff(indices) / self.info["f_s"])
            plt.xlabel("Time between extracted mesh transient peaks [s]")
            plt.ylabel("Frequency of occurrence")

            # plt.figure("Peak value")
            # plt.hist(peaks)
            # plt.xlabel("Vibration magnitude [mg]")
            # plt.ylabel("Frequency of occurrence")

        return indices, peaks, properties

    def get_transients(self, signal, time_before_peak, time_after_peak, plot=False):
        """
        Extracts the gear mesh transients for a given signal section (window)
        Parameters
        ----------
        signal

        Returns
        -------

        """

        samples_before = int(time_before_peak * self.info["f_s"])
        samples_after = int(time_after_peak * self.info["f_s"])
        # Note that the transient length is independent of rotational speed as it is
        # assumed that the response should be time invariant to some extent
        ind, peaks, prop = self.get_peaks(signal)

        t_gm = np.diff(ind) / self.info["f_s"]

        ind = ind[1:-1]  # eliminate first and final peak
        ind_start = ind - samples_before
        ind_end = ind + samples_after

        transients_store = np.zeros((len(ind), samples_after + samples_before))
        for i in range(len(ind)):
            transient = signal[ind_start[i]:ind_end[i]]
            transients_store[i, :] = transient

        if plot:
            ts = 1 / self.info["f_s"]
            time_end = (samples_before + samples_after) * ts
            time_range = np.linspace(0, time_end, samples_before + samples_after)
            plt.figure()
            plt.plot(time_range, transients_store.T)

        return transients_store, peaks, t_gm

    def interpolate_transients(self, transients, plot=False):
        """
        Improve time resolution and smoothness of transients by interpolation
        Parameters
        ----------
        transients

        Returns
        -------

        """

        new_length = 200
        interpolate_store = np.zeros((np.shape(transients)[0], new_length))

        for sig, i in zip(transients, range(len(transients))):
            # Do FFT on signal segment (transient)

            interp_func = interp.interp1d(np.linspace(0, 1, len(sig)), sig, kind="cubic")
            d = interp_func(np.linspace(0, 1, new_length))

            interpolate_store[i, :] = d

        if plot:
            plt.figure("Interpolated transients")
            plt.plot(np.linspace(0, 1, new_length), interpolate_store.T)
            plt.xlabel("Time [s]")
            plt.ylabel("Magnitude")

        return interpolate_store

    def get_peak_freq_stats_over_all_windows(self, windows, plot_results=False, plot_checks=False):
        all_peaks = np.array([])
        all_t_gm = np.array([])
        all_prom_freqs = np.array([])

        time_before_peak = 0.0002
        time_after_peak = 0.0008
        for window in windows:
            trans, peaks, t_gm = self.get_transients(window, time_before_peak, time_after_peak)

            # prom_freqs = get_prominent_freqs_over_all_transients(trans)
            prom_freqs = 1 / self.get_prominent_period_over_all_transients(trans)

            all_peaks = np.hstack((all_peaks, peaks))
            all_t_gm = np.hstack((all_t_gm, t_gm))
            all_prom_freqs = np.hstack((all_prom_freqs, prom_freqs))

        if plot_results:
            plt.figure("Transient peak value")
            plt.hist(all_peaks)
            plt.xlabel("Transient Peak Vibration Magnitude [mg]")
            plt.ylabel("Frequency of occurrence")

            plt.figure("Most prominent frequency")
            plt.hist(all_prom_freqs)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Frequency of occurrence")

        if plot_checks:
            fig, axs = plt.subplots(2, 2)

            fig.suptitle(self.dataset_name)
            axs[0, 0].set_title("Time between extracted peaks")
            axs[0, 0].hist(all_t_gm)
            axs[0, 0].set_xlabel("Time between extracted mesh transient peaks [s]")
            axs[0, 0].set_ylabel("Frequency of occurrence")

            # Choose one of the windows randomly
            n = np.random.randint(0, np.shape(windows)[0])
            sig = windows[n, :]

            axs[1, 0].set_title("Transients selected in window")
            indices, peaks, properties = self.get_peaks(sig, plot=False)
            wind_len = len(sig)
            ts = 1 / self.info["f_s"]
            time_end = wind_len * ts
            trange = np.linspace(0, time_end, wind_len)
            axs[1, 0].plot(trange, sig)
            axs[1, 0].scatter(indices * ts, peaks, marker="x", c="black")
            axs[1, 0].hlines(self.info["Acc_Carrier_SD"], trange[0], trange[-1], colors="r", linestyles="dashed")

            axs[1, 1].set_title("Extracted Transients")
            transients, peaks, t_gm = self.get_transients(sig, time_before_peak, time_after_peak, plot=False)
            trans_len = np.shape(transients)[1]
            time_end = trans_len * ts
            time_range = np.linspace(0, time_end, trans_len)
            axs[1, 1].plot(time_range, transients.T)

            axs[0, 1].set_title("First 5 sec of measurement")
            n_samples = int(5 * self.info["f_s"])

            n_trig = np.where(self.derived_attributes["trigger_time_mag"] < 4.9)
            trigtimes = self.derived_attributes["trigger_time_mag"]
            axs[0, 1].plot(self.dataset["Time"].values[0:n_samples], self.dataset["Acc_Carrier"].values[0:n_samples])
            axs[0, 1].vlines(trigtimes[n_trig], self.info["min_acc_carrier"], self.info["max_acc_carrier"], "r")

            window_duration = self.derived_attributes["window_fraction"] * self.info["carrier_period_ave"]
            wind_high = trigtimes[1:-1] + 0.5 * window_duration
            wind_low = trigtimes[1:-1] - 0.5 * window_duration

            axs[0, 1].vlines(wind_low[n_trig], self.info["min_acc_carrier"], self.info["max_acc_carrier"], "c")
            axs[0, 1].vlines(wind_high[n_trig], self.info["min_acc_carrier"], self.info["max_acc_carrier"], "c")

        peaks_stats = np.array([np.mean(all_peaks), np.std(all_peaks), np.median(all_peaks)])
        prom_freqs_stats = np.array([np.mean(all_prom_freqs), np.std(all_prom_freqs), np.median(all_peaks)])
        return peaks_stats, prom_freqs_stats

    def get_sdof_stats_over_all_windows(self, windows, plot_results=False, plot_checks=False):
        all_mod_params = np.empty((4, 0))

        time_before_peak = 0.0002
        time_after_peak = 0.0008
        for window in windows:
            trans, peaks, t_gm = self.get_transients(window, time_before_peak, time_after_peak)

            mod_params = self.get_sdof_fit_over_all_transients(trans)

            all_mod_params = np.hstack((all_mod_params, mod_params))

        model_param_stats = np.array(
            [np.mean(all_mod_params, axis=1), np.std(all_mod_params, axis=1), np.median(all_mod_params, axis=1)])
        if plot_results:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].set_title("Damping ratio zeta")
            axs[0, 0].hist((all_mod_params[0, :]))
            axs[0, 0].set_xlabel("Dampling ratio zeta")
            axs[0, 0].set_ylabel("Frequency of occurrence")

            axs[1, 0].set_title("Undamped natural frequency omega_n")
            axs[1, 0].hist((all_mod_params[1, :] / (2 * np.pi)))
            axs[1, 0].set_xlabel("Undamped natural frequency omega_n [Hz]")
            axs[1, 0].set_ylabel("frequency of occurrence")

            axs[0, 1].set_title("d_0")
            axs[0, 1].hist((all_mod_params[2, :]))
            axs[0, 1].set_xlabel("d_0 [m]")
            axs[0, 1].set_ylabel("frequency of occurrence")

            axs[1, 1].set_title("v_0")
            axs[1, 1].hist((all_mod_params[3, :]))
            axs[1, 1].set_xlabel("v_0")
            axs[1, 1].set_ylabel("frequency of occurrence")

        if plot_checks:
            #     fig, axs = plt.subplots(2, 2)
            #
            #     fig.suptitle(self.dataset_name)
            #     axs[0, 0].set_title("Time between extracted peaks")
            #     axs[0, 0].hist(all_t_gm)
            #     axs[0, 0].set_xlabel("Time between extracted mesh transient peaks [s]")
            #     axs[0, 0].set_ylabel("Frequency of occurrence")
            #
            # Choose one of the windows randomly
            n = np.random.randint(0, np.shape(windows)[0])
            sig = windows[n, :]
            #
            #     axs[1, 0].set_title("Transients selected in window")
            #     indices, peaks, properties = self.get_peaks(sig, plot=False)
            #     wind_len = len(sig)
            #     ts = 1 / self.info["f_s"]
            #     time_end = wind_len * ts
            #     trange = np.linspace(0, time_end, wind_len)
            #     axs[1, 0].plot(trange, sig)
            #     axs[1, 0].scatter(indices * ts, peaks, marker="x", c="black")
            #     axs[1,0].hlines(self.info["Acc_Carrier_SD"],trange[0],trange[-1],colors ="r", linestyles = "dashed")
            #
            transients, peaks, t_gm = self.get_transients(sig, time_before_peak, time_after_peak, plot=False)
            ts = 1 / self.info["f_s"]
            trans_len = np.shape(transients)[1]
            time_end = trans_len * ts
            time_range = np.linspace(0, time_end, trans_len)

            params = self.get_sdof_fit_over_all_transients(transients[0:5, :])
            plt.figure()
            for trans, model in zip(transients[0:5, :], range(4)):
                plt.scatter(time_range, trans)
                sdof_obj = an_sdof.one_dof_sys(trans, self.info["f_s"])
                model_parameters = params[:, model]
                model = sdof_obj.xdd_func(model_parameters[0], model_parameters[1], time_range, model_parameters[2],
                                          model_parameters[3])

                plt.plot(time_range, model)
                plt.ylim(-1000, 1000)
                plt.xlabel("time [s]")
        #
        #     axs[0, 1].set_title("First 5 sec of measurement")
        #     n_samples = int(5 * self.info["f_s"])
        #
        #     n_trig = np.where(self.derived_attributes["trigger_time_mag"] < 4.9)
        #     trigtimes = self.derived_attributes["trigger_time_mag"]
        #     axs[0, 1].plot(self.dataset["Time"].values[0:n_samples], self.dataset["Acc_Carrier"].values[0:n_samples])
        #     axs[0, 1].vlines(trigtimes[n_trig], self.info["min_acc_carrier"], self.info["max_acc_carrier"], "r")
        #
        #     window_duration = self.derived_attributes["window_fraction"] * self.info["carrier_period_ave"]
        #     wind_high = trigtimes[1:-1] + 0.5 * window_duration
        #     wind_low = trigtimes[1:-1] - 0.5 * window_duration
        #
        #     axs[0, 1].vlines(wind_low[n_trig], self.info["min_acc_carrier"], self.info["max_acc_carrier"], "c")
        #     axs[0, 1].vlines(wind_high[n_trig], self.info["min_acc_carrier"], self.info["max_acc_carrier"], "c")
        #
        # peaks_stats = np.array([np.mean(all_peaks), np.std(all_peaks)])
        # prom_freqs_stats = np.array([np.mean(all_prom_freqs), np.std(all_prom_freqs)])
        # return peaks_stats, prom_freqs_stats
        return model_param_stats

    def get_prominent_freqs_over_all_transients(self, transients, plot=False):
        """
        Get the most prominent frequency of vibration of the transient
        Parameters
        ----------
        sig

        Returns
        -------

        """
        # Remove peaks in frequency spectrum with period longer than transient length
        length = np.shape(transients)[1]
        t_transient = length / self.info["f_s"]
        f_lower_cutoff = 2.5 * 1 / t_transient

        fs = self.info["f_s"]
        freq = np.fft.fftfreq(length, 1 / fs)[0:int(length / 2)]
        print("freq_res:", np.average(np.diff(freq)))
        cut_off_index = np.argmax(np.diff(np.sign(freq - f_lower_cutoff))) + 1

        fft_mag_store = np.zeros((np.shape(transients)[0], int(length / 2)))

        for sig, i in zip(transients, range(len(transients))):
            # Do FFT on signal segment (transient)
            d = sig

            Y = np.fft.fft(d) / length
            magnitude = np.abs(Y)[0:int(length / 2)]
            # phase = np.angle(Y)[0:int(length / 2)]

            fft_mag_store[i, :] = magnitude

        spectra_peak_locs = np.argmax(fft_mag_store[:, cut_off_index:], axis=1)
        most_promiment_freqs = freq[cut_off_index:][spectra_peak_locs]

        if plot:
            plt.figure("Frequency spectra of transients")
            plt.plot(freq, fft_mag_store.T)
            plt.scatter(freq, fft_mag_store.T[:, 0])
            # plt.vlines(f_lower_cutoff, np.min(fft_mag_store), np.max(fft_mag_store))
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")

            plt.figure("Most prominent frequency in transient")
            plt.hist(most_promiment_freqs)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Frequency of occurrence")

        return most_promiment_freqs

    def get_prominent_period_over_all_transients(self, transients, plot=False):
        """
        Get the most prominent frequency of vibration of the transient
        Parameters
        ----------
        sig

        Returns
        -------

        """

        fs = self.info["f_s"]

        all_period_store = np.array([])

        for signal, i in zip(transients, range(len(transients))):
            # Find peaks in signal

            indices, properties = sig.find_peaks(np.abs(signal))

            period = 2 * np.diff(indices) / fs  # Period in seconds based on positive and negative peaks

            all_period_store = np.hstack((all_period_store, period))

        if plot:
            # plt.figure("Frequency spectra of transients")
            # plt.plot(freq, fft_mag_store.T)
            # plt.scatter(freq, fft_mag_store.T[:,0])
            # #plt.vlines(f_lower_cutoff, np.min(fft_mag_store), np.max(fft_mag_store))
            # plt.xlabel("Frequency [Hz]")
            # plt.ylabel("Magnitude")

            plt.figure("Most prominent frequency in transient")
            plt.hist(all_period_store)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Frequency of occurrence")

        return all_period_store

    def get_sdof_fit_over_all_transients(self, transients, plot=False):
        """
        Fit the solution to a sdof lmm to the data
        Parameters
        ----------
        sig

        Returns
        -------

        """

        fs = self.info["f_s"]

        all_param_store = np.empty((4, 0))

        for signal, i in zip(transients, range(len(transients))):
            # Find peaks in signal

            sdof_obj = an_sdof.one_dof_sys(signal, fs)
            sol = sdof_obj.do_least_squares()
            opt_params = np.array([sol["x"]]).T

            all_param_store = np.hstack((all_param_store, opt_params))

        # if plot:
        #     # plt.figure("Frequency spectra of transients")
        #     # plt.plot(freq, fft_mag_store.T)
        #     # plt.scatter(freq, fft_mag_store.T[:,0])
        #     # #plt.vlines(f_lower_cutoff, np.min(fft_mag_store), np.max(fft_mag_store))
        #     # plt.xlabel("Frequency [Hz]")
        #     # plt.ylabel("Magnitude")
        #
        #     plt.figure("Most prominent frequency in transient")
        #     plt.hist(all_period_store)
        #     plt.xlabel("Frequency [Hz]")
        #     plt.ylabel("Frequency of occurrence")

        return all_param_store
        # return all_period_store


class Dataset(Tachos_And_Triggers, Dataset_Plotting, Signal_Processing, Time_Synchronous_Averaging, Callibration):
    """This class creates objects that include a particular dataset, then planetary gearbox configuration used and derived attributes from the dataset"""

    def __init__(self, dataset, PG_Object, name):

        """
        Initializes the Dataset object

        :param dataset: a pandas dataframe with 9 columns as obtained from testing.
        :param PG_Object: Created with the PG class. Includes information such as number of teeth of respective gears
        """
        # Give the dataset a name to identify it later
        self.dataset_name = name

        # Set the dataframe to be an attribute of the dataset
        self.dataset = dataset

        # Apply the scaling functions as obtained through callibration
        self.change_df_column("T_amb", self.Torque)
        self.change_df_column("T_oil", self.Temperature)

        self.PG = PG_Object
        self.info = {}  # Dictionary that stores information about the dataset
        self.derived_attributes = {}  # Dictionary that stores computed attributes from the dataset

        # Save the info of the dataset to the object
        self.compute_info()

        # Run the preprocessing steps as selected in the compute_derived_attributes method
        self.compute_derived_attributes()

    def compute_derived_attributes(self):
        """
        Add the functions that should be run on object initialization. Certain attributes might be expensive to calculate
        :return:
        """

        # Compute trigger times for the magnetic pickup
        trigger_index, trigger_time = self.trigger_times("1PR_Mag_Pickup", 8)
        self.derived_attributes.update({"trigger_time_mag": trigger_time, "trigger_index_mag": trigger_index})

        order_t, order_sig, samples_per_rev = self.order_track("Acc_Sun")
        self.derived_attributes.update({"order_track_time": order_t, "order_track_signal": order_sig,
                                        "order_track_samples_per_rev": samples_per_rev})

        # Compute the number of fatigue cycles (Note that this is stored in info and not derived attributes
        # n_carrier_revs = np.shape(self.derived_attributes["trigger_index_mag"])[0]
        # n_fatigue_cycles = self.PG.fatigue_cycles(n_carrier_revs)
        # self.info.update({"n_fatigue_cycles": n_fatigue_cycles})

        #  Compute the RPM over time according to the magnetic pickup
        #try:
        rpm, trpm, average_rpm = self.getrpm("1PR_Mag_Pickup", 8, 1, 1, self.info["f_s"])
        self.derived_attributes.update({"rpm_mag": rpm, "t_rpm_mag": trpm})

        self.info.update(
            {"rpm_carrier_ave": average_rpm})  # Notice that info is updated not in compute_info function
        self.info.update(
            {"rpm_sun_ave": average_rpm * self.PG.GR})  # Notice that info is updated not in compute info function
        self.info.update({"rpm_sun_sd": np.std(rpm * self.PG.GR)})

        self.info.update({"carrier_period_ave": 1 / (average_rpm / 60)})

        # Compute Gear mesh frequency
        self.derived_attributes.update({"GMF_ave": self.PG.GMF(self.info["rpm_sun_ave"] / 60)})

        # Compute planet pass frequency
        self.derived_attributes.update({"PPF_ave": self.info["rpm_carrier_ave"] / 60})

        # Get the vibration windows as planet gear passes transducer
        window_frac = 0.30  # Make use of a tenth of the revolutions vibration
        window_offset_frac = 0  # How far to offset the window from the magnetic switch pulse
        self.derived_attributes.update({"window_fraction": window_frac,
                                        "window_offset_frac": window_offset_frac})

        winds = self.window_extract(window_offset_frac, window_frac, "Acc_Carrier")  # Notice that the Carrier
        self.derived_attributes.update({"extracted_windows": winds})                 # accelerometer is used

    #except:
        #print("Possible problem with tachometer trigger threshold")
        #self.derived_attributes.update({"rpm_mag": "NaN", "t_rpm_mag": "NaN"})
        #self.info.update({"rpm_carrier_ave": "NaN"})  # Notice that info is updated not in compute_info function
        #self.info.update({"rpm_sun_ave": "NaN"})  # Notice that info is updated not in compute info function

        # Compute TSA for sun gear acc
        # TSA = self.Compute_TSA()
        # self.derived_attributes.update({"TSA_Sun": TSA})

    def compute_info(self):
        """
        Computes the info for a certain dataset and stores it in the info dict. This is run in the init of the Dataset class
        :return:
        """

        # Compute the sampling rate of the dataset
        fs = self.sampling_rate()
        self.info.update({"f_s": fs})

        # Compute the time duration of the dataset
        duration = self.dataset["Time"].values[-1]
        self.info.update({"duration": duration})

        # Min and Max Values
        mini = np.min(self.dataset["Acc_Carrier"].values)
        maxi = np.max(self.dataset["Acc_Carrier"].values)
        SD = np.std(self.dataset["Acc_Carrier"].values)

        self.info.update({"min_acc_carrier": mini})
        self.info.update({"max_acc_carrier": maxi})
        self.info.update({"Acc_Carrier_SD": SD})


class PG(object):
    """This class creates planetary gearbox objects in order to determine their expected frequency response"""

    def __init__(self, Z_r, Z_s, Z_p):
        self.Z_r = Z_r
        self.Z_s = Z_s
        self.Z_p = Z_p
        self.GR = self.GR_calc()

        self.carrier_revs_to_repeat = 12  # The number of revolution of the carrier required for a given planet gear
        # tooth to mesh with the same ring gear tooth. This could be calculated based
        # on the input parameters with the meshing sequence function is extended

        self.Mesh_Sequence = self.Meshing_sequence()  # Calculate the meshing sequence

    def GMF1(self, f_sun):
        """Function that returns the gear mesh frequency for a given sun gear rotation speed f_s
        Parameters
        ----------
        f_s: float
            Sun gears shaft frequency


        Returns
        -------
        GMF: Float
            The gear mesh frequency in Hz
            """
        return f_sun * self.Z_r * self.Z_s / (self.Z_r + self.Z_s)

    def GR_calc(self):
        """Gear ratio for planetary gearbox with stationary ring gear
        Parameters
        ----------


        Returns
        -------
        GR: Float
            The ratio
            """
        return 1 + self.Z_r / self.Z_s

    def f_p(self, f_c):
        """Frequency of rotation of planetary gears
        Parameters
        ----------
        f_c: Float
             Frequency of rotation of planet carrier


        Returns
        -------
        f_p: Float
            Frequency of rotation of planet gear
            """
        return f_c * self.Z_r / self.Z_p

    def GMF(self, f_s):
        """Calculates the gear mesh frequency for a given a sun gear input frequency. The gearbox is therefore running in the speed down configuration
        Parameters
        ----------
        f_s: Float
             Frequency of rotation of sun gear


        Returns
        -------
        GMF: Float
            Frequency of rotation of planet gear
            """
        fc = f_s / self.GR
        return self.f_p(fc) * self.Z_p

    def FF1(self, f_s):
        """Calculates the gear mesh frequency for a given a sun gear input frequency. The gearbox is therefore running in the speed down configuration
        Parameters
        ----------
        f_sun: Float
             Frequency of rotation of sun gear


        Returns
        -------
        FF1: Float
            Fault frequency due to fault on planet gear
            """
        f_c = f_s / self.GR  # Calculate the frequency of rotation of the carrier
        fp = self.f_p(f_c)
        FF1 = 2 * fp  # The fault will manifest in the vibration twice per revolution of the planet gear:
        # once at the sun gear and once at the ring gear
        return FF1

    def Meshing_sequence(self):
        """Determines the order in which the teeth of a planet gear will mesh with the ring gear
        Parameters
        ----------


        Returns
        -------
        Mesh_Sequence: array
            Array of tooth numbers (zero to Np-1) that show the order in which the teeth will mesh
            """

        Mesh_Sequence = []
        for n_rev in range(self.carrier_revs_to_repeat):  # Notice that the sequence starts repeating after 12 rotations
            Mesh_Sequence.append((n_rev * self.Z_r) % self.Z_p)

        return Mesh_Sequence

    def fatigue_cycles(self, carrier_revs):
        """
        Calculates the number of fatigue cycles given the number of planet rotations
        Parameters
        ----------
        carrier_revs: float
                    Number of rotations of the carrier

        Returns
        -------
        fatigue_cycles
        """
        return float(carrier_revs) * (self.Z_r / self.Z_p)  # (number of revs)(number of cycles per revolution)

    @classmethod
    def RPM_to_f(cls, RPM):
        """Function converts revolutions per minute to Herz
        Parameters
        ----------
        RPM: Float
            Rotational speed in RPM


        Returns
        -------
        f:  Float
            Frequency [Hz]
            """
        return RPM / 60


plt.close("all")

Zr = 62
Zs = 13
Zp = 24

Bonfiglioli = PG(Zr, Zs, Zp)

Input_RPM = 550  # RPM

image_save_path = r"C:\Users\douwm\Google Drive\Meesters\Meeting_Preparations\Date_Here"
Z_crack_im_path = r"C:\Users\douwm\Google Drive\Meesters\Crack_Photos_Preliminary_Test\Z_Check_Growth"
