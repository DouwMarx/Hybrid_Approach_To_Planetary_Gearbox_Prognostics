import numpy as np
import scipy.signal as s
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import pickle
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
        plt.text(0,max(rpm),"Average motor speed: " + str(int(self.info["rpm_sun_ave"]))+ " RPM")

    def plot_fft(self, data,fs,plot_gmf = False):
        """
        Computes and plots the FFT for a given signal
        Parameters
        ----------
        data: String
            Name of the heading of the dataset to be FFTed

        Returns
        -------

        """
        freq, mag, phase = self.fft(data,fs)


        plt.figure()
        #max_height = 2
        plt.plot(freq, mag, "k")
        plt.ylabel("Magnitude")
        plt.xlabel("Frequency [Hz]")

        if plot_gmf == True:
            GMF = self.PG.GMF(self.info["rpm_sun_ave"]/60)
            #FF1 = GMF/self.PG.Z_p
            FF = self.PG.FF1(self.info["rpm_sun_ave"]/60)
            max_height = np.max(mag)
            plt.vlines(np.arange(1, 5) * GMF, 0, max_height, 'r', zorder=10, label="GMF and Harmonics")
            #plt.vlines(np.arange(1, 3) * FF, max_height, 'c', zorder=10, label="GMF and Harmonics")
            #plt.vlines(np.arange(1, 4) * FF1, 0, max_height, 'g', zorder=10, label="FF1 and Harmonics")

        #plt.xlim(0, 6000)
        #plt.show()

        return


    def plot_order_spectrum(self, data,fs,samples_per_rev,plot_gmf = False):
        """
        Computes and plots the FFT for a given signal
        Parameters
        ----------
        data: String
            Name of the heading of the dataset to be FFTed

        Returns
        -------

        """
        freq, mag, phase = self.fft(data,fs)


        plt.figure()
        #max_height = 2
        plt.plot(freq/samples_per_rev, mag, "k")
        plt.ylabel("Magnitude")
        plt.xlabel("Carrier orders")

        if plot_gmf == True:
            GMF = self.PG.GMF(self.info["rpm_sun_ave"]/60)

            FF = self.PG.FF1(self.info["rpm_sun_ave"]/60)
            #FF1 = GMF/self.PG.Z_p
            max_height = np.max(mag)
            plt.vlines(np.arange(1, 8) * GMF/samples_per_rev, 0, max_height, 'r', zorder=10, label="GMF and Harmonics")
            plt.vlines(np.arange(1, 3) * FF/samples_per_rev, 0, max_height, 'c', zorder=10, label= "Fault frequency and Harmonics" )
            #plt.vlines(np.arange(1, 4) * FF1, 0, max_height, 'g', zorder=10, label="FF1 and Harmonics")

        #plt.xlim(0, 6000)
        #plt.show()

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

class Tachos_And_Triggers(object):
    """This class is used to work with magnetic switch and tacho data"""

    def trigger_times(self, tacho, trig_level):
        """Calculates the times at which the damaged planet gear passes the magnetic switch
        Parameters
        ----------
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

        if len(Pos)<3:
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
        #print(self.dataset["Time"])
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

    def order_track(self,data):
        d = self.dataset[data].values
        t = self.dataset["Time"]
        fs = self.info["f_s"]

        trigger_times = self.derived_attributes["trigger_time_mag"]

        tnew = np.array([])
        #tnew = np.array([0])
        ave_rot_time = np.average(np.diff(trigger_times))
        print(ave_rot_time)
        samples_per_rev = int(fs*ave_rot_time)
        print(samples_per_rev)
        for index in range(len(trigger_times) - 1):
            section = np.linspace(trigger_times[index], trigger_times[index + 1], samples_per_rev)
            tnew = np.hstack((tnew, section))


        interp_obj = interp.interp1d(t, d, kind='cubic')
        interp_sig = interp_obj(tnew)

        return tnew, interp_sig, samples_per_rev

class Time_Synchronous_Averaging(object):
    """
    Used to compute the time synchronous average for a planetary gearbox.
    See A TECHNIQUE FOR CALCULATING THE TIME DOMAIN AVERAGES OF THE VIBRATION OF THE INDIVIDUAL PLANET GEARS AND THE SUN GEAR IN AN EPICYCLIC GEARBOX, McFadden 1994

    Take note that the sun gear accelerometer is used
    """

    def Window_extract(self,sample_offset):
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

        Returns
        -------
        windows: nxm Array
            n windows each with m samples
            """

        fs = self.info["f_s"]
        Z_p = self.PG.Z_p
        #acc = self.dataset["Acc_Sun"].values
        acc = self.derived_attributes["order_track_signal"]

        fc_ave = 1 / (self.info["rpm_carrier_ave"] /60)

        f_p_ave = self.PG.f_p(fc_ave)

#        window_length = 2*int(fs * (1 / f_p_ave) / Z_p)
        window_length = 1 * int(fs/2 * (1 / f_p_ave) / Z_p)

        if window_length % 2 == 0:  # Make sure that the window length is uneven
            window_length += 1

        #print("window length calculated as ", window_length, "samples")

        window_half_length = int((window_length - 1) / 2)
        #window_center_index = self.derived_attributes["trigger_index_mag"] + sample_offset
        window_center_index = np.arange(0, len(acc), fs/2).astype(int)

        n_revs = np.shape(window_center_index)[ 0] - 2  # exclude the first and last revolution to prevent errors with insufficient window length

        windows = np.zeros((n_revs, window_length))  # Initialize an empty array that will hold the extracted windows

        window_count = 0
        for index in window_center_index[1:-1]:  # exclude the first and last revolution to prevent errors with insufficient window length
            windows[window_count, :] = acc[index - window_half_length:index + window_half_length + 1]
            window_count += 1
        return windows**2

    def Window_average(self, window_array, rotations_to_repeat):
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

        n_samples_for_average = int(np.floor(np.shape(window_array)[0] / rotations_to_repeat))
        #n_samples_for_average =1
        #print(n_samples_for_average)

        averages = np.zeros((rotations_to_repeat, np.shape(window_array)[1]))

        for sample in range(n_samples_for_average):
            averages += window_array[sample * rotations_to_repeat:(sample + 1) * rotations_to_repeat, :]

        return averages

    def Aranged_averaged_windows(self, window_averages, meshing_sequence):
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

    def Compute_TSA(self, sample_offset, plot = False):

        winds = self.Window_extract(sample_offset)

        aves = self.Window_average(winds, 12)

        mesh_seq = list(np.ndarray.astype(np.array(self.PG.Meshing_sequence()) / 2, int))
        arranged, together = self.Aranged_averaged_windows(aves, mesh_seq)

        if plot:
            plt.figure()
            minimum = np.min(together)
            maximum = np.max(together)

            plt.plot(together)
            for line in range(np.shape(aves)[0]):
                plt.vlines(line*np.shape(aves)[1], minimum, maximum)


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

        return 8.9067*Voltage + 0.1318

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

        return 23.77*Voltage - 25.009

    def change_df_column(self,Column_name ,Callibration_function_to_apply):
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


class Dataset(Tachos_And_Triggers, Dataset_Plotting, Signal_Processing, Time_Synchronous_Averaging, Callibration):
    """This class creates objects that include a particular dataset, then planetary gearbox configuration used and derived attributes from the dataset"""
    def __init__(self, dataset, PG_Object):
        """
        Initializes the Dataset object

        :param dataset: a pandas dataframe with 9 columns as obtained from testing.
        :param PG_Object: Created with the PG class. Includes information such as number of teeth of respective gears
        """
        self.dataset = dataset
        # Apply the scaling functions as obtained through callibration
        self.change_df_column("T_amb", self.Torque)
        self.change_df_column("T_oil", self.Temperature)

        self.PG = PG_Object
        self.info = {}   # Dictionary that stores information about the dataset
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
        trigger_index, trigger_time = self.trigger_times("1PR_Mag_Pickup", 1)
        self.derived_attributes.update({"trigger_time_mag" : trigger_time, "trigger_index_mag" : trigger_index})

        order_t, order_sig, samples_per_rev = self.order_track("Acc_Sun")
        self.derived_attributes.update({"order_track_time": order_t, "order_track_signal": order_sig, "order_track_samples_per_rev": samples_per_rev})

        # Compute the number of fatigue cycles (Note that this is stored in info and not derived attributes
        #n_carrier_revs = np.shape(self.derived_attributes["trigger_index_mag"])[0]
        #n_fatigue_cycles = self.PG.fatigue_cycles(n_carrier_revs)
        #self.info.update({"n_fatigue_cycles": n_fatigue_cycles})

        #  Compute the RPM over time according to the magnetic pickup
        try:
            rpm, trpm, average_rpm = self.getrpm("1PR_Mag_Pickup", 8, 1, 1, self.info["f_s"])
            self.derived_attributes.update({"rpm_mag": rpm, "t_rpm_mag": trpm})
            self.info.update({"rpm_carrier_ave": average_rpm})  # Notice that info is updated not in compute_info function
            self.info.update({"rpm_sun_ave": average_rpm*self.PG.GR})  # Notice that info is updated not in compute info function
        except:
            self.derived_attributes.update({"rpm_mag": "NaN", "t_rpm_mag": "NaN"})
            self.info.update({"rpm_carrier_ave": "NaN"})  # Notice that info is updated not in compute_info function
            self.info.update({"rpm_sun_ave": "NaN"})  # Notice that info is updated not in compute info function

        # Compute TSA for sun gear acc
        #TSA = self.Compute_TSA()
        #self.derived_attributes.update({"TSA_Sun": TSA})

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
        return f_sun*self.Z_r*self.Z_s/(self.Z_r + self.Z_s)

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

    def f_p(self,f_c):
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
        return f_c*self.Z_r/self.Z_p

    def GMF(self,f_s):
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
        fc = f_s/self.GR
        return self.f_p(fc)*self.Z_p

    def FF1(self,f_s):
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
        f_c = f_s/self.GR # Calculate the frequency of rotation of the carrier
        fp = self.f_p(f_c)
        FF1 = 2*fp # The fault will manifest in the vibration twice per revolution of the planet gear:
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
        for n_rev in range(self.carrier_revs_to_repeat): #Notice that the sequence starts repeating after 12 rotations
            Mesh_Sequence.append((n_rev*self.Z_r)%self.Z_p)

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
        return float(carrier_revs)*(self.Z_r/self.Z_p) # (number of revs)(number of cycles per revolution)


    @classmethod
    def RPM_to_f(cls,RPM):
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
        return RPM/60


plt.close("all")

Zr = 62
Zs = 13
Zp = 24

Bonfiglioli = PG(Zr,Zs,Zp)

Input_RPM = 550  #RPM

image_save_path = r"C:\Users\douwm\Google Drive\Meesters\Meeting_Preparations\Date_Here"
Z_crack_im_path = r"C:\Users\douwm\Google Drive\Meesters\Crack_Photos_Preliminary_Test\Z_Check_Growth"



