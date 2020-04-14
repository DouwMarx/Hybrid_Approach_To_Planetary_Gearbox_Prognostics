import numpy as np
import scipy.signal as sig


class Features(object):
    """
    Class for creating features for analytical and measured response respectively
    """

    def __init__(self, desired_features):
        """
        Initializes Features class based on the type of data being used and the desired features to be computed

        Parameters
        ----------
        data: either a .pgdata  (measdat?) object of a physdat object
        desired_features: Dictionary of features to compute and their parameters
        """


        self.desired_features = desired_features

        return

    def get_features(self, data):
        self.data_type = type(data)  # Determine whether the provided data is extracted features, dataset or physics model output
        if self.data_type == ".pgdata":
            data = data.dataset["Acc_Sun"]

        if self.data_type == ".physdat":
            data = data.response

        if self.data_type == ".measfeat":
            return data.feature_vector


        feature_vector = []
        if "RMS" in self.desired_features.keys():
            feature_vector.append(self.rms(data))

        return np.array(feature_vector)


    def rms(self,signal):
        return np.sqrt(np.mean(signal**2))

