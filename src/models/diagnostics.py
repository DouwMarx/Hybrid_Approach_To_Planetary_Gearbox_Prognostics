import numpy as np
import scipy.optimize as opt


class Inverse_Probelm(object):
    """
    Class for computing the solution to the inverse problem where the optimum crack length is computed for a given measured signal
    """
    def __init__(self,feature_extraction_object,lmm_object,reconciliation_model_object,dataset_object):
        """
        Initialize the inverse Problem object

        Parameters
        ----------
        feature_extraction_object:
        lmm_object
        fem_surrogate_object
        reconciliation_model_object
        dataset_object
        """

        self.feature_extraction = feature_extraction_object
        self.feature_extraction = feature_extraction_object
        self.lmm = lmm_object
        self.reconciliation_model = reconciliation_model_object
        self.dataset = dataset_object
        return

    def objective_function(self):
        """
        Objective function to be minimized with respect to crack length
        Returns
        -------
        cost : float
        """
        physics_features = self.feature_extraction.get_features(self.lmm.get_response())
        measured_features = self.feature_extraction.get_features(self.dataset)
        reconciled_features = self.reconciliation_model.get_adjusted_features(measured_features)

        return np.linalg.norm(physics_features - reconciled_features)
