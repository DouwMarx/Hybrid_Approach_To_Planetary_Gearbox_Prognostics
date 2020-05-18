import definitions
import numpy as np
import scipy.optimize as opt

import time
import matplotlib.pyplot as plt
import src.models.lumped_mas_model as pglmm
import models.lumped_mas_model.llm_models as lmm_models


class Inverse_Probelm(object):
    """
    Class for computing the solution to the inverse problem where the optimum crack length is computed for a given measured signal
    """

    def __init__(self, feature_extraction_object, lmm_object, reconciliation_model_object, dataset_object):
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


class Diagnostics(object):
    """
    Used to find the optimum parameters for a given set of measurements
    """

    def __init__(self, pg_data, optimize_for, PG_info_dict):
        self.data = pg_data  # Depending what should be used in minimizing the cost, this could take a different value

        # pdict = {"level1": {"a": 3},
        #        "b": 5,
        #        "c": 2}

        self.optimize_for = optimize_for
        self.pglmm_info = PG_info_dict

        self.opt_for_var = list(self.optimize_for.keys())  # Variables to optimize for
        self.opt_for_bnds = list(self.optimize_for.values())  # Bounds of variables

        # Finds the dimensionality of the unknowns to compile the current parameter vector candidate
        self.opt_for_var_dimensionality = []

        self.bound_array = self.opt_for_bnds[0]
        try:
            self.opt_for_var_dimensionality.append(np.shape(self.opt_for_bnds[0].T)[1])
        except:
            self.opt_for_var_dimensionality.append(1)

        for bound in self.opt_for_bnds[1:]:
            self.bound_array = np.vstack((self.bound_array, bound))
            try:
                self.opt_for_var_dimensionality.append(np.shape(bound.T)[1])
            except:
                self.opt_for_var_dimensionality.append(1)
        return

    def f_min(self, theta):

        # Exchage the values in the dictionary used to initialize the LMM with the values the optimizer wants to use
        dim_count = 0
        for var_to_opt_for, theta_index, dimensionality in zip(self.opt_for_var, range(len(self.opt_for_var)),
                                                               self.opt_for_var_dimensionality):
            #  edit_dict_var(self.pglmm_info, var_to_opt_for, theta[theta_index])
            if dimensionality == 1:
                edit_dict_var(self.pglmm_info, var_to_opt_for, theta[dim_count])  # TODO: Fix to work for the multidimensional intial values
            else:
                edit_dict_var(self.pglmm_info, var_to_opt_for, np.array([theta[dim_count: dim_count + dimensionality]]).T)
            dim_count += dimensionality

        PG = pglmm.Planetary_Gear(self.pglmm_info)
        sol = PG.get_transducer_vibration()
        return self.cost(sol)

    def cost(self, candidate):
        """
        The cost function. How is the model punished for not matching the data.
        Different cost functions can be implemented.
        Parameters
        ----------
        candidate: This is the candidate solution of the model given certain model parameters.

        Returns
        -------

        """
        return np.linalg.norm(candidate * 1000 - self.data * 1000)

    def do_optimisation(self):
        sol = opt.differential_evolution(self.f_min,
                                         self.opt_for_bnds,
                                         polish=True,
                                         disp=True)
        return sol


def edit_dict_var(adict, k, v):
    """
    Replaces the value of a nested dictionary with a new one.
    Based on answer by https://stackoverflow.com/users/1481060/sotapme
    Parameters
    ----------
    adict
    k
    v

    Returns
    -------

    """
    for key in adict.keys():
        if key == k:
            adict[key] = v
        elif type(adict[key]) is dict:
            edit_dict_var(adict[key], k, v)


def system(param_dict):
    return param_dict["level1"]["a"] * np.linspace(-1, 1, 100) ** 2 + param_dict["b"] * np.linspace(-1, 1, 100) + \
           param_dict["c"]
