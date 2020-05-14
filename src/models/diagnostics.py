import numpy as np
import scipy.optimize as opt
import collections
import time
import matplotlib.pyplot as plt


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

    def __init__(self, pg_data, optimize_for):
        self.data = pg_data  # Depending what should be used in minimizing the cost, this could take a different value

        pdict = {"level1": {"a": 3},
                 "b": 5,
                 "c": 2}

        self.optimize_for = optimize_for
        self.pglmm_info = pdict

        self.opt_for_var = list(self.optimize_for.keys())
        self.opt_for_bnds = list(self.optimize_for.values())

        return

    def f_min(self, theta):

        # Exchage the values in the dictionary used to initialize the LMM with the values the optimizer wants to use
        for var_to_opt_for, theta_index in zip(self.opt_for_var, range(len(self.opt_for_var))):
            edit_dict_var(self.pglmm_info, var_to_opt_for, theta[theta_index])

        f = system(self.pglmm_info)  # Initialize the lumped mass model and get its solution given the model parameters

        return self.cost(f)

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
        return np.linalg.norm(candidate - self.data)


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


pdict = {"level1": {"a": 3},
         "b": 5,
         "c": 2}

opim_for = {"a": [1, 4],
            "b": [0, 10]}

measured = np.random.normal(system(pdict), 0.5)
plt.figure()
plt.scatter(np.linspace(-1, 1, 100), measured)
plt.plot(np.linspace(-1, 1, 100), system(pdict))



diag_obj = Diagnostics(measured, opim_for)
print(diag_obj.do_optimisation())

plt.plot(np.linspace(-1,1,100), system(diag_obj.pglmm_info))
