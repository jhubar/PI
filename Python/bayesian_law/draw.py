import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
import random

class bayesian_uncertainty():

    def __init__(self):

        # Model's hyperparameters
        self.beta = 0  # Contagion probability
        self.sigma = 0  # Probability to go from E to I
        self.gamma = 0  # Probability parameter to go from I to R (to be cure)
        self.hp = 0  # Probability to go from I to H
        self.hcr = 0  # Hospit Cure Rate
        self.pc = 0  # Probability to fall in ICU each day from H
        self.pd = 0  # Probability to die each day in icu
        self.pcr = 0  # Probability to recover from critical

        # Data to fit
        self.raw_dataset = None  # Original dataset, before preprocessing
        self.dataset = None  # Numpy matrix format
        self.dataframe = None  # Dataframe format

        # Initial state: to be used to make predictions
        self.S_0 = None  # Sensible: peoples who can catch the agent
        self.E_0 = None  # Exposed: people in incubation: Can't spread the agent
        self.I_0 = None  # Infectious: people who can spread the disease
        self.H_0 = None  # Hospitalized peoples: can't spread any more the agent
        self.C_0 = None  # Critical: peoples who are in ICU
        self.R_0 = None  # Recovered people: can't catch again the agent due to immunity
        self.D_0 = None  # Dead: people who die in ICU
        self.N = None  # The total size of the population

        # Data to store
        self.dataJSON = {}
    def dataframe_uncertainty(self, df):

        # Convert the dataframe to a numpy array:
        np_df = df.to_numpy()
        np_df_upper = np.copy(np_df)
        np_df_lower = np.copy(np_df)
        sensitivity_upper_bound = 0.7
        sensitivity_lower_bound = 0.85

        day = np_df[:,0]
        num_positive_lower = np.array(np_df[:,1]+np_df[:,1]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))
        num_positive_upper = np.array(np_df[:,1]+np_df[:,1]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))

        num_tested = np_df[:,2]
        num_hospitalised_lower = np.array(np_df[:,3]+np_df[:,3]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))
        num_hospitalised_upper = np.array(np_df[:,3]+np_df[:,3]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))

        num_cumulative_hospitalizations = np_df[:,4]
        num_critical_lower = np.array(np_df[:,5]+np_df[:,5]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))
        num_critical_upper = np.array(np_df[:,5]+np_df[:,5]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))

        num_fatalities_lower = np.array(np_df[:,6]+np_df[:,6]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))
        num_fatalities_upper = np.array(np_df[:,6]+np_df[:,6]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))

        new_df = np.vstack((day,num_positive_lower,num_positive_upper,num_tested,num_hospitalised_upper,num_hospitalised_lower,num_cumulative_hospitalizations,num_critical_lower,num_critical_upper,num_fatalities_lower,num_fatalities_upper))

        return new_df

    def import_dataset(self, target='covid_20'):

        if target == 'covid_20':
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # Import the dataframe:
            self.raw_dataset = pd.read_csv(url, sep=',', header=0)

            # ========================================================================= #
            #  Uncertainty
            # ========================================================================= #


            self.dataframe = self.dataframe_uncertainty(self.raw_dataset)

            print(self.dataframe)

def first_method():
    # Initialize the model
    data_with_uncertainty = bayesian_uncertainty()

    # Import the dataset:
    data_with_uncertainty.import_dataset(target='covid_20')


if __name__ == "__main__":
    first_method()

# nb_sympt_tested = random.uniform(0.5, 1.0)
# nb_not_sympt_tested = 1-nb_sympt_tested
#
# sensitivity = random.uniform(0.7, 0.85)
# not_sensitivity = 1 - sensitivity
#
#
#
# test_y = np.array([0]*int(nb_not_sympt_tested*100) + [1]*int(nb_sympt_tested*100+1))
# print(test_y)
# predicted_y_probs = np.concatenate((np.random.beta(not_sensitivity*10,sensitivity*10,int(nb_sympt_tested*100)), np.random.beta(sensitivity*10,not_sensitivity*10,int(nb_not_sympt_tested*100+1))))
#
#
#
# def estimate_beta(X):
#     xbar = np.mean(X)
#     vbar = np.var(X,ddof=1)
#     alphahat = xbar*(xbar*(1-xbar)/vbar - 1)
#     betahat = (1-xbar)*(xbar*(1-xbar)/vbar - 1)
#     return alphahat, betahat
#
# positive_beta_estimates = estimate_beta(predicted_y_probs[test_y == 1])
# negative_beta_estimates = estimate_beta(predicted_y_probs[test_y == 0])
#
# unit_interval = np.linspace(0,1,100)
# plt.plot(unit_interval, scipy.stats.beta.pdf(unit_interval, *positive_beta_estimates), c='g', label="positive")
# plt.plot(unit_interval, scipy.stats.beta.pdf(unit_interval, *negative_beta_estimates), c='r', label="negative")
#
# # Show the threshold.
# plt.axvline(sensitivity, c='black', ls='dashed')
# plt.xlim(0,1)
#
# # Add labels
# plt.legend()
#
# plt.savefig('p_tested_Sym.png')
