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
import uncertainpy as un
import chaospy as cp                 # To create distributions
from scipy.integrate import odeint

import math
import random

class SEIR_Uncertainty():

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
        self.ran = random.uniform(0.5, 1)

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

    # def get_parameters(self):
    #     """
    #     Function who return a tuple with model's hyper-parameters
    #     """
    #     self.beta=
    #     self.sigma=
    #     self.gamma
    #     self.hp=
    #     self.hcr=
    #     self.pc=
    #     self.pd=
    #     self.pcr=
    #
    #     return (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr)
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
        num_hospitalised =  np_df[:,3]
        num_cumulative_hospitalizations = np_df[:,4]
        num_critical = np_df[:,5]
        num_fatalities = np_df[:,6]
        num_sym_lower = np_df[:,2]

        num_sym_upper = np_df[:,2]+(np_df[:,2]*self.ran)


        new_df = np.vstack((day,num_positive_lower,num_positive_upper,num_tested
            ,num_hospitalised,num_cumulative_hospitalizations,num_critical,num_fatalities
            ,num_sym_lower,num_sym_upper))




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

            self.dataframe = pd.DataFrame(self.dataframe.T, columns=['day', 'num_positive_lower', 'num_positive_upper',
                                                                   'num_tested', 'num_hospitalized',
                                                                   'num_cumulative_hospitalizations', 'num_critical',
                                                                   'num_fatalities', 'num_sym_lower', 'num_sym_upper'])

            plt.plot(self.dataframe)
            pr_mean = self.dataframe['num_sym_lower']
            next_pr_mean = (2*pr_mean[len(pr_mean)-1]-pr_mean[len(pr_mean)-2])
            next_nb_tested = next_pr_mean*self.ran
            print(next_nb_tested)
            plt.legend()
            plt.savefig('p_Sym.png')

    def predict(self, duration):
        """
        Predict epidemic curves from t_0 for the given duration
        """
        # Initialisation vector:
        initial_state = self.get_initial_state()
        # Time vector:
        time = np.arange(duration)
        # Parameters vector
        parameters = self.get_parameters()
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=parameters)

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4],
                          predict[:, 5], predict[:, 6], predict[:, 7])).T


def coffee_cup(kappa, T_env):
    # Initial temperature and time array
    time = np.linspace(0, 200, 150)            # Minutes
    T_0 = 95
    N = 1000000                                 #

    # The equation describing the model
    def f(T, time, kappa, T_env):
        ds = -kappa*(T - T_env)
        return ds




    # Solving the equation by integration.
    temperature = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

    # Return time and model output
    return time, temperature


def first_method():

    # Initialize the model
    data_with_uncertainty = SEIR_Uncertainty()

    # Import the dataset:
    data_with_uncertainty.import_dataset(target='covid_20')


    # plt.show()
    plt.close()

    # Create a model from the coffee_cup function and add labels
    model = un.Model(run=coffee_cup, labels=["Time (min)", "Temperature (C)"])

    # Create the distributions
    kappa_dist = cp.Uniform(0.025, 0.075)
    T_env_dist = cp.Uniform(15, 25)

    # Define the parameter dictionary
    parameters = {"kappa": kappa_dist, "T_env": T_env_dist}

    # Set up the uncertainty quantification
    UQ = un.UncertaintyQuantification(model=model,
                                      parameters=parameters)

    # Perform the uncertainty quantification using
    # polynomial chaos with point collocation (by default)
    data = UQ.quantify()




if __name__ == "__main__":
    first_method()
