import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from scipy.integrate import odeint   # To integrate our equation
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import uncertainpy as un
import chaospy as cp                 # To create distributions
import json
import math
import random
from smooth import own_NRMAS_index, own_NRMAS
from plot import plot_current_data
from plot import preporcessing
from uncertainty import add_uncertainty


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
        # self.ran = random.uniform(0.5, 1)
        self.ran = np.random.normal(0.75, (0.25**(1/2)))

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
        self.window = 7
        self.dataJSON = {}

    def saveJson(self):
        with open('Data/SEIR+.json', 'w') as outfile:
            json.dump(self.dataJSON, outfile)

    def get_initial_state(self):
        """
        Function who return a tuple with the initial state of the model
        """
        return (self.S_0, self.E_0, self.I_0, self.H_0, self.R_0, self.N, self.C_0, self.D_0)

    def get_parameters(self):
        """
        Function who return a tuple with model's hyper-parameters
        """
        return (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr)

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr):
        """
        Differential equations of the model
        """
        S, E, I, H, R, N, C, D = state

        dS = -(beta * S * I) / N
        dE = (beta * S * I) / N - E * sigma
        dI = (E * sigma) - (gamma * I) - (hp * I)
        dH = (hp * I) - (hcr * H) - (pc * H)
        dC = (pc * H) - (pd * C) - (pcr * C)
        dR = (gamma * I) + (hcr * H)
        dD = (pd * C)
        dN = 0

        return dS, dE, dI, dH, dR, dN, dC, dD

    def import_dataset(self, target='covid_20'):

        if target == 'covid_20':
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # Import the dataframe:
            self.raw_dataset = pd.read_csv(url, sep=',', header=0)

            self.dataframe = add_uncertainty(self,self.raw_dataset)
            # Need to add smoth in data
            plot_current_data(self)

            # Ad a new column at the end with cumulative positive cases at the right
            cumul_positive = self.dataframe['num_positive'].to_numpy()
            cumul_positive_non_smooth = self.raw_dataset['num_positive']
            for i in range(1, len(cumul_positive)):
                cumul_positive[i] += cumul_positive[i - 1]
                cumul_positive_non_smooth[i] += cumul_positive_non_smooth[i - 1]
            self.dataframe.insert(12, 'cumul_positive', cumul_positive)
            # self.raw_dataset.insert(12, 'cumul_positive', cumul_positive_non_smooth)

            # Delete the first line with zero test
            for i in range(0, 1):
                self.raw_dataset.drop(axis=0, index=i, inplace=True)
                self.dataframe.drop(axis=0, index=i, inplace=True)
            # To reset dataframe index:
            tmp = self.raw_dataset.to_numpy()
            self.raw_dataset = pd.DataFrame(tmp, columns=self.raw_dataset.columns)
            tmp = self.dataframe.to_numpy()
            self.dataframe = pd.DataFrame(tmp, columns=self.dataframe.columns)

            # Store a numpy version:
            self.dataset = self.dataframe.to_numpy()

            # Store the initial state who fit with input data
            self.N = 1000000
            self.I_0 = self.dataset[0][7]
            self.H_0 = self.dataset[0][3]
            self.E_0 = 3 * self.dataset[1][1]  # Because mean of incubation period = 3 days
            self.R_0 = 0
            self.C_0 = 0
            self.D_0 = 0
            self.S_0 = self.N - self.I_0 - self.H_0 - self.E_0

            # Initialize default value to hyper-parameters:
            self.beta = 0.35
            self.sigma = 1 / 3
            self.gamma = 1 / 7
            self.hp = 0
            self.hcr = 0
