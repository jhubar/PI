import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math

"""
=======================================================================================================

=======================================================================================================
"""


class SEIR():

    def __init__(self):

        # Model's hyperparameters
        self.beta = None        # Contagion probability
        self.sigma = None       # Probability to go from E to I
        self.gammaA = None      # Probability parameter to go from IA to R (to be cured)
        self.gammaB = None      # Probability to go to H from IB
        self.hp = None          # Hospit Probability: Proportion E * sigma who go to IB state (infectious and later hospitalized)
        self.hcr = 0            # Hospit Cure Rate

        # Data to fit
        self.dataset = None     # Numpy matrix format
        self.dataframe = None   # Dataframe format

        # Initial state: to be used to make predictions
        self.S_0 = None         # Sensible: peoples who can catch the agent
        self.E_0 = None         # Exposed: people in incubation: Can't spread the agent
        self.IA_0 = None        # The part of infectious people who will not going to fall in hospital
        self.IB_0 = None        # The part of infectious people who will fall in hospital
        self.H_0 = None         # Hospitalized peoples: can't spread more the agent
        self.R_0 = None         # Recovered people: can't catch again the agent due to immunity
        self.N = None           # The total size of the population

    def get_initial_state(self):
        """
        Function who return a tuple with the initial state of the model
        """
        return (self.S_0, self.E_0, self.IA_0, self.IB_0, self.H_0, self.R_0, self.N)

    def get_parameters(self):
        """
        Function who return a tuple with model's hyper-parameters
        """
        return (self.beta, self.sigma, self.gammaA, self.gammaB, self.hp, self.hcr)

    def differential(self, state, time, beta, sigma, gammaA, gammaB, hp, hcr):
        """
        Differential equations of the model
        """
        S, E, IA, IB, H, R, N = state

        dS = -(beta * S * (IA + IB)) / N
        dE = (beta * S * (IA + IB)) / N - E * sigma
        dIA = (1 - hp) * (E * sigma) - (gammaA * IA)
        dIB = hp * E * sigma - gammaB * IB
        dH = gammaB * IB - hcr * H
        dR = gammaA * IA + hcr * H
        dN = 0

        return dS, dE, dIA, dIB, dH, dR, dN

    def predict(self, duration):
        """
        Predict epidemic curves
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

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4], predict[:, 5])).T

    def fit(self):
        """
        Try to find our hyper-parameters values
        """
        # Generate initial state:
        initial_state = self.get_initial_state()
        # Time vector:
        time = self.dataset[:, 0]
        # Bounds: Given ranges
        bounds = [(0, 1), (1/5, 1), (1/10, 1/4), (1/10, 1/4), (0, 1), (0, 1)]
        # Start values
        start_values = [self.beta, self.sigma, self.gammaA, self.gammaB, self.hp, self.hcr]
        # Use Scipy.optimize.minimize with L-BFGS_B method
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, 'fit_on_cumul_positive'),
                       method='L-BFGS-B', bounds=bounds)

        print(res)
        self.beta = res.x[0]
        self.sigma = res.x[1]
        self.gammaA = res.x[2]
        self.gammaB = res.x[3]
        self.hp = res.x[4]


    def SSE(self, parameters, initial_state, time, method='fit_on_cumul_positive'):
        """
        Compute and return the sum of square errors input dataset and predictions of our model
        with the given super-parameters.
        Method parameter permit to chose the way to compare prediction's and dataset.
            - 'fit_on_cumul_positive' : compare the cumulative positive test column with the sum of
                our IA, IB, H and R curves
        """
        # Set parameters:
        tpl = tuple(parameters)
        params = (tpl[0], tpl[1], tpl[2], tpl[3], tpl[4], 0)
        # Make predictions:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=params)
        if method == 'fit_on_cumul_positive':
            sse = 0.0
            for i in range(0, len(time)):
                sse += (self.dataset[i][7] - predict[i][2] - predict[i][3] - predict[i][4] - predict[i][5]) ** 2
            return sse

        if method == 'fit_on_cumul_hospit':
            # Note: Only possible if hcr parameter set on zero
            sse = 0.0
            for i in range(0, len(time)):
                sse += (self.dataset[i][4] - predict[i][4])
            return sse


    def import_dataset(self, target='covid_20'):

        if target == 'covid_20':
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # Import the dataframe:
            self.dataframe = pd.read_csv(url, sep=',', header=0)

            # Modify the first line to take account of unobserved early infections
            self.dataframe['num_positive'][0] = 20  # Approx 20 * hospitalizations
            self.dataframe['num_tested'][0] = 20

            # Ad a new column at the end with cumulative positive cases at the right
            cumul_positive = self.dataframe['num_positive'].to_numpy()
            for i in range(1, len(cumul_positive)):
                cumul_positive[i] += cumul_positive[i-1]
            self.dataframe.insert(7, 'cumul_positive', cumul_positive)
            # Store a numpy version:
            self.dataset = self.dataframe.to_numpy()

            # Store the initial state who fit with input data
            self.N = 1000000
            self.IB_0 = 3 * self.dataframe['num_hospitalised'][0]
            self.IA_0 = self.dataframe['cumul_positive'][0] - self.IB_0
            self.H_0 = self.dataframe['num_hospitalised'][0]
            self.E_0 = 4 * self.dataframe['num_positive'][1]
            self.R_0 = 0
            self.S_0 = self.N - self.IB_0 - self.IA_0 - self.H_0 - self.E_0

            # Initialize default value to hyper-parameters:
            self.beta = 0.35
            self.sigma = 1/3
            self.gammaA = 1/7
            self.gammaB = 1/7
            self.hp = 1/20.89
            self.hcr = 0





    def plot_predict(self, pred, args='no_S'):

        if "no_S" not in args:
            plt.plot(pred[:, 0], pred[:, 1], c='black', label="S")
        plt.plot(pred[:, 0], pred[:, 2], c='yellow', label="E")
        plt.plot(pred[:, 0], pred[:, 3] + pred[:, 4], c='red', label="I")
        plt.plot(pred[:, 0], pred[:, 5], c='purple', label="H")
        plt.plot(pred[:, 0], pred[:, 6], c='blue', label='R')
        plt.show()

        if 'compare' in args:
            plt.scatter(self.dataframe['Day'], self.dataframe['cumul_positive'], c='red')
            plt.scatter(self.dataframe['Day'], self.dataframe['num_hospitalised'], c='blue')

            cumul_positive = []
            hospit = []
            for i in range(0, len(self.dataframe['Day'].to_numpy())):
                cumul_positive.append(pred[i][3] + pred[i][4] + pred[i][5] + pred[i][6])  # somme de I, H et R
                hospit.append(pred[i][5])
            plt.plot(self.dataframe['Day'], cumul_positive, c='red')
            plt.plot(self.dataframe['Day'], hospit, c='blue')
            if "log" in args:
                plt.yscale('log')
            plt.show()


def first_method():

    # Initialize the model
    model = SEIR()

    # Import the dataset:
    model.import_dataset(target='covid_20')

    # Fit the model:
    model.fit()

    # Make predictions:
    predictions = model.predict(50)
    model.plot_predict(predictions, args='compare')
    model.plot_predict(predictions, args='compare log')








if __name__ == "__main__":

    first_method()



