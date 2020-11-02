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
        # Parameter's values
        self.N = 999999
        self.beta = 0.3865
        self.gamma = 1 / 7  # 7 days average to be cure
        self.sigma = 1 / 3  # 3 days average incubation time
        self.hp = 1 / 20.89  # Hospit probability Proportion of infected people who begin hospitalized
        self.hcr = 0  # Hospit Cure Rate: proba de guérir en hospitalisation

    def differential(self, state, time, beta, gamma, sigma, hp, hcr):
        """
        Differential equations of the model
        """
        S, E, I, H, R = state

        dS = -(beta * S * I) / (S + E + I + H + R)
        dE = (beta * S * I) / (S + E + I + H + R) - E * sigma
        dI = (E * sigma) - (gamma * I) - (I * hp * gamma)
        dH = (I * hp * gamma) - hcr * H
        dR = gamma * I + hcr * H

        return dS, dE, dI, dH, dR

    def predict(self, S_0, E_0, I_0, H_0, R_0, duration):
        """
        Predict epidemic curves
        """
        # Initialisation vector:
        initial_state = (S_0, E_0, I_0, H_0, R_0)
        # Time vector:
        time = np.arange(duration)
        # Parameters vector
        parameters = (self.beta, self.gamma, self.sigma, self.hp, self.hcr)
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=parameters)

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4])).T


    def SSE(self, parameters, initial_state, time, data, method='fit_on_hospit'):

        # Set parameters:
        params = tuple(parameters)
        # Make predictions:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=params)

        if method == 'fit_on_hospit':
            # On fit alors que sur la courbe de hospitalisations
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][3] - predict[i][3]) ** 2
            return sse
        if method == 'fit_on_hospit_cumul':
            # si hcr est = à 0, on peut fiter la courbe des hospit avec cumul hopit car pas de guérison
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][4] - predict[i][3]) ** 2
            return sse
        if method == 'fit_on_cumul_positive':

            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][7] - predict[i][2] - predict[i][3] - predict[i][4]) ** 2
            return sse
        if method == 'fit_on_cumul_mixt':
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][7] - predict[i][2] - predict[i][3] - predict[i][4]) ** 2
                sse += (data[i][4] - predict[i][3]) ** 2
            return sse

def dataframe_builder():

    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
    df = pd.read_csv(url, sep=",", header=0)
    # Delete the first line:
    df = df.drop([0], axis=0)
    # Insert cumul_positive column at the end
    cumul_positive = df["num_positive"].to_numpy()
    for i in range(1, len(cumul_positive)):
        cumul_positive[i] += cumul_positive[i - 1]
    df.insert(7, "cumul_positive", cumul_positive)
    # Make a numpy version:
    df_np = df.to_numpy()

def first_method():

    df_np = dataframe_builder()

    # Init the model:
    model = SEIR()
    # Set initial state
    H_0 = df_np[0][7]
    E_0 = 3 * df_np[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
    I_0 = df_np[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
    S_0 = 999999 - H_0 - I_0 - E_0
    R_0 = 0




if __name__ == "__main__":
    # first_method()
    first_method()


