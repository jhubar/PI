import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import requests # to dowload csv file in github
from numpy import asarray
from numpy import savetxt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.integrate import odeint
from scipy.optimize import minimize


class SIR_model():

    def __init__(self):
        """
        Init the model
        """
        self.beta = 0.5717
        self.gamma = 0.4282
        self.population = 1000000
        self.S_0 = 999999
        self.I_0 = 1
        self.R_0 = 0
        self.sir_start_date = "2020-02-02"
        self.sir_end_date = "2020-02-18"
        self.dataset = np.ones(1)

    def predictor(self, SIR_values, time):
        dS = -self.beta * SIR_values[0] * SIR_values[1] / self.population
        dI = (self.beta * SIR_values[0] * SIR_values[1] / self.population) - (self.gamma * SIR_values[1])
        dR = self.gamma * SIR_values[1]
        return dS, dI, dR

    def manual_predictor(self, SIR_values, time, beta, gamma):
        dS = -beta * SIR_values[0] * SIR_values[1] / self.population
        dI = (beta * SIR_values[0] * SIR_values[1] / self.population) - (gamma * SIR_values[1])
        dR = gamma * SIR_values[1]
        return dS, dI, dR
    def RSS(self, parameters):

        # Vector of initial values:
        SIR_init = [self.S_0, self.I_0, self.R_0]
        # time vector:
        t = np.zeros(self.dataset.shape[0])
        for i in range(0, len(t)):
            t[i] = i
        predict = odeint(self.manual_predictor, SIR_init, t, args=parameters)

        # Compute SSE
        SSE = (self.dataset[:, 1] - predict[:, 1] - predict[:, 2])**2
        return SSE






def covid_20():

    # Import datas
    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
    data = pd.read_csv(url, sep=',', header=0)
    print(data)
    dataset = data.to_numpy()
    # make cumul data:
    for i in range(1, dataset.shape[0]):
        dataset[i][1] += dataset[i-1][1]

    model = SIR_model()
    model.dataset = dataset
    model.RSS((0.5717, 0.42))








if __name__ == "__main__":

    covid_20()
