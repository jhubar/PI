import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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

    def curves(self):


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
        args = (parameters[0], parameters[1])
        predict = odeint(self.manual_predictor, SIR_init, t, args=args)

        # Compute SSE
        SSE = (self.dataset[:, 1] - predict[:, 1] - predict[:, 2])**2
        return np.sum(SSE)

    def fit(self):

        start_values = np.array([0.5, 0.5])

        #ret_val = minimize(self.RSS, method="L-BFGS-B", x0=start_values, bounds=((0.01, 0.09), (0.01, 0.09)))
        ret_val = minimize(self.RSS, x0=start_values, method="L-BFGS-B", bounds=((0.01, 0.9), (0.01, 0.9)))
        print(ret_val)

    def fit_manual(self, dataset, beta_min=0, beta_max=1, gamma_min=0, gamma_max=1, range_size=100):
        """
        Find optimal value of beta and gamma parameters for the given dataset
        """
        beta_interval = (beta_max - beta_min) / range_size
        gamma_interval = (gamma_max - gamma_min) / range_size
        beta_range = [beta_interval + beta_min]
        gamma_range = [gamma_interval + gamma_min]
        for i in range(1, range_size):
            beta_range.append(beta_range[i - 1] + beta_interval)
            gamma_range.append(gamma_range[i - 1] + gamma_interval)

        SSE = np.zeros((range_size, range_size))
        best_val = (999999999999, 0, 0)
        for b in range(0, range_size):
            for g in range(0, range_size):
                parameters = (beta_range[b], gamma_range[g])
                SSE[b][g] = self.RSS(parameters)
                if SSE[b][g] < best_val[0]:
                    best_val = (SSE[b][g], beta_range[b], beta_range[g])

        print(best_val)










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
    model.fit_manual(dataset, beta_min=0.01, beta_max=0.9, gamma_min=0.1, gamma_max=0.8, range_size=200 )








if __name__ == "__main__":

    covid_20()
