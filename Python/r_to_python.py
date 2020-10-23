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


class SIR_model():

    def __init__(self):
        """
        Init the model
        """
        self.beta = 0.5
        self.gamma = 0.5
        self.population = 1000000
        self.S_0
        self.I_0
        self.R_0
        self.sir_start_date = "2020-02-02"
        self.sir_end_date = "2020-02-02

    def predictor(self, SIR_values):

        dS = -self.beta * SIR_values[0] * SIR_values[1] / self.population
        dI = (self.beta * SIR_values[0] * SIR_values[1] / self.population) - (self.gamma * SIR_values[1])
        dR = self.gamma * SIR_values[1]
        return dS, dI, dR

    def deriv(self, ):

    def RSS(self, parameters):

        # Compute range to test
        beta_interval = (beta_max - beta_min) / range_size
        gamma_interval = (gamma_max - gamma_min) / range_size
        beta_range = [beta_interval + beta_min]
        gamma_range = [gamma_interval + gamma_min]
        for i in range(1, range_size):
            beta_range.append(beta_range[i-1] + beta_interval)
            gamma_range.append(gamma_range[i-1] + gamma_interval)

        # Vector of initial values:
        SIR_init = [self.S_0, self.I_0, self.R_0]
        predict = odeint(predictor, SIR_init, 17)



    def optim(self,RSS, [1,2], upper):



def covid_20():

    # Import datas
    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
    data = pd.read_csv(url, sep=',', header=0)
    print(data)
    data_matrix = data.to_numpy()


def plot():

    title = "COVID-20 fitted vs observed cumulative incidence, Belgium"
    subtitle = "(Red = fitted from SIR model, blue = observed)"


    plt.figure()
    plt.title(title)
    plt.subtitle(subtitle)
    plt.xlabal("Date")
    plt.ylabal("")




if __name__ == "__main__":

    covid_20()
