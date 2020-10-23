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
        self.beta = 0
        self.gamma = 0

    def RSS(self, parameters):






def covid_20():

    # Import datas
    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
    data = pd.read_csv(url, sep=',', header=0)
    print(data)
    data_matrix = data.to_numpy()


if __name__ == "__main__":

    covid_20()