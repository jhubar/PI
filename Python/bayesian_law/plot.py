
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

def plot_current_data(self):

    fig = plt.figure(figsize=(25,20))

    ax = plt.subplot()
    ax.plot(self.dataframe['day'], self.dataframe['num_positive_lower'], label='lower')
    ax.plot(self.dataframe['day'], self.dataframe['num_positive_mean'], label='mean')
    ax.plot(self.dataframe['day'], self.dataframe['num_positive'], label='current data')
    ax.plot(self.dataframe['day'], self.dataframe['num_positive_upper'], label='upper')
    ax.fill_between(self.dataframe['day'], self.dataframe['num_positive_lower'], self.dataframe['num_positive_upper'])
    pr_mean = self.dataframe['num_sym_lower']
    next_pr_mean = (2*pr_mean[len(pr_mean)-1]-pr_mean[len(pr_mean)-2])
    next_nb_tested = next_pr_mean*self.ran
    print(next_nb_tested)
    ax.legend(fontsize=30)
    fig.savefig('Plot/current_data_with_uncertainty.png')
