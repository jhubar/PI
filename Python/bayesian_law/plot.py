
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


def cumul_positif_comp(self):
    plt.scatter(predictions[:, 0], self.raw_dataset['cumul_positive'], c='blue', label='Original data')
    plt.plot(predictions[:, 0], cumul_positive, c='red', label='Predictions')
    plt.title('Comparison between cumulative of positive test and I + R predictions')
    plt.xlabel('Time in days')
    plt.ylabel('Number of peoples')
    plt.legend()
    plt.savefig('Plot/cumul_positif_comp.png', transparent=True)
    plt.close()

def cumul_hospit_comp(self):
    plt.scatter(predictions[:, 0], self.dataset[:, 4], c='blue', label='Original data')
    plt.plot(predictions[:, 0], cumul_hospit, c='red', label='Predictions')
    plt.title('Comparison between cumulative hospitalisation data and predictions')
    plt.xlabel('Time in days')
    plt.ylabel('Number of peoples')
    plt.legend()
    plt.savefig('Plot/cumul_hospit_comp.png', transparent=True)
    plt.close()

def non_cum_hospit_comp(self):
    plt.scatter(predictions[:, 0], self.dataset[:, 3], c='blue', label='Original data')
    plt.plot(predictions[:, 0], hospit, c='red', label='Predictions')
    plt.title('Comparison between non-cumulative hospitalisation data and predictions')
    plt.xlabel('Time in days')
    plt.legend()
    plt.ylabel('Number of peoples')
    plt.savefig('Plot/non_cum_hospit_comp.png', transparent=True)
    plt.close()

def critical_com(self):
    plt.scatter(predictions[:, 0], self.dataset[:, 5], c='blue', label='Original data')
    plt.plot(predictions[:, 0], critical, c='red', label='Predictions')
    plt.title('Comparison ICU data and critical predictions')
    plt.legend()
    plt.xlabel('Time in days')
    plt.ylabel('Number of peoples')
    plt.savefig('Plot/critical_com.png', transparent=True)
    plt.close()

def fatal_com(self):
    plt.scatter(predictions[:, 0], self.dataset[:, 6], c='blue', label='Original data')
    plt.plot(predictions[:, 0], fatalities, c='red', label='Predictions')
    plt.title('Comparison fatalities cumulative data and D curve')
    plt.legend()
    plt.xlabel('Time in days')
    plt.ylabel('Number of peoples')
    plt.savefig('Plot/fatal_com.png', transparent=True)
    plt.close()

def preporcessing(self):
    # Plot difference between original data and smoothed data on positive values:
    y_raw = self.raw_dataset['num_positive'].to_numpy()
    y_smooth = self.dataframe['num_positive'].to_numpy()
    x_axe = np.arange(len(y_raw))
    plt.scatter(x_axe, y_raw, c='blue', label='original positives')
    plt.plot(x_axe, y_smooth, c='red', label="smoothed positives")
    # The same for hospitalisation data:
    y_raw = self.raw_dataset['num_hospitalised']
    y_smooth = self.dataframe['num_hospitalised']
    plt.scatter(x_axe, y_raw, c='green', label='original hospitalised')
    plt.plot(x_axe, y_smooth, c='orange', label="smoothed hospitalised")
    plt.title("Data Pre-processing")
    plt.xlabel("Time in days")
    plt.ylabel("Number of people")
    plt.legend()
    plt.savefig("Plot/preprocessing.png", transparent=True)
    # plt.show()
    plt.close()
