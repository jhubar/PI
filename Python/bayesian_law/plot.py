
import matplotlib.pyplot as plt
import scipy.stats
import scipy.stats as stats
from scipy.stats import norm
from scipy.integrate import odeint   # To integrate our equation
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import uncertainpy as un
import chaospy as cp
import seaborn as sns
import json
import math
import random

def plot_corr(corr):
    fig = plt.figure(figsize=(25,20))
    ax = plt.subplot()
    ax = sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot=True)
    plt.savefig('Plot/corr.png')
    plt.close()

def plot_normal(mu,std,lab = 'Nan', arg = 'continue'):
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    plt.plot(x, stats.norm.pdf(x, mu, std), label = lab)
    plt.legend()
    if arg =='stop':
        plt.savefig('Plot/normal_dis.png')
        plt.close()


def __plot_predict__(self, pred, args='no_S'):

    self.dataJSON['predict'] = []
    for i in range(0, len(pred[:, 0])):
        self.dataJSON['predict'].append({
            "predict_day": str(pred[i][0]),
            "predict_S": str(pred[i][1]),
            "predict_E": str(pred[i][2]),
            "predict_I": str(pred[i][3]),
            "predict_H": str(pred[i][4]),
            "predict_R": str(pred[i][5]),
            "predict_C": str(pred[i][7]),
            "predict_F": str(pred[i][8]),

        })

    self.dataJSON['model'] = []
    self.dataJSON['model'].append({
        "beta": str(self.beta),
        "sigma": str(self.sigma),
        "gamma": str(self.gamma),
        "hp": str(self.hp),
        "hcr": str(self.hcr),
    })


    if "no_S" not in args:
        plt.plot(pred[:, 0], pred[:, 1], c='green', label="S")
    plt.plot(pred[:, 0], pred[:, 2], c='yellow', label="E")
    plt.plot(pred[:, 0], pred[:, 3], c='red', label="I")
    plt.plot(pred[:, 0], pred[:, 4], c='purple', label="H")
    plt.plot(pred[:, 0], pred[:, 5], c='blue', label='R')
    plt.plot(pred[:, 0], pred[:, 7], c='orange', label='C')
    plt.plot(pred[:, 0], pred[:, 8], c='black', label='D')
    plt.xlabel("Time (Days)")
    plt.ylabel("Number of peoples")
    plt.legend()
    plt.title("Evolution of epidemic curves")
    if "no_S" not in args:

        plt.savefig("Plot/long_time_predictions_no_s.png", transparent=True)
    else:
        plt.savefig("Plot/long_time_predictions.png", transparent=True)
    # plt.show()
    plt.close()

    if 'compare' in args:
        plt.scatter(self.dataframe['Day'], self.dataframe['cumul_positive'], c='red')
        if self.pc == 0 and self.hcr == 0:
            plt.scatter(self.dataframe['Day'], self.dataframe['num_cumulative_hospitalizations'], c='blue',
                        label='cumul_hosp')
        else:
            plt.scatter(self.dataframe['Day'], self.dataframe['num_hospitalised'], c='blue', label='hosp')

        cumul_positive = []
        hospit = []
        for i in range(0, len(self.dataframe['Day'].to_numpy())):
            cumul_positive.append(
                pred[i][3] + pred[i][4] + pred[i][5] + pred[i][7] + pred[i][8])  # sum of I, H, R, C and D
            hospit.append(pred[i][4])

        print(len(self.dataframe['Day']))
        self.dataJSON['log'] = []
        for i in range(0, len(self.dataframe['Day'])):
            self.dataJSON['log'].append({
                "day": str(self.dataframe['Day'][i]),
                "cumul_positive": str(self.dataframe['cumul_positive'][i]),
                "hospit": str(self.dataframe['num_hospitalised'][i]),
                "cumul_positive_fit": str(cumul_positive[i]),
                "hospit_fit": str(hospit[i]),
            })
        plt.plot(self.dataframe['Day'], cumul_positive, c='red')
        plt.plot(self.dataframe['Day'], hospit, c='blue')
        if "log" in args:
            plt.yscale('log')
        plt.show()

    if 'hospit' in args:
        hospit = []
        for i in range(0, len(self.dataframe['Day'].to_numpy())):
            hospit.append(pred[i][4])
        plt.scatter(self.dataframe['Day'], self.dataframe['num_hospitalised'], c='green')
        plt.plot(self.dataframe['Day'], hospit, c='red')
        plt.show()

def plot_current_data(self):

    fig = plt.figure(figsize=(25,20))

    ax = plt.subplot()
    ax.plot(self.dataframe['day'], self.dataframe['num_sym_lower'],color='blue', label='num symp lower',alpha=0.5)
    ax.plot(self.dataframe['day'], self.dataframe['num_sym_mean'],color='blue', label='num symp mean')
    ax.plot(self.dataframe['day'], self.dataframe['num_sym_upper'],color='blue', label='num symp upper',alpha=0.5)
    ax.fill_between(self.dataframe['day'], self.dataframe['num_sym_lower'], self.dataframe['num_sym_upper'],alpha=0.5)
    ax.plot(self.dataframe['day'], self.dataframe['num_positive_lower'], color='orange',label='num positive lower',alpha=0.5)
    ax.plot(self.dataframe['day'], self.dataframe['num_positive_mean'], color='orange', label='num positive mean')
    ax.plot(self.dataframe['day'], self.dataframe['num_positive_upper'], color='orange', label='num positive upper',alpha=0.5)
    ax.plot(self.dataframe['day'], self.dataframe['num_positive'],color='black', label='current data')
    ax.fill_between(self.dataframe['day'], self.dataframe['num_positive_lower'], self.dataframe['num_positive_upper'],alpha=0.5)
    pr_mean = self.dataframe['num_sym_lower']
    next_pr_mean = (2*pr_mean[len(pr_mean)-1]-pr_mean[len(pr_mean)-2])
    next_nb_tested = next_pr_mean*self.ran
    plt.xlabel('Time in days',fontsize=30)
    plt.ylabel('Number of peoples',fontsize=30)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.legend(fontsize=30)
    fig.savefig('Plot/current_data_with_uncertainty.png')
    plt.close()


def plot_cumul_positif_comp(self,cumul_positive,predictions):
    plt.scatter(predictions[:, 0], self.raw_dataset['cumul_positive'], c='blue', label='Original data')
    plt.plot(predictions[:, 0], cumul_positive, c='red', label='Predictions')
    plt.title('Comparison between cumulative of positive test and I + R predictions')
    plt.xlabel('Time in days')
    plt.ylabel('Number of peoples')
    plt.legend()
    plt.savefig('Plot/cumul_positif_comp.png', transparent=True)
    plt.close()

def plot_cumul_hospit_comp(self,cumul_hospit,predictions):
    plt.scatter(predictions[:, 0], self.dataset[:, 4], c='blue', label='Original data')
    plt.plot(predictions[:, 0], cumul_hospit, c='red', label='Predictions')
    plt.title('Comparison between cumulative hospitalisation data and predictions')
    plt.xlabel('Time in days')
    plt.ylabel('Number of peoples')
    plt.legend()
    plt.savefig('Plot/cumul_hospit_comp.png', transparent=True)
    plt.close()

def plot_non_cum_hospit_comp(self,hospit,predictions):
    plt.scatter(predictions[:, 0], self.dataset[:, 3], c='blue', label='Original data')
    plt.plot(predictions[:, 0], hospit, c='red', label='Predictions')
    plt.title('Comparison between non-cumulative hospitalisation data and predictions')
    plt.xlabel('Time in days')
    plt.legend()
    plt.ylabel('Number of peoples')
    plt.savefig('Plot/non_cum_hospit_comp.png', transparent=True)
    plt.close()

def plot_critical_com(self,critical,predictions):
    plt.scatter(predictions[:, 0], self.dataset[:, 5], c='blue', label='Original data')
    plt.plot(predictions[:, 0], critical, c='red', label='Predictions')
    plt.title('Comparison ICU data and critical predictions')
    plt.legend()
    plt.xlabel('Time in days')
    plt.ylabel('Number of peoples')
    plt.savefig('Plot/critical_com.png', transparent=True)
    plt.close()

def plot_fatal_com(self,fatalities,predictions):
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

def plot_pcr_pd_slide(self,proportion_range,SSE):
    # plot :
    plt.plot(proportion_range, np.flip(SSE), c='blue')
    plt.title('Proportion of pcr_A assigned to pd')
    plt.legend()
    plt.yscale('log')
    plt.ylabel('log sum of square error')
    plt.xlabel('pcr_A proportion')
    plt.savefig("Plot/pcr_pd_slide.png", transparent=True)
    # plt.show()
    plt.close()

def plot_hcr_fitting(self,hcr_range,SSE):
    plt.plot(hcr_range, SSE, c='blue', label='hcr value')
    plt.title('Evolution of the sum of square error according to the value of hcr')
    plt.legend()
    plt.yscale('log')
    plt.ylabel('log sum of square error')
    plt.xlabel('hcr value')
    plt.savefig("Plot/hcr_fitting.png", transparent=True)
    # plt.show()
    plt.close()

def plot_gamma_hp_slide(self,proportion_range, SSE):
    plt.plot(proportion_range, np.flip(SSE), c='blue', label='Gamma value')
    plt.title('Proportion of Gamma-A assigned to hp')
    plt.yscale('log')
    plt.legend()
    plt.ylabel('log sum of square error')
    plt.xlabel('gamma proportion')
    plt.savefig("Plot/gamma_hp_slide.png", transparent=True)
    # plt.show()
    plt.close()
