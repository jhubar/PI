
import matplotlib.pyplot as plt
import scipy.stats
import scipy.stats as stats
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
from scipy import signal

def add_uncertainty(self, df):

    # Convert the dataframe to a numpy array:
    np_df = df.to_numpy()
    np_df_upper = np.copy(np_df)
    np_df_lower = np.copy(np_df)
    sensitivity_upper_bound = 0.7
    sensitivity_lower_bound = 0.85

    day = np_df[:,0]

    num_positive = np_df[:,1]
    num_positive_lower = np.array(np_df[:,1]+np_df[:,1]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))
    num_positive_upper = np.array(np_df[:,1]+np_df[:,1]*((1-sensitivity_upper_bound)/sensitivity_upper_bound))
    num_positive_mean = (num_positive_upper+num_positive_lower)/2


    norm_num_positive = np.mean(num_positive)/np.mean(num_positive)
    norm_num_positive_lower = np.mean(num_positive)/np.mean(num_positive_lower)
    norm_num_positive_upper = np.mean(num_positive)/np.mean(num_positive_upper)
    norm_num_positive_mean = np.mean(num_positive)/np.mean(num_positive_mean)

    std_num_postive = np.mean((norm_num_positive_lower-norm_num_positive_mean)/3)
    x_1 = np.linspace(norm_num_positive_mean - 3*std_num_postive, norm_num_positive_mean + 3*std_num_postive, 100)
    plt.plot(x_1, stats.norm.pdf(x_1, norm_num_positive_mean, std_num_postive),label ='positive')



    num_tested = np_df[:,2]
    num_tested_upper = 2* num_tested
    num_tested_mean = (3/2)* num_tested

    norm_num_tested =np.mean(num_tested)/np.mean(num_tested)
    norm_num_tested_upper =np.mean(num_tested)/np.mean(num_tested_upper)
    norm_num_tested_mean = np.mean(num_tested)/np.mean(num_tested_mean)

    std_num_tested = np.mean((norm_num_tested-norm_num_tested_mean)/3)
    x_2 = np.linspace(norm_num_tested_mean - 3*std_num_tested, norm_num_tested_mean + 3*std_num_tested, 100)
    plt.plot(x_2, stats.norm.pdf(x_2, norm_num_tested_mean, std_num_tested), label = 'tested')


    num_hospitalised =  np_df[:,3]
    num_cumulative_hospitalizations = np_df[:,4]
    num_critical = np_df[:,5]
    num_fatalities = np_df[:,6]

    # number infected
    num_sym_lower = num_positive_lower
    num_sym_lower = own_NRMAS(num_sym_lower,self.window)


    num_sym_upper = np_df[:,2]+num_positive_upper
    num_positive_upper = own_NRMAS(num_positive_upper,self.window)

    num_positive_mean = (num_positive_lower+num_positive_upper)/2

    num_positive_lower = own_NRMAS(num_positive_lower,self.window)
    num_sym_upper = own_NRMAS(num_sym_upper,self.window)

    num_sym_mean = (num_sym_lower+num_sym_upper)/2

    norm_sym_mean =  num_positive/num_sym_mean
    norm_sym_lower = num_positive/num_sym_lower
    norm_sym_upper = num_positive/num_sym_upper

    std_vec = (norm_sym_lower-norm_sym_mean)/3

    std = np.mean(std_vec)
    mu = np.mean(norm_sym_mean)

    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    plt.plot(x, stats.norm.pdf(x, mu, std),label ='normal de hier')
    plt.legend()
    plt.savefig('Plot/normal_dis.png')



    new_df = np.vstack((day,num_positive,num_positive_lower,num_positive_upper,num_tested
        ,num_hospitalised,num_cumulative_hospitalizations,num_critical,num_fatalities
        ,num_sym_lower,num_sym_upper,num_positive_mean,num_sym_mean))

    return pd.DataFrame(new_df.T, columns=['day'                             #0
                                          ,'num_positive'                    #1
                                          ,'num_positive_lower'              #2
                                          ,'num_positive_upper'              #3
                                          ,'num_tested'                      #4
                                          ,'num_hospitalised'                #5
                                          ,'num_cumulative_hospitalizations' #6
                                          ,'num_critical'                    #7
                                          ,'num_fatalities'                  #8
                                          ,'num_sym_lower'                   #9
                                          ,'num_sym_upper'                   #10
                                          ,'num_positive_mean'               #11
                                          ,'num_sym_mean'])                  #12