
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
from smooth import own_NRMAS_index, own_NRMAS
from plot import plot_current_data

def add_uncertainty(self, df):

    # Convert the dataframe to a numpy array:
    np_df = df.to_numpy()
    np_df_upper = np.copy(np_df)
    np_df_lower = np.copy(np_df)
    sensitivity_upper_bound = 0.7
    sensitivity_lower_bound = 0.85

    day = np_df[:,0]
    num_positive =np_df[:,1]
    num_positive_lower = np.array(np_df[:,1]+np_df[:,1]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))
    num_positive_lower = own_NRMAS(num_positive_lower,self.window)
    num_positive_upper = np.array(np_df[:,1]+np_df[:,1]*((1-sensitivity_upper_bound)/sensitivity_upper_bound))
    num_positive_upper = own_NRMAS(num_positive_upper,self.window)
    num_positive_mean = (num_positive_lower+num_positive_upper)/2
    num_tested = np_df[:,2]
    num_hospitalised =  np_df[:,3]
    num_cumulative_hospitalizations = np_df[:,4]
    num_critical = np_df[:,5]
    num_fatalities = np_df[:,6]
    num_sym_lower = np_df[:,2]

    num_sym_upper = np_df[:,2]+(np_df[:,2]*self.ran)


    new_df = np.vstack((day,num_positive,num_positive_lower,num_positive_upper,num_tested
        ,num_hospitalised,num_cumulative_hospitalizations,num_critical,num_fatalities
        ,num_sym_lower,num_sym_upper,num_positive_mean))

    return pd.DataFrame(new_df.T, columns=['day'                             #0
                                          ,'num_positive'                    #1
                                          ,'num_positive_lower'              #2
                                          ,'num_positive_upper'              #3
                                          ,'num_tested'                      #4
                                          ,'num_hospitalized'                #5
                                          ,'num_cumulative_hospitalizations' #6
                                          ,'num_critical'                    #7
                                          ,'num_fatalities'                  #8
                                          ,'num_sym_lower'                   #9
                                          ,'num_sym_upper'                   #10
                                          ,'num_positive_mean'])             #11
