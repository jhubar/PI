import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
import random

class bayesian_uncertainty():

    def __init__(self):

        # Model's hyperparameters
        self.beta = 0  # Contagion probability
        self.sigma = 0  # Probability to go from E to I
        self.gamma = 0  # Probability parameter to go from I to R (to be cure)
        self.hp = 0  # Probability to go from I to H
        self.hcr = 0  # Hospit Cure Rate
        self.pc = 0  # Probability to fall in ICU each day from H
        self.pd = 0  # Probability to die each day in icu
        self.pcr = 0  # Probability to recover from critical

        # Data to fit
        self.raw_dataset = None  # Original dataset, before preprocessing
        self.dataset = None  # Numpy matrix format
        self.dataframe = None  # Dataframe format

        # Initial state: to be used to make predictions
        self.S_0 = None  # Sensible: peoples who can catch the agent
        self.E_0 = None  # Exposed: people in incubation: Can't spread the agent
        self.I_0 = None  # Infectious: people who can spread the disease
        self.H_0 = None  # Hospitalized peoples: can't spread any more the agent
        self.C_0 = None  # Critical: peoples who are in ICU
        self.R_0 = None  # Recovered people: can't catch again the agent due to immunity
        self.D_0 = None  # Dead: people who die in ICU
        self.N = None  # The total size of the population

        # Data to store
        self.dataJSON = {}
    def dataframe_uncertainty(self, df):

        # Convert the dataframe to a numpy array:
        np_df = df.to_numpy()
        np_df_upper = np.copy(np_df)
        np_df_lower = np.copy(np_df)
        sensitivity_upper_bound = 0.7
        sensitivity_lower_bound = 0.85

        day = np_df[:,0]
        num_positive_lower = np.array(np_df[:,1]+np_df[:,1]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))
        num_positive_upper = np.array(np_df[:,1]+np_df[:,1]*((1-sensitivity_lower_bound)/sensitivity_lower_bound))
        num_tested = np_df[:,2]
        num_hospitalised =  np_df[:,3]
        num_cumulative_hospitalizations = np_df[:,4]
        num_critical = np_df[:,5]
        num_fatalities = np_df[:,6]
        num_sym_lower = np_df[:,2]
        num_sym_upper = np_df[:,2]*2


        new_df = np.vstack((day,num_positive_lower,num_positive_upper,num_tested
            ,num_hospitalised,num_cumulative_hospitalizations,num_critical,num_fatalities
            ,num_sym_lower,num_sym_upper))




        return new_df

    def import_dataset(self, target='covid_20'):

        if target == 'covid_20':
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # Import the dataframe:
            self.raw_dataset = pd.read_csv(url, sep=',', header=0)

            # ========================================================================= #
            #  Uncertainty
            # ========================================================================= #


            self.dataframe = self.dataframe_uncertainty(self.raw_dataset)

            print(self.dataframe)

def first_method():
    # Initialize the model
    data_with_uncertainty = bayesian_uncertainty()

    # Import the dataset:
    data_with_uncertainty.import_dataset(target='covid_20')
    print(data_with_uncertainty)

    # plt.show()
    plt.close()



if __name__ == "__main__":
    first_method()
