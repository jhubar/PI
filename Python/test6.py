import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from scipy.signal import savgol_filter

"""
=======================================================================================================
SEIHR +  MODEL
=======================================================================================================
"""


class SEIR():

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

        self.sensitivity = 0.775

        # Data to fit
        self.raw_dataset = None  # Original dataset, before preprocessing
        self.dataset = None  # Numpy matrix format
        self.dataframe = None  # Dataframe format
        self.np_raw_data = None

        # Initial state: to be used to make predictions
        self.S_0 = None  # Sensible: peoples who can catch the agent
        self.E_0 = None  # Exposed: people in incubation: Can't spread the agent
        self.I_0 = None  # Infectious: people who can spread the disease
        self.H_0 = None  # Hospitalized peoples: can't spread any more the agent
        self.C_0 = None  # Critical: peoples who are in ICU
        self.R_0 = None  # Recovered people: can't catch again the agent due to immunity
        self.D_0 = None  # Dead: people who die in ICU
        self.N = None  # The total size of the population

        # Uncertainty infected
        # self.infected_lower_bound = 0
        # self.infected_upper_bound = 0
        # self.cp.Uniform(self.infected_lower_bound,self.infected_upper_bound)

        # Data to store
        self.dataJSON = {}


    def saveJson(self):
        with open('Data/SEIR+.json', 'w') as outfile:
            json.dump(self.dataJSON, outfile)

    def get_initial_state(self):
        """
        Function who return a tuple with the initial state of the model
        """
        return (self.S_0, self.E_0, self.I_0, self.H_0, self.R_0, self.N, self.C_0, self.D_0)

    def get_initial_state_to_fit(self):
        """
        Function who return a tuple with the initial state of the model
        """
        return (self.S_0, self.E_0, self.I_0, self.H_0, self.R_0, self.N, self.C_0, self.D_0, self.I_0)

    def get_parameters(self):
        """
        Function who return a tuple with model's hyper-parameters
        """
        return (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr)

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr, sensitivity):
        """
        Differential equations of the model
        """
        S, E, I, H, R, N, C, D = state

        dS = -(beta * S * I) / N
        dE = (beta * S * I) / N - E * sigma
        dI = (E * sigma) - (gamma * I) - (hp * I)
        dH = (hp * I) - (hcr * H) - (pc * H)
        dC = (pc * H) - (pd * C) - (pcr * C)
        dR = (gamma * I) + (hcr * H)
        dD = (pd * C)
        dN = 0

        return dS, dE, dI, dH, dR, dN, dC, dD

    def differential_fitting(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr, sensitivity):
        """
        Differential equations of the model
        """
        S, E, I, H, R, N, C, D, Conta = state

        dS = -(beta * S * I) / N
        dE = (beta * S * I) / N - E * sigma
        dI = (E * sigma) - (gamma * I) - (hp * I)
        dH = (hp * I) - (hcr * H) - (pc * H)
        dC = (pc * H) - (pd * C) - (pcr * C)
        dR = (gamma * I) + (hcr * H)
        dD = (pd * C)
        dN = 0

        dconta = (beta * S * I) / N

        return dS, dE, dI, dH, dR, dN, dC, dD, dconta


    def predict(self, duration):
        """
        Predict epidemic curves from t_0 for the given duration
        """
        # Initialisation vector:
        initial_state = self.get_initial_state()
        # Time vector:
        time = np.arange(duration)
        # Parameters vector
        parameters = self.get_parameters()
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=parameters)

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4],
                          predict[:, 5], predict[:, 6], predict[:, 7])).T

    def fit(self):
        """
        This method use the given data to find values of our model who minimise square error
        between predictions and original data.
        """

        # Generate initial state:
        initial_state = self.get_initial_state_to_fit()
        # Time vector:
        time = self.dataset[:, 0]
        # Bounds: Given ranges for beta, sigma and gamma
        #bounds = [(0, 1), (1 / 5, 1), (1 / 10, 1 / 4), (0.7, 0.85)]
        bounds = [(0, 1), (0, 1), (0, 1), (0.7, 0.85)]
        # Start values
        start_values = [self.beta, self.sigma, self.gamma, self.sensitivity]
        # Use Scipy.optimize.minimize with L-BFGS_B method
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, 'method_1'),
                       method='L-BFGS-B', bounds=bounds, options={"maxfun": 25000})

        print(res)
        self.beta = res.x[0]
        self.sigma = res.x[1]
        self.gamma = res.x[2]
        self.sensitivity = res.x[3]


    def SSE(self, parameters, initial_state, time, method='fit_on_cumul_positive'):

        if method == 'method_1':
            # Set parameters:
            tpl = tuple(parameters)
            params = (tpl[0], tpl[1], tpl[2], 0, 0, 0, 0, 0, tpl[3])
            print(params)
            sensitivity = tpl[3]
            # Make predictions:
            predict = odeint(func=self.differential_fitting,
                             y0=initial_state,
                             t=time,
                             args=params)

            # make a non cumulative positive array:
            contam = []
            contam.append(predict[0][8])
            for i in range(1, predict.shape[0]):
                contam.append(predict[i][8] - predict[i-1][8])

            # Pour chaque pred(t), on calcule p(Et|x_t)
            # Ca suit une binomiale dependant de la sensibilité
            proba = []

            for t in range(0, len(time)):
                k = self.np_raw_data[t][1]
                n = int(contam[t])
                # print('k= {} - n= {}'.format(k, n))
                if k > n:

                    proba.append(0)
                else:

                    # cumpute probability from binomial distribution
                    num = math.factorial(k)
                    denom = math.factorial(k) * math.factorial(n - k)
                    pb = num/denom
                    pb = pb * sensitivity ** k * (1 - sensitivity) ** (n - k)
                    proba.append(pb)
            print( -100 * sum(proba))
            return (-100) * np.sum(proba)



        return sse

    def dataframe_smoothing1(self, df):
        # Convolution
        # Convert the dataframe to a numpy array:
        np_df = df.to_numpy()
        # Smoothing period = 7 days
        smt_prd = 7
        smt_vec = np.ones(smt_prd)
        smt_vec /= smt_prd
        # Sore smoothed data in a new matrix:
        smoothed = np.copy(np_df)
        # How many smothing period can we place in the dataset:
        nb_per = math.floor(np_df.shape[0] / smt_prd)
        # Perform smoothing for each attributes
        for i in range(1, np_df.shape[1]):
            smoothed[:, i] = np.convolve(np_df[:, i], smt_vec, mode="same")
            # Don't smooth the last week
            for j in range(smoothed.shape[0] - smt_prd, smoothed.shape[0]):
                smoothed[j][i] = np_df[j][i]

        # Write new values in a dataframe
        new_df = pd.DataFrame(smoothed, columns=df.columns)

        return new_df

    def dataframe_smoothing(self, df):
        # From andreas NRMAS
        # Convert the dataframe to a numpy array:
        np_df = df.to_numpy()
        # Smoothing period = 7 days
        smt_prd = 7
        smt_vec = np.ones(smt_prd)
        smt_vec /= smt_prd
        # Sore smoothed data in a new matrix:
        smoothed = np.copy(np_df)
        # How many smothing period can we place in the dataset:
        nb_per = math.floor(np_df.shape[0] / smt_prd)
        # Perform smoothing for each attributes
        for i in range(1, np_df.shape[1]):
            smoothed[:, i] = own_NRMAS(np_df[:, i], 7)

        # Write new values in a dataframe
        new_df = pd.DataFrame(smoothed, columns=df.columns)

        return new_df

    def dataframe_smoothing3(self, df):
        # From andreas Own MAS
        # Convert the dataframe to a numpy array:
        np_df = df.to_numpy()
        # Smoothing period = 7 days
        smt_prd = 7
        smt_vec = np.ones(smt_prd)
        smt_vec /= smt_prd
        # Sore smoothed data in a new matrix:
        smoothed = np.copy(np_df)
        # How many smothing period can we place in the dataset:
        nb_per = math.floor(np_df.shape[0] / smt_prd)
        # Perform smoothing for each attributes
        for i in range(1, np_df.shape[1]):
            smoothed[:, i] = own_MAS(np_df[:, i], 7)

        # Write new values in a dataframe
        new_df = pd.DataFrame(smoothed, columns=df.columns)

        return new_df

    def dataframe_smoothing2(self, df):
        # Salvago
        # Convert the dataframe to a numpy array:
        np_df = df.to_numpy()
        # Smoothing period = 7 days
        smt_prd = 7
        smt_vec = np.ones(smt_prd)
        smt_vec /= smt_prd
        # Sore smoothed data in a new matrix:
        smoothed = np.copy(np_df)
        # How many smothing period can we place in the dataset:
        nb_per = math.floor(np_df.shape[0] / smt_prd)
        # Perform smoothing for each attributes
        for i in range(1, np_df.shape[1]):
            smoothed[:, i] = savgol_filter(np_df[:, i], smt_prd, 2)

        # Write new values in a dataframe
        new_df = pd.DataFrame(smoothed, columns=df.columns)

        return new_df

    def import_dataset(self, target='covid_20'):

        if target == 'covid_20':
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # Import the dataframe:
            self.raw_dataset = pd.read_csv(url, sep=',', header=0)
            self.raw_dataset['num_positive'][0] = 1
            self.np_raw_data = self.raw_dataset.to_numpy()

            # ========================================================================= #
            # Pre-processing
            # ========================================================================= #

            # Smoothing
            self.dataframe = self.dataframe_smoothing(self.raw_dataset)

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
            plt.savefig("fig/preprocessing.png", transparent=True)
            # plt.show()
            plt.close()

            # Ad a new column at the end with cumulative positive cases at the right
            cumul_positive = self.dataframe['num_positive'].to_numpy()
            cumul_positive_non_smooth = self.raw_dataset['num_positive']
            for i in range(1, len(cumul_positive)):
                cumul_positive[i] += cumul_positive[i - 1]
                cumul_positive_non_smooth[i] += cumul_positive_non_smooth[i - 1]
            self.dataframe.insert(7, 'cumul_positive', cumul_positive)
            self.raw_dataset.insert(7, 'cumul_positive', cumul_positive_non_smooth)

            # Delete the first line with zero test
            #for i in range(0, 1):
            #    self.raw_dataset.drop(axis=0, index=i, inplace=True)
            #    self.dataframe.drop(axis=0, index=i, inplace=True)
            # To reset dataframe index:
            tmp = self.raw_dataset.to_numpy()
            self.raw_dataset = pd.DataFrame(tmp, columns=self.raw_dataset.columns)
            tmp = self.dataframe.to_numpy()
            self.dataframe = pd.DataFrame(tmp, columns=self.dataframe.columns)

            # Store a numpy version:
            self.dataset = self.dataframe.to_numpy()

            # Store the initial state who fit with input data
            self.N = 1000000
            self.I_0 = 16
            self.H_0 = self.dataset[0][3]
            self.E_0 = 5 * self.I_0  # Because mean of incubation period = 3 days
            self.R_0 = 0
            self.C_0 = 0
            self.D_0 = 0
            self.S_0 = self.N - self.I_0 - self.H_0 - self.E_0

            # Initialize default value to hyper-parameters:
            self.beta = 0.5
            self.sigma = 1 / 2
            self.gamma = 1 / 5
            self.hp = 0
            self.hcr = 0

    def plot_predict(self, pred, args='no_S'):

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

        if 'predict' in args:
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

                plt.savefig("fig/long_time_predictions_no_s.png", transparent=True)
            else:
                plt.savefig("fig/long_time_predictions.png", transparent=True)
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


def own_NRMAS_index(vector, window, index):
    smoothed_value = 0
    nb_considered_values = 0
    max_size = (window - 1) / 2
    smoothing_window = np.arange(-max_size, max_size + 1, 1)

    for j in range(window):

        sliding_index = int(index + smoothing_window[j])

        if (sliding_index >= 0) and (sliding_index <= len(vector) - 1):
            smoothed_value += vector[sliding_index]
            nb_considered_values += 1

    return smoothed_value / nb_considered_values


def own_NRMAS(vector, window):
    smoothed_vector = np.zeros(len(vector))

    if (window % 2) == 0:
        print("Error window size even")
        return

    for i in range(len(vector)):
        smoothed_vector[i] = own_NRMAS_index(vector, window, i)

    return smoothed_vector


def own_MAS_index(vector, window, index):
    smoothed_value = 0
    max_size = (window - 1) / 2

    # cas de base
    if (window == 1):
        smoothed_value = vector[index]

    # case not boundaries
    elif ((index - max_size) >= 0) and ((index + max_size) <= len(vector) - 1):
        smoothing_window = np.arange(-max_size, max_size + 1, 1)
        for j in range(window):
            smoothed_value += vector[int(index + smoothing_window[j])] / window

    # recusivité
    else:
        return own_MAS_index(vector, window - 2, index)

    return smoothed_value


def own_MAS(vector, window):
    smoothed_vector = np.zeros(len(vector))

    if (window % 2) == 0:
        print("Error window size even")
        return

    for i in range(len(vector)):
        smoothed_vector[i] = own_MAS_index(vector, window, i)

    return smoothed_vector

def first_method():
    # Initialize the model
    model = SEIR()

    # Import the dataset:
    model.import_dataset(target='covid_20')

    # Fit the model:
    model.fit()




if __name__ == "__main__":
    first_method()
