
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math


class SIR():


    def __init__(self):

        self.beta = 0.57009
        #self.beta = 0.35427
        self.gamma = 0.48789
        #self.gamma = 0.18040


    def differential(self, SIR_values, time, beta=-1, gamma=-1):
        """
        Differential equations of the model
        """
        b = beta
        g = gamma
        if b == -1:
            b = self.beta
        if g == -1:
            g = self.gamma
        S = SIR_values[0]
        I = SIR_values[1]
        R = SIR_values[2]
        dS = -b * S * I / (S + I + R)
        dI = (b * S * I / (S + I + R)) - (g * I)
        dR = g * I
        return dS, dI, dR

    def least_squares(self, parameters, initial, dataset):
        """
        Compute least squares error
        """
        time = np.arange(dataset.shape[0])
        # Make predictions
        predictions = odeint(self.differential, initial, time, args=parameters)
        # Compute sum of squared error
        error = 0
        for i in range(0, dataset.shape[0]):
            #print("iter: {}, dataset= {}, pred I ={}, pred R ={}".format(i, dataset[i][1], predictions[i][1] , predictions[i][2]))
            #error += (round(dataset[i][1], 4) - round(predictions[i][1], 4) - round(predictions[i][2], 4))**2
            error += (dataset[i][1] - predictions[i][2]) ** 2

        return error

    def fit(self, data="covid_20"):
        """
        Fit the model
        """
        # Load the dataset
        if data == "covid_20":
            dataset = self.load_testing_data(args="positives", dataset=data)
        else:
            dataset = self.load_testing_data(args="cumul_positives", dataset=data)
        if dataset[0][1] == 0:
            dataset[0][1] = 1
        pop_size = 1000000
        if data == "covid_19":
            pop_size = 11500000
        # Initial state
        initial_state = (pop_size - dataset[0][1], dataset[0][1], 0)
        # Sub parameters
        beta_min = 0
        beta_max = 0.9
        gamma_min = 0.1
        gamma_max = 0.9
        range_size = 200
        # Range vectors
        beta_range = np.linspace(beta_min, beta_max, range_size)
        #beta_range = np.round(beta_range, decimals=4)
        gamma_range = np.linspace(gamma_min, gamma_max, range_size)
        #gamma_range = np.round(gamma_range, decimals=4)
        # Least square matrix:
        SSE = np.zeros((range_size, range_size))
        # Fill the matrix:
        best = (math.inf, 0, 0)
        best_beta = []
        best_beta_sse = []
        R_0_repport = []
        for b in range(0, range_size):
            best_tmp = (math.inf, 0)
            for g in range(0, range_size):
                parameters = (beta_range[b], gamma_range[g])
                SSE[b][g] = self.least_squares(parameters=parameters, initial=initial_state, dataset=dataset)
                if SSE[b][g] < best[0]:
                    best = (SSE[b][g], beta_range[b], gamma_range[g])
                if SSE[b][g] < best_tmp[0]:
                    best_tmp = (SSE[b][g], gamma_range[g])

            best_beta.append(best_tmp[1])
            best_beta_sse.append(best_tmp[0])
            R_0_repport.append(beta_range[b]/best_tmp[1])
            print(b)
        self.plot_sse_space(SSE, beta_range, gamma_range)
        print("best: ")
        print(best)

        plt.plot(beta_range, best_beta)
        plt.show()
        plt.plot(beta_range, best_beta_sse)
        plt.show()
        print(R_0_repport)

    def sequential_fit(self, data="covid_20"):

        # Load the dataset
        dataset = self.load_testing_data(args="cumul_positives", dataset=data)
        if dataset[0][1] == 0:
            dataset[0][1] = 1
        pop_size = 1000000
        if data == "covid_19":
            pop_size = 11500000
        # Initial state
        initial_state = (pop_size - dataset[0][1], dataset[0][1], 0)
        # Sub parameters
        beta_min = 0.0
        beta_max = 0.8
        gamma_min = 0.0
        gamma_max = 0.7
        range_size = 200
        # Init gamma value to 1:
        self.gamma = 2

        # fit beta:
        self.fit_beta(dataset, beta_min, beta_max, range_size, pop_size, print_graph=True)
        # fit gamma:
        self.fit_gamma(dataset, gamma_min, gamma_max, range_size, pop_size, print_graph=True)


    def fit_beta(self, dataset, min_val=0, max_val=1, range_size=200, pop_size=1000000, print_graph=False):

        # Make beta_range:
        beta_range = np.linspace(min_val, max_val, range_size)
        beta_range = np.round(beta_range, decimals=4)
        # Set initial state:
        init_state = (pop_size - dataset[0][1], dataset[0][1], 0)
        # Compute SSE:
        SSE = []
        best_val = (math.inf, 0)
        for i in range(0, range_size):
            parameters = (beta_range[i], self.gamma)
            SSE.append(self.least_squares(parameters=parameters, initial=init_state, dataset=dataset))
            if SSE[i] < best_val[0]:
                best_val = (SSE[i], beta_range[i])
        self.beta = best_val[1]
        if print_graph:
            plt.plot(beta_range, SSE)
            plt.xlabel("beta value")
            plt.yscale('log')
            plt.ylabel("SSE")
            plt.title("Effect of beta on SSE with a value of gamma = {}. Best beta = {}".format(self.gamma, best_val[1]))
            plt.show()
            print("Beta fit with gamma={} : {} with an SSE of: {}".format(self.gamma, best_val[1], best_val[0]))
    def fit_gamma(self, dataset, min_val=0, max_val=1, range_size=200, pop_size=1000000, print_graph=False):

        # Make gamma_range:
        gamma_range = np.linspace(min_val, max_val, range_size)
        gamma_range = np.round(gamma_range, decimals=4)
        # Set initial state:
        init_state = (pop_size - dataset[0][1], dataset[0][1], 0)
        # Compute SSE:
        SSE = []
        best_val = (math.inf, 0)
        for i in range(0, range_size):
            parameters = (self.beta, gamma_range[i])
            SSE.append(self.least_squares(parameters=parameters, initial=init_state, dataset=dataset))
            if SSE[i] < best_val[0]:
                best_val = (SSE[i], gamma_range[i])

        if print_graph:
            plt.plot(gamma_range, SSE)
            plt.xlabel("gamma value")
            plt.yscale('log')
            plt.ylabel("SSE")
            plt.title("Effect of gamma on SSE with a value of beta = {}. Best gamma = {}".format(self.beta, best_val[1]))
            plt.show()
            print("Gamma fit with beta={} : {} with an SSE of: {}".format(self.beta, best_val[1], best_val[0]))

    def plot_sse_space(self, SSE, beta_range, gamma_range):

        X, Y = np.meshgrid(beta_range, gamma_range)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, SSE)
        ax.set_zscale('log')
        ax.view_init(15, 60)
        plt.show()

    def predict(self, S_0, I_0, R_0, duration):
        """
        Predict epidemic curves
        """
        # Initialisation vector:
        init = [S_0, I_0, R_0]
        # Time vector:
        time = np.arange(duration)
        # Make predictions:
        predict = odeint(self.differential, init, time)
        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2])).T

    def plot_curves(self, predictions, title="Epidemic curves"):
        """
        plot epidemic curves
        """
        plt.plot(predictions[:, 0], predictions[:, 1], c="green", label="S")
        plt.plot(predictions[:, 0], predictions[:, 2], c="red", label="I")
        plt.plot(predictions[:, 0], predictions[:, 3], c="blue", label="R")
        plt.title(title)
        plt.xlabel("Time in days")
        plt.ylabel("Number of people")
        plt.show()

    def load_testing_data(self, args="dataframe", dataset="covid_20"):
        """
        Load the last dataset from git hub
        Returns: the matrix of data to fit the model
        """
        if dataset == "covid_20":
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # load pandas dataframe and convert to nparray:
            df = pd.read_csv(url, sep=",", header=0)
            # Remove first row
            #df = df.drop([0], axis=0)
            # Convert to numpy array
            np_df = df.to_numpy()
            # Sum confirmed and hospit ??? voir si nécessaire
            # np_df[:, 1] = np_df[:, 1] + np_df[:, 2]
            time = np.arange(np_df.shape[0])
            positives = np.vstack((time, np_df[:, 3])).T
        if dataset == "covid_19":
            url = "https://raw.githubusercontent.com/julien1941/PI/master/R/cov_19_be.csv?token=AOOPK5GEPW4DUM5XDUD73OC7TVWKC"
            # load pandas dataframe
            df = pd.read_csv(url, sep=",", header=0)
            # Remove 40 first rows
            df = df.to_numpy()
            df = df[14:69, :]
            for i in range(0, df.shape[0]):
                df[i][0] = i
            positives = np.vstack((df[:, 0], df[:, 2])).T
            positives[0] = 1
        if args == "positives":
            return positives
        if args == "cumul_positives":
            for i in range(1, positives.shape[0]):
                positives[i][1] = positives[i-1][1] + positives[i][1]
            return positives
        else:
            return df


    def compare_with_dataset(self, data="covid_20"):
        """
        Compare model's result with given dataset
        """
        # Load the dataset of cumulative confirmed cases
        if data == "covid_20":
            dataset = self.load_testing_data(args="positives", dataset=data)
        else:
            dataset = self.load_testing_data(args="cumul_positives", dataset=data)
        print(dataset)
        # Make predictions
        predictions = self.predict(S_0=1000000-dataset[0][1], I_0=dataset[0][1], R_0=0, duration=dataset.shape[0])
        # Compute sum of I and R
        print(predictions.shape)
        print(predictions)
        IR = predictions[:, 2] + predictions[:, 3]

        plt.scatter(dataset[:, 0], dataset[:, 1], c="green")
        plt.plot(dataset[:, 0], IR, c="red")
        plt.show()








def covid_20():

    model = SIR()
    model.fit(data='covid_20')
    model.compare_with_dataset()

def covid_19():

    model = SIR()
    model.compare_with_dataset(data="covid_19")
    #model.fit(data="covid_19")
    #model.sequential_fit(data="covid_19")



    pass





if __name__ == "__main__":

    covid_20()
    #covid_19()