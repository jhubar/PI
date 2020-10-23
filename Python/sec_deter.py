
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math


class SIR():


    def __init__(self):

        self.beta = 0.5717
        self.gamma = 0.4282

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
            error += math.sqrt((dataset[i][1] - predictions[i][1] - predictions[i][2])**2)

        return error

    def fit(self):
        """
        Fit the model
        """
        # Load the dataset
        dataset = self.load_testing_data(args="cumul_positives")
        if dataset[0][1] == 0:
            dataset[0][1] = 1
        # Initial state
        initial_state = (1000000 - dataset[0][1], dataset[0][1], 0)
        # Sub parameters
        beta_min = 0.1
        beta_max = 0.9
        gamma_min = 0.1
        gamma_max = 0.9
        range_size = 200
        # Range vectors
        beta_range = np.linspace(beta_min, beta_max, range_size)
        gamma_range = np.linspace(gamma_min, gamma_max, range_size)
        # Least square matrix:
        SSE = np.zeros((range_size, range_size))
        # Fill the matrix:
        best = (math.inf, 0, 0)
        best_beta = []
        for b in range(0, range_size):
            best_tmp = (math.inf, 0)
            for g in range(0, range_size):
                parameters = (beta_range[b], gamma_range[g])
                SSE[b][g] = self.least_squares(parameters=parameters, initial=initial_state, dataset=dataset)
                if SSE[b][g] < best[0]:
                    best = (SSE[b][g], beta_range[b], gamma_range[g])
                if SSE[b][g] < best_tmp[0]:
                    best_tmp = (SSE[b][g], gamma_range[g])
                if SSE[b][g] > 1000:
                    SSE[b][g] = 200
            best_beta.append(best_tmp[1])
            print(b)
        self.plot_sse_space(SSE, beta_range, gamma_range)
        print("best: ")
        print(best)

        plt.scatter(beta_range, best_beta)
        plt.show()

    def plot_sse_space(self, SSE, beta_range, gamma_range):

        X, Y = np.meshgrid(beta_range, gamma_range)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, SSE)
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

    def load_testing_data(self, args="dataframe"):
        """
        Load the last dataset from git hub
        Returns: the matrix of data to fit the model
        """
        url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
        # load pandas dataframe and convert to nparray:
        df = pd.read_csv(url, sep=",", header=0).to_numpy()
        positives = np.vstack((df[:, 0], df[:, 1])).T

        if args == "positives":
            return positives
        if args == "cumul_positives":
            for i in range(1, positives.shape[0]):
                positives[i][1] = positives[i-1][1]
            return positives
        else:
            return df













def covid_20():

    model = SIR()
    model.fit()




if __name__ == "__main__":

    covid_20()