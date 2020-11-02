import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math


class SEIR():

    def __init__(self):
        # Parameter's values
        self.N = 999999
        self.beta = 0.3865
        self.gamma = 1 / 7  # 7 days average to be cure
        self.sigma = 1 / 3  # 3 days average incubation time
        self.hp = 1 / 20.89  # Hospit probability Proportion of infected people who begin hospitalized
        self.hcr = 0  # Hospit Cure Rate: proba de guérir en hospitalisation

    def set_hospit_prop(self, hospit_prop):
        self.hospit_prop = hospit_prop

    def differential(self, state, time, beta, gamma, sigma, hp, hcr):
        """
        Differential equations of the model
        """
        S, E, I, H, R = state

        dS = -(beta * S * I) / (S + E + I + H + R)
        dE = (beta * S * I) / (S + E + I + H + R) - E * sigma
        dI = (E * sigma) - (gamma * I) - (I * hp)
        dH = (I * hp * gamma) - hcr * H
        dR = gamma * I + hcr * H

        return dS, dE, dI, dH, dR


    def predict(self, S_0, E_0, I_0, H_0, R_0, duration):
        """
        Predict epidemic curves
        """
        # Initialisation vector:
        initial_state = (S_0, E_0, I_0, H_0, R_0)
        # Time vector:
        time = np.arange(duration)
        # Parameters vector
        parameters = (self.beta, self.gamma, self.sigma, self.hp, self.hcr)
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=parameters)

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4])).T


    def SSE(self, parameters, initial_state, time, data, method='fit_on_hospit'):

        # Set parameters:
        params = tuple(parameters)
        # Make predictions:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=params)

        if method == 'fit_on_hospit':
            # On fit alors que sur la courbe de hospitalisations
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][3] - predict[i][3]) ** 2
            return sse
        if method == 'fit_on_hospit_cumul':
            # si hcr est = à 0, on peut fiter la courbe des hospit avec cumul hopit car pas de guérison
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][4] - predict[i][3]) ** 2
            return sse
        if method == 'fit_on_cumul_positive':

            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][7] - predict[i][2] - predict[i][3] - predict[i][4]) ** 2
            return sse
        if method == 'fit_on_cumul_mixt':
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][7] - predict[i][2] - predict[i][3] - predict[i][4]) ** 2
                sse += (data[i][4] - predict[i][3]) ** 2
            return sse

    def fit_scipy_A(self, dataset):

        time = dataset[:, 0]
        # Set initial state:
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        I_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        S_0 = 999999 - H_0 - I_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, I_0, H_0, R_0)
        self.hp = 0
        start_values = [self.beta, self.gamma, self.sigma, self.hp, self.hcr]
        bounds = [(0, 1), (1/10, 1/4), (1/5, 1), (0, 0), (self.hcr, self.hcr)]
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, dataset, 'fit_on_cumul_positive'),
                       method='L-BFGS-B', bounds=bounds)
        print(res)
        self.beta = res.x[0]
        self.gamma = res.x[1]
        self.sigma = res.x[2]

    def fit_gamma(self, dataset):
        # Set initial state:
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        I_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        S_0 = 999999 - H_0 - I_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, I_0, H_0, R_0)
        time = dataset[:, 0]
        # Optimisation ittérative:
        range_size = 1000
        gamma_range = np.linspace(0, 1, range_size)
        best = (math.inf, 0)
        SSE = []
        for b in range(0, range_size):
            parameters = (self.beta, gamma_range[b], self.sigma, self.hp, self.hcr)
            sse = self.SSE(parameters, initial_state, time, dataset, method='fit_on_cumul_positive')
            SSE.append(sse)
            if sse < best[0]:
                best = (sse, gamma_range[b])
        print("Iterative best fit. Best value for gamma = {} with sse = {}".format(best[1], best[0]))
        # Set the best value of beta:
        self.gamma = best[1]
        # print graph of beta evolution with sse:
        plt.plot(gamma_range, SSE, c='blue', label='SSE evolution')
        plt.yscale('log')
        plt.xlabel('gamma value')
        plt.show()

    def fit_beta(self, dataset):
        # Set initial state:
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        I_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        S_0 = 999999 - H_0 - I_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, I_0, H_0, R_0)
        time = dataset[:, 0]
        # Optimisation ittérative:
        range_size = 1000
        beta_range = np.linspace(0, 1, range_size)
        best = (math.inf, 0)
        SSE = []
        for b in range(0, range_size):
            parameters = (beta_range[b], self.gamma, self.sigma, self.hp, self.hcr)
            sse = self.SSE(parameters, initial_state, time, dataset, method='fit_on_cumul_positive')
            SSE.append(sse)
            if sse < best[0]:
                best = (sse, beta_range[b])
        print("Iterative best fit. Best value for beta = {} with sse = {}".format(best[1], best[0]))
        # Set the best value of beta:
        self.beta = best[1]
        # print graph of beta evolution with sse:
        plt.plot(beta_range, SSE, c='blue', label='SSE evolution')
        plt.yscale('log')
        plt.xlabel('beta value')
        plt.show()

    def fit_sigma(self, dataset):
        # Set initial state:
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        I_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        S_0 = 999999 - H_0 - I_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, I_0, H_0, R_0)
        time = dataset[:, 0]
        # Optimisation ittérative:
        range_size = 1000
        sigma_range = np.linspace(0, 1, range_size)
        best = (math.inf, 0)
        SSE = []
        done1 = False

        for b in range(0, range_size):
            parameters = (self.beta, self.gamma, sigma_range[b], self.hp, self.hcr)
            sse = self.SSE(parameters, initial_state, time, dataset, method='fit_on_cumul_positive')
            SSE.append(sse)
            if sigma_range[b] >= 0.3333 and not done1:
                done1 = True
                print("SSE for sigma = {} = {}".format(sigma_range[b], sse))
            if sse < best[0]:
                best = (sse, sigma_range[b])
        print("Iterative best fit. Best value for sigma = {} with sse = {}".format(best[1], best[0]))
        # Set the best value of beta:
        self.sigma = best[1]
        # print graph of beta evolution with sse:
        plt.plot(sigma_range, SSE, c='blue', label='SSE evolution')
        plt.yscale('log')
        plt.xlabel('sigma value')
        plt.show()


def plot_predict_and_compare(df, pred, args='predict'):
    if 'predict' in args:
        # just print predicted epidemic curves
        if "no_S" not in args:
            plt.plot(pred[:, 0], pred[:, 1], c='black', label="S")
        plt.plot(pred[:, 0], pred[:, 2], c='yellow', label="E")
        plt.plot(pred[:, 0], pred[:, 3], c='red', label="I")
        plt.plot(pred[:, 0], pred[:, 4], c='purple', label="H")
        plt.plot(pred[:, 0], pred[:, 5], c='blue', label='R')
        plt.show()

    if 'compare' in args:
        plt.scatter(df['Day'], df['cumul_positive'], c='red')
        plt.scatter(df['Day'], df['num_hospitalised'], c='blue')

        cumul_positive = []
        hospit = []
        for i in range(0, len(df['Day'].to_numpy())):
            cumul_positive.append(pred[i][3] + pred[i][4] + pred[i][5])  # somme de I, H et R
            hospit.append(pred[i][4])
        plt.plot(df['Day'], cumul_positive, c='red')
        plt.plot(df['Day'], hospit, c='blue')
        if "log" in args:
            plt.yscale('log')
        plt.show()




def first_method():
    """
    Import the dataset and add informations
    """
    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
    df = pd.read_csv(url, sep=",", header=0)
    # Delete the first line:
    df = df.drop([0], axis=0)
    # Insert cumul_positive column at the end
    cumul_positive = df["num_positive"].to_numpy()
    for i in range(1, len(cumul_positive)):
        cumul_positive[i] += cumul_positive[i - 1]
    df.insert(7, "cumul_positive", cumul_positive)
    # print(df)
    # Make a numpy version of the dataframe:
    df_np = df.to_numpy()
    # Init the model:
    model = SEIR()
    # Set initial state
    H_0 = df_np[0][7]
    E_0 = 3 * df_np[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
    I_0 = df_np[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
    S_0 = 999999 - H_0 - I_0 - E_0
    R_0 = 0

    """ *****************************************************************************
        ETAPE 1: 
        Fit_scipy A:
        On cherche les parametres beta, gamma et sigma en fitant le cumul des positifs
        avec I + H + R. On considère que hp fait partie de gamma, on séparera les deux
        paramètres à l'étape suivante
        *****************************************************************************
    """

    model.fit_scipy_A(df_np)

    model.sigma = 8

    model.fit_scipy_A(df_np)

    model.fit_beta(df_np)

    model.fit_gamma(df_np)

    #model.fit_sigma(df_np)




    predictions = model.predict(S_0, E_0, I_0, H_0, R_0, duration=150)
    plot_predict_and_compare(df, predictions, args='predict compare')

    predictions = model.predict(S_0, E_0, I_0, H_0, R_0, duration=50)

    plot_predict_and_compare(df, predictions, args='predict compare no_S log')
    print("final value for model's parameters: ")
    print(" beta = {}".format(model.beta))
    print(" gamma = {}".format(model.gamma))
    print(" sigma = {}".format(model.sigma))
    print(" hp = {}".format(model.hp))
    print(" hcr = {}".format(model.hcr))


if __name__ == "__main__":
    # first_method()
    first_method()


