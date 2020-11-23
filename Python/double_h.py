import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math

"""
=======================================================================================================

=======================================================================================================
"""


class SEIR():

    def __init__(self):
        # Parameter's values
        self.N = 999999
        self.beta = 0.3865
        self.gamma = 1 / 7  # 7 days average to be cure
        self.sigma = 1 / 3  # 3 days average incubation time
        self.hpA = 1 / 20.89  # Hospit probability Proportion of infected people who begin hospitalized
        self.hpB = 1/18
        self.hcr = 0  # Hospit Cure Rate: proba de guérir en hospitalisation

    


    def set_hospit_prop(self, hospit_prop):
        self.hospit_prop = hospit_prop

    def differential(self, state, time, beta, gamma, sigma, hpA, hpB, hcr):
        """
        Differential equations of the model
        """
        S, E, IA, IB, H, R = state

        dS = -(beta * S * (IA + IB)) / (S + E + (IA + IB) + H + R)
        dE = (beta * S * (IA + IB)) / (S + E + (IA + IB) + H + R) - E * sigma
        dIA = (1 - hpA) * (E * sigma) - (gamma * IA)
        dIB = hpA * E * sigma - hpB * IB
        dH = hpB * (E * sigma) - hcr * H
        dR = gamma * IA + hcr * H

        return dS, dE, dIA, dIB, dH, dR

    def predict(self, S_0, E_0, IA_0, IB_0, H_0, R_0, duration):
        """
        Predict epidemic curves
        """
        # Initialisation vector:
        initial_state = (S_0, E_0, IA_0, IB_0, H_0, R_0)
        # Time vector:
        time = np.arange(duration)
        # Parameters vector
        parameters = (self.beta, self.gamma, self.sigma, self.hpA, self.hpB, self.hcr)
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=parameters)

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4], predict[:, 5])).T

    def brute_force(self, dataset):

        range_size = 50
        beta_range = np.linspace(2, 8, 50)
        sigma_range = np.linspace(1/5, 1, 50)
        gamma_range = np.linspace(1/10, 1/4, 50)
        hpA_range = np.linspace(1/25, 1/15, 10)
        hpB_range = np.linspace(1/10, 1/4, 50)
        self.hcr = 0

        time = dataset[:, 0]
        # Set initial state
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        IA_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        IB_0 = 1
        S_0 = 999999 - H_0 - IA_0 - IB_0 - E_0
        R_0 = 0

        initial_state = (S_0, E_0, IA_0, IB_0, H_0, R_0)

        smallest_sse = (math.inf, 0, 0, 0, 0, 0)

        nb_iter = len(beta_range) * len(sigma_range) * len(gamma_range) * len(hpA_range) * len(hpB_range)

        iter = 0
        for b in range(0, len(beta_range)):  # Beta
            for s in range(0, len(sigma_range)):   # sigma
                for g in range(0, len(gamma_range)):  # gamma
                    for ha in range(0, len(hpA_range)):
                        for hb in range(0, len(hpB_range)):
                            params = (beta_range[b], gamma_range[g], sigma_range[s],
                                      hpA_range[ha], hpB_range[hb], 0)
                            sse = self.SSE(params, initial_state, time, dataset, 'fit_on_cumul_mixt')
                            if sse < smallest_sse[0]:
                                smallest_sse = (sse, beta_range[b], gamma_range[g], sigma_range[s],
                                                hpA_range[ha], hpB_range[hb])
                            iter += 1
                        loading = (iter/nb_iter)*100
                        print("fitting: {}%".format(loading))

        print("Best fit: beta = {}, gamma = {}, sigma = {}".format(smallest_sse[1], smallest_sse[2], smallest_sse[3]))
        print("hpA = {}, hpB = {}".format(smallest_sse[4], smallest_sse[5]))
        print("With sse of {}".format(smallest_sse[0]))
        self.beta = smallest_sse[1]
        self.gamma = smallest_sse[2]
        self.sigma = smallest_sse[3]
        self.hpA = smallest_sse[4]
        self.hpB = smallest_sse[5]


    def fit_scipy(self, dataset):
        """
        On commence par fitter les paramètres beta (dans un interval de 0 à 1, pas d'indications pour lui),
        gamma dans un interval de 1/10 à 1/4 donné dans l'énnoncé, et sigma dans un interval de 1/5 à 1
        (également donné).
        Nous fittons les données de testage cummulées sur la somme des courbes I, H et R.

        """
        time = dataset[:, 0]
        # Set initial state
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        IA_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        IB_0 = 1
        S_0 = 999999 - H_0 - IA_0 - IB_0 - E_0
        R_0 = 0

        initial_state = (S_0, E_0, IA_0, IB_0, H_0, R_0)
        start_values = [self.beta, self.gamma, self.sigma, self.hpA, self.hpB, self.hcr]
        bounds = [(0, 1), (1 / 10, 1 / 4), (1 / 5, 1), (0, 1), (1/10, 1/4), (self.hcr, self.hcr)]
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, dataset, 'fit_on_hospit_cumul'),
                       method='L-BFGS-B', bounds=bounds)
        print(res)
        self.beta = res.x[0]
        self.gamma = res.x[1]
        self.sigma = res.x[2]
        self.hpB = res.x[4]
        """
        Deuxième étape, il ne reste plus qu'à fitter hcr sur la courbe des hospitalisations (non cumulées)
        NOTE : Je ne sais pas pourquoi, mais ça converge pas quand j'utilise minimize alors que ça converge très bien
        en itérant toutes les valeurs
        """
        #self.fit_hcr(dataset)

    def fit_sigma(self, dataset):
        # Set initial state
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        IA_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        IB_0 = 1
        S_0 = 999999 - H_0 - IA_0 - IB_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, IA_0, IB_0, H_0, R_0)
        time = dataset[:, 0]
        # Optimisation ittérative:
        range_size = 1000
        sigma_range = np.linspace(0, 1, range_size)
        best = (math.inf, 0)
        SSE = []
        done1 = False

        for b in range(0, range_size):
            parameters = (self.beta, self.gamma, sigma_range[b], self.hpA, self.hpB, self.hcr)
            sse = self.SSE(parameters, initial_state, time, dataset, method='fit_on_cumul_mixt')
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

    def fit_hp_b(self, dataset):
        # Set initial state
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        IA_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        IB_0 = 1
        S_0 = 999999 - H_0 - IA_0 - IB_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, IA_0, IB_0, H_0, R_0)
        time = dataset[:, 0]
        # Optimisation ittérative:
        range_size = 1000
        hp_range = np.linspace(0, 1, range_size)
        best = (math.inf, 0)
        SSE = []
        done1 = False

        for b in range(0, range_size):
            parameters = (self.beta, self.gamma, self.sigma, self.hpA, hp_range[b], self.hcr)
            sse = self.SSE(parameters, initial_state, time, dataset, method='fit_on_cumul_mixt')
            SSE.append(sse)
            if hp_range[b] >= 0.3333 and not done1:
                done1 = True
                print("SSE for sigma = {} = {}".format(hp_range[b], sse))
            if sse < best[0]:
                best = (sse, hp_range[b])
        print("Iterative best fit. Best value for hpB = {} with sse = {}".format(best[1], best[0]))
        # Set the best value of beta:
        self.hpB = best[1]
        # print graph of beta evolution with sse:
        plt.plot(hp_range, SSE, c='blue', label='SSE evolution')
        plt.yscale('log')
        plt.xlabel('hp B value')
        plt.show()

    def fit_beta(self, dataset):
        # Set initial state
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        IA_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        IB_0 = 1
        S_0 = 999999 - H_0 - IA_0 - IB_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, IA_0, IB_0, H_0, R_0)
        time = dataset[:, 0]
        # Optimisation ittérative:
        range_size = 1000
        beta_range = np.linspace(0, 1, range_size)
        best = (math.inf, 0)
        SSE = []
        done1 = False

        for b in range(0, range_size):
            parameters = (beta_range[b], self.gamma, self.sigma, self.hpA, self.hpB, self.hcr)
            sse = self.SSE(parameters, initial_state, time, dataset, method='fit_on_cumul_mixt')
            SSE.append(sse)
            if beta_range[b] >= 0.3333 and not done1:
                done1 = True
                print("SSE for sigma = {} = {}".format(beta_range[b], sse))
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

    def fit_gamma(self, dataset):
        # Set initial state
        H_0 = dataset[0][7]
        E_0 = 3 * dataset[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        IA_0 = dataset[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        IB_0 = 1
        S_0 = 999999 - H_0 - IA_0 - IB_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, IA_0, IB_0, H_0, R_0)
        time = dataset[:, 0]
        # Optimisation ittérative:
        range_size = 1000
        gamma_range = np.linspace(0, 1, range_size)
        best = (math.inf, 0)
        SSE = []
        done1 = False

        for b in range(0, range_size):
            parameters = (self.beta, gamma_range[b], self.sigma, self.hpA, self.hpB, self.hcr)
            sse = self.SSE(parameters, initial_state, time, dataset, method='fit_on_cumul_mixt')
            SSE.append(sse)
            if gamma_range[b] >= 0.3333 and not done1:
                done1 = True
                print("SSE for sigma = {} = {}".format(gamma_range[b], sse))
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

    def SSE(self, parameters, initial_state, time, data, method='fit_on_hospit'):

        # Set parameters:
        params = tuple(parameters)
        # Make predictions:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=params)

        if method == 'fit_on_cumul_positive':

            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][7] - predict[i][2] - predict[i][3] - predict[i][4] - predict[i][5]) ** 2
            return sse

        if method == 'fit_on_hospit_cumul':
            # si hcr est = à 0, on peut fiter la courbe des hospit avec cumul hopit car pas de guérison
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][4] - predict[i][4])**2
            return sse

        if method == 'fit_on_cumul_mixt':
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][7] - predict[i][2] - predict[i][3] - predict[i][4]) ** 2
                sse += (data[i][4] - predict[i][4]) ** 2
            return sse



def plot_predict_and_compare(df, pred, args='predict'):
    if 'predict' in args:
        # just print predicted epidemic curves Warning

        if "no_S" not in args:
            plt.plot(pred[:, 0], pred[:, 1], c='black', label="S")
        plt.plot(pred[:, 0], pred[:, 2], c='yellow', label="E")
        plt.plot(pred[:, 0], pred[:, 3] + pred[:, 4], c='red', label="I")
        plt.plot(pred[:, 0], pred[:, 5], c='purple', label="H")
        plt.plot(pred[:, 0], pred[:, 6], c='blue', label='R')
        plt.show()

    if 'compare' in args:
        plt.scatter(df['Day'], df['cumul_positive'], c='red')
        plt.scatter(df['Day'], df['num_hospitalised'], c='blue')

        cumul_positive = []
        hospit = []
        for i in range(0, len(df['Day'].to_numpy())):
            cumul_positive.append(pred[i][3] + pred[i][4] + pred[i][5] + pred[i][6])  # somme de I, H et R
            hospit.append(pred[i][5])
        plt.plot(df['Day'], cumul_positive, c='red')
        plt.plot(df['Day'], hospit, c='blue')
        if "log" in args:
            plt.yscale('log')
        plt.show()


def sec_method():
    """
    Import the dataset and add informations
    """
    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"

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
    IA_0 = df_np[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
    IB_0 = 1
    S_0 = 999999 - H_0 - IA_0 - IB_0 - E_0
    R_0 = 0

    """ *****************************************************************************
        ETAPE 1:
        Trouver le paramètre hp, de la même façon que dans methode 1
        *****************************************************************************
    """
    # Find the linear relation between the two curves:
    posit = df['cumul_positive'].to_numpy()
    hospit = df['num_cumulative_hospitalizations'].to_numpy()
    factor = []
    for i in range(15, len(posit)):  # Begin after 15 days because stabilisation of the rapport
        factor.append(posit[i] / hospit[i])
    factor = np.array(factor)
    # Predict positive curve from this:
    predict_cumul_positive = df['num_cumulative_hospitalizations'].to_numpy()
    predict_cumul_positive = predict_cumul_positive * np.mean(factor)
    # Set the value to the model.
    model.hpA = (1 / np.mean(factor))
    print('"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""')
    print(" Analyze of the proportion of hospitalized in the positives case")
    print(" Average = {}".format(np.mean(factor)))
    print(" Standard deviation = {}".format(np.std(factor)))
    plt.plot(df['Day'], df['cumul_positive'], c='blue', label='Cumul_positive')
    plt.plot(df['Day'], df['num_cumulative_hospitalizations'], c='red', label='Cumul_hospit')
    plt.plot(df['Day'], predict_cumul_positive, c='green', label='Predicted positives from hospit')
    plt.ylabel('nb people')
    plt.xlabel('time in days')
    plt.title('Compare cumul positive and cumul hospit in dataset')
    plt.show()

    """ *****************************************************************************
        ETAPE 2:
        On fit le modèle:
            - D'abord les paramètres beta, gamma et sigma. Le paramètre hp a été fixé
                à l'étape 1 et le paramètre hcr est fixé à zéro car n'intervient pas
                dans le cas ou l'on fit sur les courbes I+R+H
            - Après on fit hcr sur la courbe des hospitalisations.
        *****************************************************************************
    """
    """    model.fit_scipy(df_np)

    model.fit_sigma(df_np)

    model.fit_hp_b(df_np)

    model.fit_beta(df_np)

    model.fit_gamma(df_np)

    model.fit_sigma(df_np)

    model.fit_hp_b(df_np)
    """
    model.brute_force(df_np)


    """ *****************************************************************************
        Nous pouvons maintenant comparer les simulations et les données ainsi que
        dessiner des prédictions à long terme.
        *****************************************************************************
    """
    predictions = model.predict(S_0, E_0, IA_0, IB_0, H_0, R_0, duration=150)
    plot_predict_and_compare(df, predictions, args='predict compare')

    predictions = model.predict(S_0, E_0, IA_0, IB_0, H_0, R_0, duration=50)

    plot_predict_and_compare(df, predictions, args='predict compare no_S log')
    print("final value for model's parameters: ")
    print(" beta = {}".format(model.beta))
    print(" gamma = {}".format(model.gamma))
    print(" sigma = {}".format(model.sigma))
    print(" hpA = {}".format(model.hpA))
    print(' hpB = {}'.format(model.hpB))
    print(" hcr = {}".format(model.hcr))


if __name__ == "__main__":
    # first_method()
    sec_method()
