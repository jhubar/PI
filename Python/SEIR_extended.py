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
        self.gamma = 1/7        # 7 days average to be cure
        self.sigma = 1/3        # 3 days average incubation time
        self.hp = 1/20.89          # Hospit probability Proportion of infected people who begin hospitalized
        self.hcr = 0         # Hospit Cure Rate: proba de guérir en hospitalisation

    def set_hospit_prop(self, hospit_prop):
        self.hospit_prop = hospit_prop

    def differential(self, state, time, beta, gamma, sigma, hp, hcr):
        """
        Differential equations of the model
        """
        S, E, I, H, R = state

        dS = -(beta * S * I) / (S + I + R)
        dE = (beta * S * I) / (S + I + R) - E * sigma
        dI = (1 - hp) * (E * sigma) - (gamma * I)   # Note: on retire la proportion des hospitalisés, ils ne participent plus à la contagion
        dH = hp * (E * sigma) - hcr * H
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

    def fit(self, dataset):

        # Set initial state:
        H_0 = df_np[0][7]
        E_0 = 3 * df_np[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        I_0 = df_np[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
        S_0 = 999999 - H_0 - I_0 - E_0
        R_0 = 0
        initial_state = (S_0, E_0, I_0, H_0, R_0)
        time = dataset[:, 0]
        # Optimisation:
        method = 'fit_on_hospit'
        start_values = [self.beta, self.gamma, self.sigma, self.hp, self.hcr]
        bounds = [(0, 1), (1/7, 1/7), (1/3, 1/3), (1/20.89, 1/20.89), (0, 1)]
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, dataset, method), method='L-BFGS-B', bounds=bounds)
        print(res)
        # Set new parameters:
        self.beta = res.x[0]
        self.hcr = res.x[4]

    def fit_beta(self, dataset):
        # Set initial state:
        H_0 = df_np[0][7]
        E_0 = 3 * df_np[1][1]  # Vu qu'un tiers de ce nombre devront être positifs à t+1
        I_0 = df_np[0][1] - H_0  # Les hospitalisés ne participent plus à la contagion
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

    def fit_hcr(self, dataset):
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
        hcr_range = np.linspace(0, 1, range_size)
        best = (math.inf, 0)
        SSE = []
        for b in range(0, range_size):
            parameters = (self.beta, self.gamma, self.sigma, self.hp, hcr_range[b])
            sse = self.SSE(parameters, initial_state, time, dataset, method='fit_on_hospit')
            SSE.append(sse)
            if sse < best[0]:
                best = (sse, hcr_range[b])
        print("Iterative best fit. Best value for hcr = {} with sse = {}".format(best[1], best[0]))
        # Set the best value of beta:
        self.hcr = best[1]
        # print graph of beta evolution with sse:
        plt.plot(hcr_range, SSE, c='blue', label='SSE evolution')
        plt.yscale('log')
        plt.xlabel('hcr value')
        plt.show()

    def fit_scipy(self, dataset):
        """
        On commence par fitter les paramètres beta (dans un interval de 0 à 1, pas d'indications pour lui),
        gamma dans un interval de 1/10 à 1/4 donné dans l'énnoncé, et sigma dans un interval de 1/5 à 1
        (également donné).
        Nous fittons les données de testage cummulées sur la somme des courbes I, H et R.

        """
        time = dataset[:, 0]
        # Set initial state:
        initial_state = (1000000 - dataset[0][1] - 3*dataset[2][1], 3 * dataset[2][1], dataset[0][1], dataset[0][4], 0)
        start_values = [self.beta, self.gamma, self.sigma, self.hp, self.hcr]
        bounds = [(0, 1), (1/10, 1/4), (1/5, 1), (self.hp, self.hp), (self.hcr, self.hcr)]
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, dataset, 'fit_on_cumul_positive'),
                       method='L-BFGS-B', bounds=bounds)
        print(res)
        self.beta = res.x[0]
        self.gamma = res.x[1]
        self.sigma = res.x[2]
        """
        Deuxième étape, il ne reste plus qu'à fitter hcr sur la courbe des hospitalisations (non cumulées)
        NOTE : Je ne sais pas pourquoi, mais ça converge pas quand j'utilise minimize alors que ça converge très bien
        en itérant toutes les valeurs
        """
        self.fit_hcr(dataset)


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
                sse += (data[i][3] - predict[i][3])**2
            return sse
        if method == 'fit_on_hospit_cumul':
            # si hcr est = à 0, on peut fiter la courbe des hospit avec cumul hopit car pas de guérison
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][4] - predict[i][3])**2
            return sse
        if method == 'fit_on_cumul_positive':

            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i][7] - predict[i][2] - predict[i][3] - predict[i][4])**2
            return sse






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
        Paramètres donnés par le prof: 
            - Gamma: 1/7
            - Sigma: 1/3

        Première étape = déterminer la relation entre les hospitalisations et les infectés
        Pour ça on analyse la proportion moyenne entre la courbe cumulée des tests positifs
        et la courbe cumulée des hospitalisés. 
        On remarque que à partir de j 15, ce rapport est constant et que la courbe des positifs est 
        20.896229143831444 * celle des hospitalisés. (Standard deviation = 0.9312720467999785)
        On a donc notre pramètre  hp. 
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
    model.set_hospit_prop(1 / np.mean(factor))

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
        Maintenant que nous savons la proportion de contaminations qui finissent
        hospitalisées (et donc qui ne participent plus à la contamination, nous pouvons
        fiter le paramètre Beta. 
        Nous pouvons le fitter sur le cumul des tests positifs à partir de la somme des 
        courbes I, H, et R. On peut le faire avec une valeur de hcr = 0 (pas de guérison
        pour les hospitalisés, car sa valeur n'entre pas en compte ici.
        *****************************************************************************
    """
    model.fit_beta(df_np)

    """ *****************************************************************************
        ETAPE 3:
        Il ne reste plus qu'à fiter le parametre hcr, qui est le taux de guérison
        chez les hospitalisés
        *****************************************************************************
    """
    model.fit_hcr(df_np)

    """ *****************************************************************************
        Nous pouvons maintenant comparer les simulations et les données ainsi que 
        dessiner des prédictions à long terme. 
        *****************************************************************************
    """
    predictions = model.predict(S_0, E_0, I_0, H_0, R_0, duration=150)
    plot_predict_and_compare(df, predictions, args='predict compare')

    predictions = model.predict(S_0, E_0, I_0, H_0, R_0, duration=50)

    plot_predict_and_compare(df, predictions, args='predict compare no_S log')


def sec_method():

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
    model.set_hospit_prop(1 / np.mean(factor))
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
    model.fit_scipy(df_np)

    """ *****************************************************************************
        Nous pouvons maintenant comparer les simulations et les données ainsi que 
        dessiner des prédictions à long terme. 
        *****************************************************************************
    """
    predictions = model.predict(S_0, E_0, I_0, H_0, R_0, duration=150)
    plot_predict_and_compare(df, predictions, args='predict compare')

    predictions = model.predict(S_0, E_0, I_0, H_0, R_0, duration=50)

    plot_predict_and_compare(df, predictions, args='predict compare no_S log')

if __name__ == "__main__":

    sec_method()


