
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class PolynomialRegressor():
    """
    This class implement a polynomial regressor
    """

    def __init__(self, m):

        # Degree of the polynom:
        self.m = m
        # Create the regressor
        self.regressor = LinearRegression()

    def fit(self, x, y):
        """
        Fit the model
        :param x: Input vector
        :param y: Output vector
        """
        # Transform the dataset:
        data = np.zeros((len(x), self.m+1))
        for i in range(0, len(x)):
            data[i][0] = 1
            for m in range(1, self.m+1):
                data[i][m] = x[i] ** m
        # Fit the model:
        self.regressor.fit(data, y)

    def predict(self, x):
        """
        Make predictions from an input vector
        :param x: input vector
        :return: a vector of predictions
        """
        # Transform the dataset:
        data = np.zeros((len(x), self.m + 1))
        for i in range(0, len(x)):
            data[i][0] = 1
            for m in range(1, self.m + 1):
                data[i][m] = x[i] ** m
        # Make predictions:
        pred = self.regressor.predict(data)
        return pred

def normal_density(sigma_sq, dx):
    """
    Compute the probability density funcion of a Normal distribution centered in zero
    :param sigma_sq: the variance of the distribution
    :param dx: the distance of the evidence
    :return:
    """
    return (np.exp(((dx ** 2) / sigma_sq) * (-0.5)) / np.sqrt(sigma_sq * 2 * np.pi))

def initial_infected_estimator(dataset):
    """
    This function estimate the initial number of infected by analyzing the mean ratio
    between the total cumulative tests and the cumulative hospitalization
    :param dataset:
    :return:
    """
    # Compute cumul positive and cumul hospit:
    cumul_positif = [dataset[0][1]]
    positifs = [dataset[0][1]]
    cumul_hospit = [dataset[0][4]]
    time = [0]
    for i in range(1, dataset.shape[0]):
        cumul_positif.append(dataset[i][1] + dataset[i-1][1])
        cumul_hospit.append(dataset[i][4])
        positifs.append(dataset[i][1])
        time.append(i)


    # Make a polynomial regression of infected:
    regressor = PolynomialRegressor(m=5)
    time_bis = []
    idx = -10
    while(idx < 50):
        time_bis.append(idx)
        idx += 1
    regressor.fit(time, positifs)
    pred = regressor.predict(time)
    pred_bis = regressor.predict(time_bis)


    # Plot the result:
    plt.scatter(dataset[:, 0], dataset[:, 1], c='black', label='Test data')
    plt.plot(time, pred, c='red', label='polynomial regression')
    plt.plot(time_bis, pred_bis, c='red')
    plt.legend()
    plt.show()

    # show result before t_0:
    for i in range(0, 20):
        print('Time index: {}, prediction: {}'.format(time_bis[i], pred_bis[i]))

    """
    Cet outil nous permet de constater que nous avons 4 personnes déjà infectées en t_0
    """




