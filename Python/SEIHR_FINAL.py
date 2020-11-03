import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math

"""
=======================================================================================================
SEIHR MODEL
=======================================================================================================
"""


class SEIR():

    def __init__(self):

        # Model's hyperparameters
        self.beta = None        # Contagion probability
        self.sigma = None       # Probability to go from E to I
        self.gamma = None       # Probability parameter to go from I to R (to be cure)
        self.hp = None          # Probability to go from I to H
        self.hcr = 0            # Hospit Cure Rate

        # Data to fit
        self.dataset = None     # Numpy matrix format
        self.dataframe = None   # Dataframe format

        # Initial state: to be used to make predictions
        self.S_0 = None         # Sensible: peoples who can catch the agent
        self.E_0 = None         # Exposed: people in incubation: Can't spread the agent
        self.I_0 = None         # Infectious: people who can spread the disease
        self.H_0 = None         # Hospitalized peoples: can't spread any more the agent
        self.R_0 = None         # Recovered people: can't catch again the agent due to immunity
        self.N = None           # The total size of the population

        # Data to store
        self.dataJSON = {}
    def saveJson(self):
        with open('Data/SEIR.json', 'w') as outfile:
            json.dump(self.dataJSON, outfile)

    def get_initial_state(self):
        """
        Function who return a tuple with the initial state of the model
        """
        return (self.S_0, self.E_0, self.I_0, self.H_0, self.R_0, self.N)

    def get_parameters(self):
        """
        Function who return a tuple with model's hyper-parameters
        """
        return (self.beta, self.sigma, self.gamma, self.hp, self.hcr)

    def differential(self, state, time, beta, sigma, gamma, hp, hcr):
        """
        Differential equations of the model
        """
        S, E, I, H, R, N = state

        dS = -(beta * S * I) / N
        dE = (beta * S * I) / N - E * sigma
        dI = (E * sigma) - (gamma * I) - (hp * I)
        dH = (hp * I) - (hcr * H)
        dR = (gamma * I) + (hcr * H)
        dN = 0

        return dS, dE, dI, dH, dR, dN

    def predict(self, duration):
        """
        Predict epidemic curves
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

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4])).T

    def fit(self):
        """
        Fitting
        """
        # =========================================================================== #
        # First part: We fit the parameters beta, sigma and gamma by computing the
        # sum of errors between the product of positive tests and the product of the
        # curves I, H and R. In fact, because our parameters hp is fixed at zero,
        # the curve H = 0
        # =========================================================================== #
        # Generate initial state:
        initial_state = self.get_initial_state()
        # Time vector:
        time = self.dataset[:, 0]
        # Bounds: Given ranges for beta, sigma and gamma
        bounds = [(0, 1), (1/5, 1), (1/10, 1/4)]
        # Start values
        start_values = [self.beta, self.sigma, self.gamma]
        # Use Scipy.optimize.minimize with L-BFGS_B method
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, 'first_part'),
                       method='L-BFGS-B', bounds=bounds)

        print(res)
        self.beta = res.x[0]
        self.sigma = res.x[1]
        self.gamma = res.x[2]
        # =========================================================================== #
        # Second part: In our model, infectious people can go from I to H and not
        # only to R. So, we have to split a part of gamma parameter to hp parameter:
        # =========================================================================== #
        self.gamma_hp_slide()

        # =========================================================================== #
        # Third part: We can now fit hcr by computing error between hospitalized
        # in the dataset and our H curve prediction
        # =========================================================================== #
        self.fit_hcr()

    def gamma_hp_slide(self):
        """
        This method find definitive value of gamma and hp by evaluating the part of gamma who is represented by infected
        people who fall in hospital.
        """
        # The number of value to test
        range_size = 1000
        initial_gamma = self.gamma
        # Each value of gamma to test
        gamma_range = np.linspace(0, initial_gamma, range_size)
        # Corresponding value of hp:
        hp_range = np.linspace(0, initial_gamma, range_size)
        hp_range = np.flip(hp_range)

        # Make simulation and compute SSE:
        initial_state = self.get_initial_state()
        time = self.dataset[:, 0]
        SSE = []
        best = (math.inf, 0, 0)
        for g in range(0, range_size):
            params = (self.beta, self.sigma, gamma_range[g], hp_range[g], 0)
            # Compute the SSE for theses params on the cumulative hospit. curve (because hcr = 0)
            sse = self.SSE(params, initial_state, time, 'second_part')
            SSE.append(sse)
            if sse < best[0]:
                best = (sse, gamma_range[g], hp_range[g])

        self.gamma = best[1]
        self.hp = best[2]
        print("Gamma/hp slide: best value with sse = {}".format(best[0]))
        print("best Gamma = {}".format(self.gamma))
        print("best hp = {}".format(self.hp))

        #plot :
        plt.plot(gamma_range, SSE, c='blue', label='Gamma value')
        #plt.yscale('log')
        plt.xlabel('gamma prooportion')
        plt.show()

    def fit_hcr(self):
        """
        This method find the value of hcr, with a precision of 1/1000 by enumuerating 1000 values of hcr between 0 and 1
        and chosing the value with the best sum of square error like describe in the SSE method.
        """
        # The number of value to test
        range_size = 1000
        # Each value of gamma to test
        hcr_range = np.linspace(0, 1, range_size)

        # Make simulation and compute SSE:
        initial_state = self.get_initial_state()
        time = self.dataset[:, 0]
        SSE = []
        best = (math.inf, 0)
        for g in range(0, range_size):
            params = (self.beta, self.sigma, self.gamma, self.hp, hcr_range[g])
            # Compute the SSE for theses params on the normal hospit. curve
            sse = self.SSE(params, initial_state, time, 'third_part')
            SSE.append(sse)
            if sse < best[0]:
                best = (sse, hcr_range[g])

        self.hcr = best[1]
        print("Best value of hcr with sse = {}".format(best[0]))
        print("hcr = {}".format(self.hcr))


        #plot :
        plt.plot(hcr_range, SSE, c='blue', label='hcr value')
        plt.yscale('log')
        plt.xlabel('hcr value')
        plt.show()

        #Data storing:
        self.dataJSON['fit_hcr'] = []
        for i in range(0,range_size):
            self.dataJSON['fit_hcr'].append({
                "hcr_value": str(hcr_range[i]),
                "log": str(SSE[i]),

            })


    def SSE(self, parameters, initial_state, time, method='fit_on_cumul_positive'):
        """
        Compute and return the sum of square errors between our dataset of observations and
        our predictions. In function of the parameters thant we want to fit, we are using
        different fitting strategies:

        1. First_part:
            This method is use to find the definitive value of beta and sigma, and a temporary value of gamma. In this
            case, hp and hcr are set at Zero and we can fit our parameters by computing the square error between the
            total cumulative positive column of the dataset and the sum of I and R predictions of the model.
            Because we don't care about hospitalized, the actual gamma value takes into account of the hp parameter.
            So the parameters hp and gamma will be separate during the second step
        2. Second_part:
            During thi part we extract the value of hp out of gamma by computing the part of gamma who represent
            peoples who go to hospital. Because hcr parameter is still set on zero, we can find the definitive values
            of hp and gamma by computing square errors between cumulative hospitalized column of the dataset and and
            the prediction of H curve. A hcr parameter set on zero means that H people will never be cure, and H is
            a cumulative hospitalized curve.
        3. Third part:
            The last part compute hcr parameter by computing square error between normal hospitalized column of the
            dataset (non-cumulative), and the curve H.
        """
        if method == 'first_part':
            # Set parameters: we set hp et hcr to zero
            tpl = tuple(parameters)
            params = (tpl[0], tpl[1], tpl[2], 0, 0)

            # Make predictions:
            predict = odeint(func=self.differential,
                             y0=initial_state,
                             t=time,
                             args=params)
            sse = 0.0
            for i in range(0, len(time)):
                sse += (self.dataset[i][7] - predict[i][2] - predict[i][3] - predict[i][4]) ** 2
            return sse

        if method == 'second_part':

            params = tuple(parameters)
            # Make predictions:
            predict = odeint(func=self.differential,
                             y0=initial_state,
                             t=time,
                             args=params)
            sse = 0.0
            for i in range(0, len(time)):
                sse += (self.dataset[i][4] - predict[i][3]) ** 2
            return sse

        if method == 'third_part':

            params = tuple(parameters)
            # Make predictions:
            predict = odeint(func=self.differential,
                             y0=initial_state,
                             t=time,
                             args=params)
            sse = 0.0
            for i in range(0, len(time)):
                sse += (self.dataset[i][3] - predict[i][3]) ** 2
            return sse


    def import_dataset(self, target='covid_20'):

        if target == 'covid_20':
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # Import the dataframe:
            self.dataframe = pd.read_csv(url, sep=',', header=0)

            # Modify the first line to take account of unobserved early infections
            self.dataframe['num_positive'][0] = 11
            self.dataframe['num_tested'][0] = 11

            # Ad a new column at the end with cumulative positive cases at the right
            cumul_positive = self.dataframe['num_positive'].to_numpy()
            for i in range(1, len(cumul_positive)):
                cumul_positive[i] += cumul_positive[i-1]
            self.dataframe.insert(7, 'cumul_positive', cumul_positive)
            # Store a numpy version:
            self.dataset = self.dataframe.to_numpy()

            # Store the initial state who fit with input data
            self.N = 1000000
            self.I_0 = self.dataframe['cumul_positive'][0]
            self.H_0 = self.dataframe['num_hospitalised'][0]
            self.E_0 = 4 * self.dataframe['num_positive'][1]
            self.R_0 = 0
            self.S_0 = self.N - self.I_0 - self.H_0 - self.E_0

            # Initialize default value to hyper-parameters:
            self.beta = 0.35
            self.sigma = 1/3
            self.gamma = 1/7
            self.hp = 0
            self.hcr = 0





    def plot_predict(self, pred, args='no_S'):

        self.dataJSON['predict'] = []
        for i in range(0,len(pred[:, 0])):
            self.dataJSON['predict'].append({
                "predict_day": str(pred[i][0]),
                "predict_S": str(pred[i][1]),
                "predict_E": str(pred[i][2]),
                "predict_I": str(pred[i][3]),
                "predict_H": str(pred[i][4]),
                "predict_R": str(pred[i][5]),

            })

        if 'predict' in args:
            if "no_S" not in args:
                plt.plot(pred[:, 0], pred[:, 1], c='black', label="S")
            plt.plot(pred[:, 0], pred[:, 2], c='yellow', label="E")
            plt.plot(pred[:, 0], pred[:, 3], c='red', label="I")
            plt.plot(pred[:, 0], pred[:, 4], c='purple', label="H")
            plt.plot(pred[:, 0], pred[:, 5], c='blue', label='R')
            plt.show()

        if 'compare' in args:
            plt.scatter(self.dataframe['Day'], self.dataframe['cumul_positive'], c='red')
            plt.scatter(self.dataframe['Day'], self.dataframe['num_hospitalised'], c='blue')

            cumul_positive = []
            hospit = []
            for i in range(0, len(self.dataframe['Day'].to_numpy())):
                cumul_positive.append(pred[i][3] + pred[i][4] + pred[i][5])  # somme de I, H et R
                hospit.append(pred[i][4])
            plt.plot(self.dataframe['Day'], cumul_positive, c='red')
            plt.plot(self.dataframe['Day'], hospit, c='blue')
            if "log" in args:
                plt.yscale('log')
            plt.show()




def first_method():

    # Initialize the model
    model = SEIR()

    # Import the dataset:
    model.import_dataset(target='covid_20')

    # Fit the model:
    model.fit()

    # Make predictions and compare with dataset
    predictions = model.predict(50)
    model.plot_predict(predictions, args='compare log')

    # Draw long term curves:
    predictions = model.predict(365)
    model.plot_predict(predictions, args='predict')

    print("=======================================================")
    print("Final value of each model parameters: ")
    print("Beta = {}".format(model.beta))
    print("sigma = {}".format(model.sigma))
    print("gamma = {}".format(model.gamma))
    print("hp = {}".format(model.hp))
    print("hcr = {}".format(model.hcr))
    print("=======================================================")
    print("This is a small step for man, but a giant step for COV-Invaders")

    model.saveJson()






if __name__ == "__main__":

    first_method()
