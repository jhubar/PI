import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math

"""
=======================================================================================================
SEIHR +  MODEL
=======================================================================================================
"""


class SEIR():

    def __init__(self):

        # Model's hyperparameters
        self.beta = 0        # Contagion probability
        self.sigma = 0       # Probability to go from E to I
        self.gamma = 0       # Probability parameter to go from I to R (to be cure)
        self.hp = 0             # Probability to go from I to H
        self.hcr = 0            # Hospit Cure Rate
        self.pc = 0             # Probability to fall in ICU each day from H
        self.pd = 0             # Probability to die each day in icu
        self.pcr = 0            # Probability to recover from critical

        # Data to fit
        self.raw_dataset = None # Original dataset, before preprocessing
        self.dataset = None     # Numpy matrix format
        self.dataframe = None   # Dataframe format

        # Initial state: to be used to make predictions
        self.S_0 = None         # Sensible: peoples who can catch the agent
        self.E_0 = None         # Exposed: people in incubation: Can't spread the agent
        self.I_0 = None         # Infectious: people who can spread the disease
        self.H_0 = None         # Hospitalized peoples: can't spread any more the agent
        self.C_0 = None         # Critical: peoples who are in ICU
        self.R_0 = None         # Recovered people: can't catch again the agent due to immunity
        self.D_0 = None         # Dead: people who die in ICU
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
        return (self.S_0, self.E_0, self.I_0, self.H_0, self.R_0, self.N, self.C_0, self.D_0)

    def get_parameters(self):
        """
        Function who return a tuple with model's hyper-parameters
        """
        return (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr)

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr):
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

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4],
                          predict[:, 5], predict[:, 6], predict[:, 7])).T

    def fit(self):
        """
        Fitting
        """
        # =========================================================================== #
        # PART 1: We fit the parameters beta, sigma and gamma by computing the
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
        # PART 2: In our model, infectious people can go from I to H and not
        # only to R. So, we have to split a part of gamma parameter to hp parameter:
        # =========================================================================== #
        self.gamma_hp_slide()

        # =========================================================================== #
        # PART 3: compute the probability to out of H.
        # WARNING: this probability contain the probability to be cure and the
        # probability to fall in ICU
        # =========================================================================== #
        self.fit_hcr()

        # =========================================================================== #
        # PART 4: People in H state can not only being cured. so we will distribute
        # the actual value of hcr in hcr (probability to be cured) and pc (probability
        # to fall in Critical cases). Because we don't have cumulative informations
        # about critical cases, we have to fit in the same time a probability pcr who
        # represent the probability to leave the critical state. So, we are optimizing
        # 1. the ratio of actual hcr who begin pc
        # 2. the value of pcr
        # =========================================================================== #
        # Generate initial state:
        initial_state = self.get_initial_state()
        # Time vector:
        time = self.dataset[:, 0]
        # Bounds: hcr/pc ratio, pcr
        bounds = [(0, 1), (0, 1)]
        # Start values
        start_values = [0.5, 0.5]
        # Use Scipy.optimize.minimize with L-BFGS_B method
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, 'part_four'),
                       method='L-BFGS-B', bounds=bounds)
        print(res)
        initial_hcr = self.hcr
        self.hcr = res.x[0] * initial_hcr
        self.pc = (1 - res.x[0]) * initial_hcr
        self.pcr = res.x[1]

        # =========================================================================== #
        # PART 5: We can now slide the actual value of pcr to spare the probability
        # to be cure from C (definitive pcr) and to die form C (pd)
        # =========================================================================== #
        self.pcr_pd_slide()

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
            params = (self.beta, self.sigma, gamma_range[g], hp_range[g], 0, 0, 0, 0)
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
        plt.yscale('log')
        plt.xlabel('gamma prooportion')
        plt.show()

    def pcr_pd_slide(self):
        """
        This method find definitive value of pcr and pd
        """
        # The number of value to test
        range_size = 1000
        initial_pcr = self.gamma
        # Each value of gamma to test
        pcr_range = np.linspace(0, initial_pcr, range_size)
        # Corresponding value of hp:
        pd_range = np.linspace(0, initial_pcr, range_size)
        pd_range = np.flip(pd_range)

        # Make simulation and compute SSE:
        initial_state = self.get_initial_state()
        time = self.dataset[:, 0]
        SSE = []
        best = (math.inf, 0, 0)
        for g in range(0, range_size):
            params = (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, pd_range[g], pcr_range[g])
            # Compute the SSE for theses params on the cumulative hospit. curve (because hcr = 0)
            sse = self.SSE(params, initial_state, time, 'part_5')
            SSE.append(sse)
            if sse < best[0]:
                best = (sse, pd_range[g], pcr_range[g])

        self.pcr = best[2]
        self.pd = best[1]
        print("pd/pcr slide: best value with sse = {}".format(best[0]))
        print("best pcr = {}".format(self.pcr))
        print("best pd = {}".format(self.pd))

        #plot :
        plt.plot(pcr_range, SSE, c='blue', label='pcr value')
        plt.yscale('log')
        plt.xlabel('pcr prooportion')
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
            params = (self.beta, self.sigma, self.gamma, self.hp, hcr_range[g], 0, 0, 0)
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
            params = (tpl[0], tpl[1], tpl[2], 0, 0, 0, 0, 0)

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
                # Error on non-cumulative hospit
                sse += (self.dataset[i][3] - predict[i][3]) ** 2
            return sse

        if method == 'part_four':

            tpl = tuple(parameters)
            params = (self.beta, self.sigma, self.gamma, self.hp,
                      tpl[0] * self.hcr, (1 - tpl[0]) * self.pc, 0, tpl[1])
            # Make predictions:
            predict = odeint(func=self.differential,
                             y0=initial_state,
                             t=time,
                             args=params)
            sse = 0.0
            for i in range(0, len(time)):
                # fit on ICU
                sse += (self.dataset[i][5] - predict[i][6]) ** 2
            return sse

        if method == 'part_5':

            params = tuple(parameters)
            # Make predictions:
            predict = odeint(func=self.differential,
                             y0=initial_state,
                             t=time,
                             args=params)
            sse = 0.0
            for i in range(0, len(time)):
                # Error on death cases
                sse += (self.dataset[i][6] - predict[i][7]) ** 2
            return sse

    def dataframe_smoothing(self, df):

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
        end_point = smt_prd * nb_per
        # Perform smoothing for each attributes
        for i in range(1, np_df.shape[1]):
            smoothed[:, i] = np.convolve(np_df[:, i], smt_vec, mode="same")
            # Don't smooth the last week
            for j in range(smoothed.shape[0] - smt_prd, smoothed.shape[0]):
                smoothed[j][i] = np_df[j][i]

        # Write new values in a dataframe
        new_df = pd.DataFrame(smoothed, columns=df.columns)


        return new_df


    def import_dataset(self, target='covid_20'):

        if target == 'covid_20':
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # Import the dataframe:
            self.raw_dataset = pd.read_csv(url, sep=',', header=0)

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
            plt.show()

            # The same for hospitalisation data:
            y_raw = self.raw_dataset['num_hospitalised']
            y_smooth = self.dataframe['num_hospitalised']
            plt.scatter(x_axe, y_raw, c='blue', label='original hospitalised')
            plt.plot(x_axe, y_smooth, c='red', label="smoothed hospitalised")
            plt.show()

            # Ad a new column at the end with cumulative positive cases at the right
            cumul_positive = self.dataframe['num_positive'].to_numpy()
            for i in range(1, len(cumul_positive)):
                cumul_positive[i] += cumul_positive[i-1]
            self.dataframe.insert(7, 'cumul_positive', cumul_positive)


            # Delete the first line with zero test
            for i in range(0, 1):
                self.raw_dataset.drop(axis=0, index=i, inplace=True)
                self.dataframe.drop(axis=0, index=i, inplace=True)
            # To reset dataframe index:
            tmp = self.raw_dataset.to_numpy()
            self.raw_dataset = pd.DataFrame(tmp, columns=self.raw_dataset.columns)
            tmp = self.dataframe.to_numpy()
            self.dataframe = pd.DataFrame(tmp, columns=self.dataframe.columns)

            # Store a numpy version:
            self.dataset = self.dataframe.to_numpy()

            # Store the initial state who fit with input data
            self.N = 1000000
            self.I_0 = self.dataset[0][7]
            self.H_0 = self.dataset[0][3]
            self.E_0 = 3 * self.dataset[1][1]    # Because mean of incubation period = 3 days
            self.R_0 = 0
            self.C_0 = 0
            self.D_0 = 0
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
            plt.show()

        if 'compare' in args:
            plt.scatter(self.dataframe['Day'], self.dataframe['cumul_positive'], c='red')
            if self.pc == 0 and self.hcr == 0:
                plt.scatter(self.dataframe['Day'], self.dataframe['num_cumulative_hospitalizations'], c='blue', label='cumul_hosp')
            else:
                plt.scatter(self.dataframe['Day'], self.dataframe['num_hospitalised'], c='blue', label='hosp')

            cumul_positive = []
            hospit = []
            for i in range(0, len(self.dataframe['Day'].to_numpy())):
                cumul_positive.append(pred[i][3] + pred[i][4] + pred[i][5] + pred[i][7] + pred[i][8])  # sum of I, H, R, C and D
                hospit.append(pred[i][4])

            print(len(self.dataframe['Day']))
            self.dataJSON['log'] = []
            for i in range(0,len(self.dataframe['Day'])):
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
    model.plot_predict(predictions, args='compare')
    model.plot_predict(predictions, args='hospit')

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
    print('pc = {}'.format(model.hp))
    print("pcr = {}".format(model.pcr))
    print("pd = {}".format(model.pd))
    print("=======================================================")
    print("This is a small step for man, but a giant step for COV-Invaders")

    model.saveJson()






if __name__ == "__main__":

    first_method()
