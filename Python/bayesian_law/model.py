import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from scipy.integrate import odeint   # To integrate our equation
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import uncertainpy as un
import chaospy as cp                 # To create distributions
import json
import math
import random
from smooth import own_NRMAS_index, own_NRMAS
from plot import plot_current_data
from plot import preporcessing
from plot import plot_cumul_positif_comp
from plot import plot_cumul_hospit_comp
from plot import plot_non_cum_hospit_comp
from plot import plot_pcr_pd_slide
from plot import plot_hcr_fitting
from plot import plot_gamma_hp_slide
from plot import plot_critical_com
from plot import plot_fatal_com
from uncertainty import add_uncertainty

from predict import __predict__
from plot import __plot_predict__


class seir():

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
        # self.ran = random.uniform(0.5, 1)
        self.ran = np.random.normal(0.75, (0.25**(1/2)))

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
        self.window = 7
        self.dataJSON = {}

    def saveJson(self):
        with open('Data/SEIR+.json', 'w') as outfile:
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

    def SSE(self, parameters, initial_state, time, method='fit_on_cumul_positive'):
        """
        Compute and return the sum of square errors between our dataset of observations and
        our predictions. In function of the parameters thant we want to fit, we are using
        different fitting strategies:

        1. PART 1:
            This method is use to find the definitive value of beta and sigma, and a temporary value of gamma. In this
            case, hp and hcr are set at Zero and we can fit our parameters by computing the square error between the
            total cumulative positive column of the dataset and the sum of I and R predictions of the model.
            Because we don't care about hospitalized, the actual gamma value takes into account of the hp parameter.
            So the parameters hp and gamma will be separate during the second step
        2. PART 2:
            During this part we extract the value of hp out of gamma by computing the part of gamma who represent
            peoples who go to hospital. Because hcr parameter is still set on zero, we can find the definitive values
            of hp and gamma by computing square errors between cumulative hospitalized column of the dataset and and
            the prediction of H curve. A hcr parameter set on zero means that H people will never be cure, and H is
            a cumulative hospitalized curve.
        3. PART 3:
            This part compute hcr parameter by computing square error between normal hospitalized column of the
            dataset (non-cumulative), and the curve H.
        4. PART 4:

        5. PART 5:
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
                sse += (self.dataset[i][12] - predict[i][2] - predict[i][3] - predict[i][4]) ** 2
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
                sse += (self.dataset[i][6] - predict[i][3]) ** 2
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
                sse += (self.dataset[i][5] - predict[i][3]) ** 2
            return sse

        if method == 'part_four':

            tpl = tuple(parameters)
            params = (self.beta, self.sigma, self.gamma, self.hp,
                      tpl[0] * self.hcr, (1 - tpl[0]) * self.pc, 0, tpl[1])
            print(params)
            # Make predictions:
            predict = odeint(func=self.differential,
                             y0=initial_state,
                             t=time,
                             args=params)
            sse = 0.0
            for i in range(0, len(time)):
                # fit on ICU
                sse += (self.dataset[i][7] - predict[i][6]) ** 2

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
                sse += (self.dataset[i][8] - predict[i][7]) ** 2
            return sse


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
        proportion_range = np.linspace(0, 1, range_size)
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

        # plot :
        plot_gamma_hp_slide(self, proportion_range, SSE)

    def pcr_pd_slide(self):
        """
        This method find definitive value of pcr and pd
        """
        # The number of value to test
        range_size = 1000
        initial_pcr = self.pcr
        # Each value of gamma to test
        pcr_range = np.linspace(0, initial_pcr, range_size)
        proportion_range = np.linspace(0, 1, range_size)
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

        # plot :
        plot_pcr_pd_slide(self,proportion_range,SSE)

        print("pd/pcr slide: best value with sse = {}".format(best[0]))
        print("best pcr = {}".format(self.pcr))
        print("best pd = {}".format(self.pd))

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

        # plot :
        plot_hcr_fitting(self,hcr_range,SSE)
        # Data storing:
        self.dataJSON['fit_hcr'] = []
        for i in range(0, range_size):
            self.dataJSON['fit_hcr'].append({
                "hcr_value": str(hcr_range[i]),
                "log": str(SSE[i]),

            })

    def fit(self):
        """
        This method use the given data to find values of our model who minimise square error
        between predictions and original data.
        """
        # =========================================================================== #
        # PART 1: We fit the parameters beta, sigma and a temp version of gamma by computing the
        # sum of errors between the daily cumulative of positive tests and the
        # product of the I, H and R curves. All others parameters are set on zero.
        # So, in this first part, we are equivalent to a basic SEIR model
        # =========================================================================== #
        # Generate initial state:
        initial_state = self.get_initial_state()
        # Time vector:
        time = self.dataset[:, 0]
        # Bounds: Given ranges for beta, sigma and gamma
        bounds = [(0, 1), (1 / 5, 1), (1 / 10, 1 / 4)]
        # Start values
        start_values = [self.beta, self.sigma, self.gamma]
        # Use Scipy.optimize.minimize with L-BFGS_B method
        res = minimize(self.SSE
                        ,np.asarray(start_values), args=(initial_state, time, 'first_part'),
                       method='L-BFGS-B', bounds=bounds)

        print(res)
        self.beta = res.x[0]
        self.sigma = res.x[1]
        self.gamma = res.x[2]

        # Compare data with predictions on cumulative positive
        predictions = __predict__(self,self.dataset.shape[0])
        # Store I + R
        cumul_positive = predictions[:, 3] + predictions[:, 5]

        plot_cumul_positif_comp(self,cumul_positive,predictions)


        # =========================================================================== #
        # PART 2: In our model, infectious people can go from I to H and not
        # only to R. So, we have to split a part of gamma parameter to hp parameter:
        # =========================================================================== #
        self.gamma_hp_slide()

        # Compare data with predictions on cumulative hospit
        predictions = __predict__(self,self.dataset.shape[0])
        # Store H
        cumul_hospit = predictions[:, 4]

        plot_cumul_hospit_comp(self,cumul_hospit,predictions)

        # =========================================================================== #
        # PART 3: compute the probability to out of H.
        # WARNING: this probability contain the probability to be cure and the
        # probability to fall in ICU
        # =========================================================================== #
        self.fit_hcr()

        # Compare data with hospit data and non cumulative h curve
        predictions = __predict__(self,self.dataset.shape[0])
        # Store H
        hospit = predictions[:, 4]
        plot_non_cum_hospit_comp(self,hospit,predictions)


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
        start_values = [0.7, 0.1]
        # Use Scipy.optimize.minimize with L-BFGS_B method
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, 'part_four'),
                       method='L-BFGS-B', bounds=bounds)
        print(res)
        initial_hcr = self.hcr
        self.hcr = res.x[0] * initial_hcr
        self.pc = (1 - res.x[0]) * initial_hcr
        self.pcr = res.x[1]

        # Compare data with critical
        predictions = __predict__(self,self.dataset.shape[0])
        # Store C
        critical = predictions[:, 7]

        plot_critical_com(self,critical,predictions)


        # =========================================================================== #
        # PART 5: We can now slide the actual value of pcr to spare the probability
        # to be cure from C (definitive pcr) and to die form C (pd)
        # =========================================================================== #
        self.pcr_pd_slide()

        # Compare data with critical
        predictions = __predict__(self,self.dataset.shape[0])
        # Store C
        fatalities = predictions[:, 8]

        plot_fatal_com(self,fatalities,predictions)

    def predict(self,duration):
        return __predict__(self,duration)

    def plot_predict(self,pred, args='no_S'):
        __plot_predict__(self, pred, args='no_S')


    def print_final_value(self):
        print("=======================================================")
        print("Final value of each model parameters: ")
        print("Beta = {}".format(self.beta))
        print("sigma = {}".format(self.sigma))
        print("gamma = {}".format(self.gamma))
        print("hp = {}".format(self.hp))
        print("hcr = {}".format(self.hcr))
        print('pc = {}'.format(self.hp))
        print("pcr = {}".format(self.pcr))
        print("pd = {}".format(self.pd))
        print("=======================================================")
        print("This is a small step for man, but a giant step for COV-Invaders")

    def import_dataset(self, target='covid_20'):

        if target == 'covid_20':
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # Import the dataframe:
            self.raw_dataset = pd.read_csv(url, sep=',', header=0)

            self.dataframe = add_uncertainty(self,self.raw_dataset)
            # Need to add smoth in data
            plot_current_data(self)


            #smooth
            own_NRMAS(self.dataframe['num_hospitalised'],7)
            own_NRMAS(self.dataframe['num_cumulative_hospitalizations'],7)
            own_NRMAS(self.dataframe['num_critical'],7)
            own_NRMAS(self.dataframe['num_fatalities'],7)

            self.raw_dataset.insert(2, 'num_positive_lower', self.dataframe['num_positive_lower'].to_numpy())
            self.raw_dataset.insert(3, 'num_positive_upper', self.dataframe['num_positive_upper'].to_numpy())
            self.raw_dataset.insert(9, 'num_sym_lower', self.dataframe['num_sym_lower'].to_numpy())
            self.raw_dataset.insert(10,'num_sym_upper', self.dataframe['num_sym_upper'].to_numpy())
            self.raw_dataset.insert(11,'num_positive_mean', self.dataframe['num_positive_mean'].to_numpy())

            preporcessing(self)

            # Ad a new column at the end with cumulative positive cases at the right
            cumul_positive = self.dataframe['num_positive'].to_numpy()
            cumul_positive_non_smooth = self.raw_dataset['num_positive']
            for i in range(1, len(cumul_positive)):
                cumul_positive[i] += cumul_positive[i - 1]
                cumul_positive_non_smooth[i] += cumul_positive_non_smooth[i - 1]
            self.dataframe.insert(12, 'cumul_positive', cumul_positive)
            self.raw_dataset.insert(12, 'cumul_positive', cumul_positive_non_smooth)

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
            self.I_0 = self.dataset[0][12]
            self.H_0 = self.dataset[0][5]
            self.E_0 = 3 * self.dataset[1][1]  # Because mean of incubation period = 3 days
            self.R_0 = 0
            self.C_0 = 0
            self.D_0 = 0
            self.S_0 = self.N - self.I_0 - self.H_0 - self.E_0

            # Initialize default value to hyper-parameters:
            self.beta = 0.35
            self.sigma = 1 / 3
            self.gamma = 1 / 7
            self.hp = 0
            self.hcr = 0
