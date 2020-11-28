import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from scipy.stats import binom as binom
import tools

from smoothing import dataframe_smoothing

class SEIR():

    def __init__(self):

        # ========================================== #
        #       Epidemic's model parameters
        # ========================================== #
        self.beta = 0.4         # Contamination rate
        self.sigma = 0.9        # Incubation rate
        self.gamma = 0.1        # Recovery rate
        self.hp = 0.05          # Hospit rate
        self.hcr = 0.2          # Hospit recovery rate
        self.pc = 0.1           # Critical rate
        self.pd = 0.1           # Critical recovery rate
        self.pcr = 0.3          # Critical mortality

        # ========================================== #
        #       Testing protocol parameters
        # ========================================== #
        self.s = 0.77           # Sensitivity
        self.t = 0.7            # Testing rate in symptomatical

        # Learning set
        self.dataframe = None
        self.dataset = None

        # ========================================== #
        #        Hyperparameters dashboard:
        # ========================================== #

        # Weights of each probability in the objective function
        self.w_1 = 1        # Weight of the daily test number
        self.w_2 = 1        # Weight of the daily number of positive tests
        self.w_3 = 1        # Weight of hospitalized data
        self.w_4 = 1        # Weight of critical data
        self.w_5 = 1        # Weight of fatalities

        # Size the variance use fore the normal distribution
        self.var_w_1 = 3
        self.var_w_2 = 3
        self.var_w_3 = 3
        self.var_w_4 = 3
        self.var_w_5 = 3

        # Optimizer constraints
        self.beta_min = 0.01
        self.beta_max = 1
        self.sigma_min = 1/5
        self.sigma_max = 1
        self.gamma_min = 1/10
        self.gamma_max = 1/4
        self.hp_min = 0.001
        self.hp_max = 1
        self.hcr_min = 0.001
        self.hcr_max = 1
        self.pc_min = 0.001
        self.pc_max = 1
        self.pd_min = 0.001
        self.pd_max = 1
        self.pcr_min = 0.001
        self.pcr_max = 1
        self.s_min = 0.70
        self.s_max = 0.85
        self.t_min = 0.5
        self.t_max = 1

        # Size of the population:
        self.N = 1000000

        # Estimation of the number of infected at t_0
        self.I_0 = 20

        # Smoothing the dataset?
        self.smoothing = False

        # Optimizer hyperparameter: LBFGSB or COBYLA
        self.optimizer = 'LBFGSB'
        # Step_size, only for LBFGSB
        self.step_size = None

        # ========================================== #
        #                   Printers
        # ========================================== #

        # Display fit details
        self.fit_display = True
        # Basis informations about objective function:
        self.basis_obj_display = True
        self.full_obj_display = False


    def get_parameters(self):

        prm = (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr, self.s, self.t)
        return prm

    def get_hyperparameters(self):

        hprm = (self.w_1, self.w_2, self.w_3, self.w_4, self.w_5,
                self.var_w_1, self.var_w_2, self.var_w_3, self.var_w_4, self.var_w_5,
                self.smoothing, self.optimizer, self.step_size,
                self. I_0)
        return hprm

    def get_initial_state(self, sensib=None, test_rate=None, sigma=None):
        """
        Generate an initial state for the model from the dataset
        according to the sensitivity and the testing rate to
        estimate the true value of the initial state
        :param sensib: Sensibility value to use. Use class value if None
        :param test_rate: Testing rate value to use. Use class value if None
        :return: An array
        NOTES:
        I_0 value estimated by the way that the proportion of
        """
        if sensib is None:
            s = self.s
        else:
            s = sensib
        if test_rate is None:
            t = self.t
        else:
            t = test_rate
        if sigma is None:
            sig = self.sigma
        else:
            sig = sigma

        I_0 = self.I_0 / (s * t)
        H_0 = self.dataset[0][3]
        E_0 = self.dataset[1][1] / (sig * s * t)
        D_0 = 0
        C_0 = 0
        S_0 = 1000000 - I_0 - H_0 - E_0
        R_0 = 0
        dE_to_I_0 = self.dataset[0][1] / (s * t)
        dI_to_H_0 = H_0
        dI_to_R_0 = 0
        init = (S_0, E_0, I_0, R_0, H_0, C_0, D_0, dE_to_I_0, dI_to_H_0, dI_to_R_0)
        return init

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr, s, t):
        """
        ODE who describe the evolution of the model with the time
        :param state: An initial state to use
        :param time: A time vector
        :return: the evolution of the number of person in each compartiment + cumulative testing rate
        + cumulative entry in hospital
        """
        S, E, I, R, H, C, D, E_to_I, I_to_H, I_to_R = state

        dS = -(beta * S * I) / self.N
        dE = ((beta * S * I) / self.N) - (sigma * E)
        dI = (sigma * E) - (gamma * I) - (hp * I)
        dH = (hp * I) - (hcr * H) - (pc * H)
        dC = (pc * H) - (pd * C) - (pcr * C)
        dD = (pd * C)
        dR = (gamma * I) + (hcr * H) + (pcr * C)

        dE_to_I = sigma * E
        dI_to_H = hp * I
        dI_to_R = gamma * I

        return dS, dE, dI, dR, dH, dC, dD, dE_to_I, dI_to_H, dI_to_R

    def predict(self, duration, initial_state=None, parameters=None):
        """
        Predict the evolution of the epidemic during the selected duration from a given initial state
        and given parameters
        :param duration: Use positive integer value
        :param initial_state: Default = use self.get_initial_state()
        :param parameters: Default = use self.get_parameters()
        :return: a numpy array of 8 columns and t rows
        """
        # Time vector:
        time = np.arange(duration)
        # Parameters to use
        prm = parameters
        if prm is None:
            prm = self.get_parameters()
        # Initial state to use:
        init = initial_state
        if init is None:
            init = self.get_initial_state(sensib=prm[8], test_rate=prm[9], sigma=prm[1])

        # Make prediction:
        predict = odeint(func=self.differential,
                         y0=init,
                         t=time,
                         args=(tuple(prm)))
        return predict

    def fit(self, method='Normal'):
        """
        Compute best epidemic parameters values according to model's hyperparameters and the dataset
        """

        # Initial values of parameters:
        init_prm = (self.beta, self.sigma, self.gamma, self.hp,
                    self.hcr, self.pc, self.pd, self.pcr, self.s, self.t)
        # Bounds
        bds = [(self.beta_min, self.beta_max), (self.sigma_min, self.sigma_max), (self.gamma_min, self.gamma_max),
               (self.hp_min, self.hp_max), (self.hcr_min, self.hcr_max), (self.pc_min, self.pc_max),
               (self.pd_min, self.pd_max), (self.pcr_min, self.pcr_max),
               (self.s_min, self.s_max), (self.t_min, self.t_max)]
        # Constraint on parameters:
        cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + self.beta_max},
                {'type': 'ineq', 'fun': lambda x: -x[1] + self.sigma_max},
                {'type': 'ineq', 'fun': lambda x: -x[2] + self.gamma_max},
                {'type': 'ineq', 'fun': lambda x: -x[3] + self.hp_max},
                {'type': 'ineq', 'fun': lambda x: -x[4] + self.hcr_max},
                {'type': 'ineq', 'fun': lambda x: -x[5] + self.pc_max},
                {'type': 'ineq', 'fun': lambda x: -x[6] + self.pd_max},
                {'type': 'ineq', 'fun': lambda x: -x[7] + self.pcr_max},
                {'type': 'ineq', 'fun': lambda x: -x[8] + self.s_max},
                {'type': 'ineq', 'fun': lambda x: -x[9] + self.t_max},
                {'type': 'ineq', 'fun': lambda x: x[0] - self.beta_min},
                {'type': 'ineq', 'fun': lambda x: x[1] - self.sigma_min},
                {'type': 'ineq', 'fun': lambda x: x[2] - self.gamma_min},
                {'type': 'ineq', 'fun': lambda x: x[3] - self.hp_min},
                {'type': 'ineq', 'fun': lambda x: x[4] - self.hcr_min},
                {'type': 'ineq', 'fun': lambda x: x[5] - self.pc_min},
                {'type': 'ineq', 'fun': lambda x: x[6] - self.pd_min},
                {'type': 'ineq', 'fun': lambda x: x[7] - self.pcr_min},
                {'type': 'ineq', 'fun': lambda x: x[8] - self.s_min},
                {'type': 'ineq', 'fun': lambda x: x[9] - self.t_min})

        # Optimizer
        res = None
        if self.optimizer == 'LBFGSB':
            res = minimize(self.objective, np.asarray(init_prm),
                           method='L-BFGS-B',
                           #options={'eps': self.step_size},
                           args=(method),
                           bounds=bds)
        else:
            if self.optimizer == 'COBYLA':
                res = minimize(self.objective, np.asarray(init_prm),
                               method='COBYLA',
                               args=(method),
                               constraints=cons)


        if self.fit_display:
            # Print optimizer result
            print(res)

        # Update model parameters:
        self.beta = res.x[0]
        self.sigma = res.x[1]
        self.gamma = res.x[2]
        self.hp = res.x[3]
        self.hcr = res.x[4]
        self.pc = res.x[5]
        self.pd = res.x[6]
        self.pcr = res.x[7]
        self.s = res.x[8]
        self.t = res.x[9]

        if method == 'bruteforce':
            return res.fun

    def objective(self, parameters, method):
        """
        The objective function to minimize during the fitting process.
        These function compute the probability of each observed values accroding to predictions
        take the logarighm value and make the sum.
        """
        # Put parameters into a tuple
        params = tuple(parameters)
        sensitivity = params[8]
        testing_rate = params[9]

        # Get an initial state:
        init_state = self.get_initial_state(sensib=sensitivity, test_rate=testing_rate, sigma=params[1])
        # Make prediction
        predictions = self.predict(duration=self.dataset.shape[0],
                            parameters=params,
                            initial_state=init_state)
        # Time to compare:
        start_t = 3
        end_t = self.dataset.shape[0]

        if self.basis_obj_display:
            print(params)

        #if method == 'bruteforce':
        #    start_t = 7
        #    end_t = 35
        # Uncumul tests predictions:
        infections = [predictions[0][7]]
        for i in range(1, end_t):
            infections.append(predictions[i][7] - predictions[i-1][7])

        # Compare with dataset and compute the likelyhood value
        error = 0.0
        for i in range(start_t, end_t):
            err1 = err2 = err3 = err4 = err5 = 0.0
            # ================================================ #
            # PART 1: Test number fitting
            # ================================================ #
            # the predicted value is multiply by
            pred = infections[i] * testing_rate
            evid = self.dataset[i][2]
            # Standardize and use normal distribution:
            sigma_sq = np.fabs(self.var_w_1 * evid)
            dx = np.fabs(pred - evid)
            if sigma_sq == 0:
                sigma_sq = 1
            prob_1 = tools.normal_density(sigma_sq, dx)
            # Add to the log probability
            if prob_1 < 0.00000000000000000001:
                err1 += 50 * self.w_1
            else:
                err1 -= np.log(prob_1) * self.w_1
            if self.full_obj_display:
                print('iter {} - prb_1 {} - sigma2 {} - dx {} - pred {} - ev {}'.format(i, prob_1, sigma_sq,
                                                                                        dx, pred, evid))
            if np.log(prob_1) * self.w_1 > 0:
                print('iter {} - prb_1 {} - sigma2 {} - dx {} - pred {} - ev {}'.format(i, prob_1, sigma_sq,
                                                                                        dx, pred, evid))

            # ================================================ #
            # PART 2: Fit on the number of positive test
            # ================================================ #
            # the predicted value is multiply by
            pred = infections[i] * testing_rate * sensitivity
            evid = self.dataset[i][1]
            # Standardize and use normal distribution:
            sigma_sq = np.fabs(self.var_w_2 * evid)
            dx = np.fabs(pred - evid)
            if sigma_sq == 0:
                sigma_sq = 1
            prob_2 = tools.normal_density(sigma_sq, dx)
            # Add to the log probability
            if prob_2 < 0.00000000000000000001:
                err2 += 50 * self.w_2
            else:
                err2 -= np.log(prob_2) * self.w_2

            # ================================================ #
            # PART 3: Fit on hospitalized data
            # ================================================ #
            # the predicted value is multiply by
            pred = predictions[i][4]
            evid = self.dataset[i][3]
            # Standardize and use normal distribution:
            sigma_sq = np.fabs(self.var_w_3 * evid)
            dx = np.fabs(pred - evid)
            if sigma_sq == 0:
                sigma_sq = 1
            prob_3 = tools.normal_density(sigma_sq, dx)
            # Add to the log probability
            if prob_3 < 0.00000000000000000001:
                err3 += 50 * self.w_3
            else:
                err3 -= np.log(prob_3) * self.w_3

            # ================================================ #
            # PART 4: Fit on critical
            # ================================================ #
            # the predicted value is multiply by
            pred = predictions[i][5]
            evid = self.dataset[i][5]
            # Standardize and use normal distribution:
            sigma_sq = np.fabs(self.var_w_4 * evid)
            dx = np.fabs(pred - evid)
            if sigma_sq == 0:
                sigma_sq = 1
            prob_4 = tools.normal_density(sigma_sq, dx)
            # Add to the log probability
            if prob_4 < 0.00000000000000000001:
                err4 += 50 * self.w_4
            else:
                err4 -= np.log(prob_4) * self.w_4

            # ================================================ #
            # PART 5: Fit on Fatalities
            # ================================================ #
            # the predicted value is multiply by
            pred = predictions[i][6]
            evid = self.dataset[i][6]
            # Standardize and use normal distribution:
            sigma_sq = np.fabs(self.var_w_5 * evid)
            dx = np.fabs(pred - evid)
            if sigma_sq == 0:
                sigma_sq = 1
            prob_5 = tools.normal_density(sigma_sq, dx)
            # Add to the log probability
            if prob_5 < 0.00000000000000000001:
                err5 += 50 * self.w_5
            else:
                err5 -= np.log(prob_5) * self.w_5

            error += err1 + err2 + err3 + err4 + err5


        return error






    def import_dataset(self):

        url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
        # Import the dataframe:
        raw = pd.read_csv(url, sep=',', header=0)
        raw['num_positive'][0] = 1
        raw['num_tested'][0] = 1
        # Ad a new column at the end with cumulative positive cases at the right
        cumul_positive = np.copy(raw['num_positive'].to_numpy())
        for i in range(1, len(cumul_positive)):
            cumul_positive[i] += cumul_positive[i-1]
        raw.insert(7, 'cumul_positive', cumul_positive)
        if self.smoothing:
            self.dataframe = dataframe_smoothing(raw)
        else: self.dataframe = raw
        self.dataset = self.dataframe.to_numpy()

        self.I_0 = self.dataset[0][1] / (self.s * self.t)
        self.E_0 = self.I_0 * 2
        self.R_0 = 0
        self.S_0 = 1000000 - self.I_0 - self.E_0