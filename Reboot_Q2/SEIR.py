import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import binom as binom
from plot import *
import tools

class SEIR():

    def __init__(self):

        # ========================================== #
        #       Epidemic's model parameters
        # ========================================== #
        self.beta = 0.3808921122001155    # Contamination rate
        self.sigma = 0.8137864853740624   # Incubation rate
        self.gamma = 0.24966835247736513    # Recovery rate
        self.hp = 0.014121595288406579     # Hospit rate
        self.hcr = 0.0998263496618203    # Hospit recovery rate
        self.pc = 0.030595079746065835     # Critical rate
        self.pd = 0.04962179357904586     # Critical mortality rate
        self.pcr = 0.1501507991566532     # Critical recovery rate

        # ========================================== #
        #       Testing protocol parameters
        # ========================================== #
        self.s = 0.7000000883724378   # Sensitivity
        self.t = 0.8809706602270286   # Testing rate in symptomatical

        # ========================================== #
        #       Testing Data
        # ========================================== #
        self.dataframe = None   # Pandas dataframe
        self.dataset = None     # Numpy 2D array format

        # Size of the population:
        self.N = 1000000

        # Estimation of the number of infected at t_0
        self.I_0 = 30
        self.E_0 = 15

        # ========================================== #
        #        Hyperparameters dashboard:
        # ========================================== #

        # Optimizer constraints
        self.beta_min = 0.01
        self.beta_max = 1
        self.sigma_min = 1 / 5
        self.sigma_max = 1
        self.gamma_min = 1 / 10
        self.gamma_max = 1 / 4
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
        #self.s_min = 0.70
        self.s_min = 0.1
        #self.s_max = 0.85
        self.s_max = 1
        #self.t_min = 0.5
        self.t_min = 0.1
        self.t_max = 1

        self.E_0_min = 1
        self.E_0_max = 150
        self.I_0_min = 1
        self.I_0_max = 150

        # Weights of each probability in the objective function
        self.w_1 = 1  # Weight of the daily test number
        self.w_2 = 1  # Weight of the daily number of positive tests
        self.w_3 = 1  # Weight of hospitalized data
        self.w_4 = 1  # Weight of critical data
        self.w_5 = 1  # Weight of fatalities

        # Size the variance use fore the normal distribution
        self.var_w_1 = 3
        self.var_w_2 = 3
        self.var_w_3 = 3
        self.var_w_4 = 3
        self.var_w_5 = 3

        # Random generator
        self.rng = np.random.default_rng()

        # Smoothing the dataset?
        self.smoothing = False

        # Optimizer hyperparameter: LBFGSB or COBYLA
        self.optimizer = 'LBFGSB'
        # Step_size, only for LBFGSB
        self.step_size = None

        # Number of simulation to perform during stochastic fitting
        self.nb_simul = 50

        # ========================================== #
        #                   Printers
        # ========================================== #

        # Display fit details
        self.fit_display = True
        # Basis informations about objective function:
        self.basis_obj_display = True
        self.full_obj_display = False
        self.fit_2_display = True
        # Stochastic evidences adapter
        self.stocha_ev_print = False

        self.timeseed = 0

        # ========================================== #
        #                   Vectorization
        # ========================================== #

        # Stochastic predictor function's vectorization:
        self.v_S_to_E = np.vectorize(self.S_to_E)
        self.v_E_to_I = np.vectorize(self.E_to_I)
        self.v_I_to_R_to_H = np.vectorize(self.I_to_R_to_H, otypes=[int, int])
        self.v_H_to_C_to_R = np.vectorize(self.H_to_C_to_R, otypes=[int, int])
        self.v_C_to_R_to_F = np.vectorize(self.C_to_R_to_F, otypes=[int, int])
        self.v_E_to_I_ev = np.vectorize(self.E_to_I_ev, excluded=['priori', 'k_vec'])

        # ========================================== #
        #        Interventions Time Line
        # ========================================== #

        # General contacts distribution
        # For youngs
        self.gc_youngs = {
            'total': 0.075 * self.N,
            'school': 0,
            'work': 0,
            'home': 4.12,
            'comu': 93
        }
        self.gc_juniors = {
            'total': 0.217 * self.N,
            'school': 298.3,
            'work': 0,
            'home': 3.25,
            'comu': 93
        }
        self.gc_mediors = {
            'total': 0.587 * self.N,
            'school': 0,
            'work': 19.87,
            'home': 2.53,
            'comu': 93
        }
        self.gc_seignors = {
            'total': 0.122 * self.N,
            'school': 0,
            'work': 0,
            'home': 1.46,
            'comu': 93
        }
        self.gc_global = {
            'youngs': self.gc_youngs,
            'juniors': self.gc_juniors,
            'mediors': self.gc_mediors,
            'seignors': self.gc_seignors
        }
        # Get a matrix format:
        self.contacts_time_line = None

        # Beta time line
        self.beta_time_line = None





    def get_parameters(self):

        prm = (
        self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd,
        self.pcr, self.s, self.t)
        return prm

    def param_translater(self, method='dict', input=None):
            """
            Return parameters of the model:
            - if input = None: return parameters of the instance of the class
                - In a dictionary if method  = 'dict'
                - In an array if method = 'array
            - if input = an array and method = 'dict': translate into a dictrionary
            - if input = a dictionary and method = 'array': translate into an array

            """
            prm = None
            if input is not None:
                if method == 'dict':
                    prm = {
                        'beta': input[0],
                        'sigma': input[1],
                        'gamma': input[2],
                        'hp': input[3],
                        'hcr': input[4],
                        'pc': input[5],
                        'pd': input[6],
                        'pcr': input[7],
                        's': input[8],
                        't': input[9]}
                else:
                    prm = input.values()
            else:
                if method == 'dict':
                    prm = {
                        'beta': self.beta,      # Contamination rate
                        'sigma': self.sigma,    # Incubation rate
                        'gamma': self.gamma,    # Recovery rate
                        'hp': self.hp,          # Hospit rate
                        'hcr': self.hcr,        # Hospit recovery rate
                        'pc': self.pc,          # Critical rate
                        'pd': self.pd,          # Critical mortality rate
                        'pcr': self.pcr,        # Critical recovery rate
                        's': self.s,            # Sensitivity
                        't': self.t}            # Testing rate in symptomatical
                if method == 'array':
                    prm = [self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd,
                           self.pcr, self.s, self.t]
            return prm

    def get_initial_state(self, sensib=None, test_rate=None, sigma=None, I_start=None, E_start=None):
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
        if I_start is None:
            I_0 = self.I_0 / (s * t)
        else:
            I_0 = I_start / (s * t)

        H_0 = self.dataset[0][3]
        if E_start is not None:
            E_0 = E_start
        else:
            if self.E_0 is not None:
                E_0 = self.E_0
            else:
                E_0 = self.dataset[1][1] / (sig * s * t)
        D_0 = 0
        C_0 = 0
        S_0 = 1000000 - I_0 - H_0 - E_0
        R_0 = 0
        dE_to_I_0 = self.dataset[0][1] / (s * t)
        dI_to_H_0 = H_0
        dI_to_R_0 = 0
        init = (
        S_0, E_0, I_0, R_0, H_0, C_0, D_0, dE_to_I_0, dI_to_H_0, dI_to_R_0)
        #init = np.around(init)
        #return np.asarray(init, dtype=int)
        return init

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr,
                     s, t):
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
            #init = self.get_initial_state(sensib=prm[8], test_rate=prm[9],
            #                                         sigma=prm[1])
            init = np.asarray(self.get_initial_state(sensib=prm[8], test_rate=prm[9],
                                          sigma=prm[1]), dtype=int)

        # Make prediction:
        predict = odeint(func=self.differential,
                         y0=init,
                         t=time,
                         args=(tuple(prm)))
        return predict

    def stochastic_predic_sans_ev(self, duration, parameters=None, init=None, nb_simul=200):
        """
        This stochastic predictor make stochastic simulations of the epidemic
        without using testing data to refine predictions.
        """
        # Get parameters:
        params = parameters
        if params is None:
            params = self.get_parameters()
        # time vector
        time = np.arange(duration)
        # General array to store predictions
        output = np.zeros((len(time), 9, nb_simul), dtype=int)
        # get and store initial state
        init_state = init
        if init_state is None:
            init_state = np.asarray(
                self.get_initial_state(sensib=params[8], test_rate=params[9],
                                       sigma=params[1]),
                dtype=int)
        for i in range(0, 9):
            output[0, i, :] = init_state[i]
        N = self.N

        #self.timeseed += 1
        #np.random.seed(self.timeseed)
        for i in range(1, len(time)):

            # Get class moves
            S_to_E = self.v_S_to_E(output[i - 1, 0, :], output[i - 1, 2, :], N,
                              params[0])
            E_to_I = self.v_E_to_I(output[i - 1, 1, :], params[1])
            I_to_R, I_to_H = self.v_I_to_R_to_H(output[i - 1, 2, :], params[2],
                                           params[3])
            H_to_C, H_to_R = self.v_H_to_C_to_R(output[i - 1, 4, :], params[5],
                                           params[4])
            C_to_R, C_to_F = self.v_C_to_R_to_F(output[i - 1, 5, :], params[7],
                                           params[6])

            # Update states:
            output[i, 0, :] = output[i - 1, 0, :] - S_to_E  # S
            output[i, 1, :] = output[i - 1, 1, :] + S_to_E - E_to_I  # E
            output[i, 2, :] = output[i - 1, 2, :] + E_to_I - I_to_R - I_to_H  # I
            output[i, 3, :] = output[i - 1, 3, :] + I_to_R + C_to_R + H_to_R  # R
            output[i, 4, :] = output[i - 1, 4, :] + I_to_H - H_to_R - H_to_C  # H
            output[i, 5, :] = output[i - 1, 5, :] + H_to_C - C_to_R - C_to_F  # C
            output[i, 6, :] = output[i - 1, 6, :] + C_to_F  # F
            output[i, 7, :] = output[i - 1, 7, :] + E_to_I  # CI
            output[i, 8, :] = output[i - 1, 8, :] + I_to_H  # CH

        return output

    def stochastic_predic(self, duration, parameters=None, nb_simul=200,
                          scenar=False):

        # Get parameters:
        params = parameters
        if params is None:
            params = np.asarray(self.get_parameters())
        # time vector
        time = np.arange(duration)
        # General array to store predictions
        output = np.zeros((len(time), 9, nb_simul), dtype=int)
        # get and store initial state
        init_state = np.asarray(
            self.get_initial_state(sensib=params[8], test_rate=params[9],
                                   sigma=params[1]), dtype=int)
        for i in range(0, 9):
            output[0, i, :] = init_state[i]
        N = 1000000

        # Get prior distributions
        priori_lngth = duration
        if duration >= self.dataset.shape[0]:
            priori_lngth = self.dataset.shape[0]
        max_n = self.dataset[priori_lngth - 1, 1] * 2
        # Matrix to store distribution
        priori = np.zeros((max_n, priori_lngth))
        # values of n
        n_vec = np.arange(priori.shape[0])
        k_vec = self.dataset[0:priori.shape[1], 1]

        # Build prior distributions
        for i in range(0, priori.shape[0]):
            # Build the binomial object
            binom_obj = binom(n=i, p=params[8] * params[9])
            # Compute the probability of the evidence given prediction
            priori[i, :] = binom_obj.pmf(k=k_vec)

        for i in range(1, len(time)):
            #if i % 10 == 0:
                #print(i)
            if scenar:
                params[0] = self.beta_time_line[i]

            # Get class moves
            S_to_E = self.v_S_to_E(output[i - 1, 0, :], output[i - 1, 2, :], N,
                              params[0])
            if i >= priori_lngth:
                E_to_I = self.v_E_to_I(output[i - 1, 1, :], params[1])
            else:
                E_to_I = self.v_E_to_I_ev(output[i - 1, 1, :], sigma=params[1],
                                     priori=priori[:, i], k_vec=n_vec)

            I_to_R, I_to_H = self.v_I_to_R_to_H(output[i - 1, 2, :], params[2],
                                           params[3])
            H_to_C, H_to_R = self.v_H_to_C_to_R(output[i - 1, 4, :], params[5],
                                           params[4])
            C_to_R, C_to_F = self.v_C_to_R_to_F(output[i - 1, 5, :], params[7],
                                           params[6])

            # Update states:

            output[i, 0, :] = output[i - 1, 0, :] - S_to_E  # S
            output[i, 1, :] = output[i - 1, 1, :] + S_to_E - E_to_I  # E
            output[i, 2, :] = output[i - 1, 2, :] + E_to_I - I_to_R - I_to_H  # I
            output[i, 3, :] = output[i - 1, 3, :] + I_to_R + C_to_R + H_to_R  # R
            output[i, 4, :] = output[i - 1, 4, :] + I_to_H - H_to_R - H_to_C  # H
            output[i, 5, :] = output[i - 1, 5, :] + H_to_C - C_to_R - C_to_F  # C
            output[i, 6, :] = output[i - 1, 6, :] + C_to_F  # F
            output[i, 7, :] = output[i - 1, 7, :] + E_to_I  # CI
            output[i, 8, :] = output[i - 1, 8, :] + I_to_H  # CH

        return output

    def E_to_I_ev(self, E, sigma, priori, k_vec):
        binom_obj = binom(n=E, p=sigma)
        poste = binom_obj.pmf(k_vec)
        distri = priori * poste

        # normalize:
        np.nan_to_num(distri, copy=False)
        smn = np.sum(distri)
        if smn == 0:
            a = np.argmax(priori)
            b = np.argmax(poste)
            return np.around((a + b) / 2)
        distri /= smn
        choise = np.random.choice(k_vec, p=distri)
        return choise

    def S_to_E(self, S, I, N, beta):

        fact = beta * I / N
        return self.rng.multinomial(S, [fact, 1 - fact])[0]

    def E_to_I(self, E, sigma):

        return self.rng.multinomial(E, [sigma, 1 - sigma])[0]

    def I_to_R_to_H(self, I, gamma, hp):
        tmp = self.rng.multinomial(I, [gamma, hp, 1 - (gamma + hp)])
        return tmp[0], tmp[1]

    def H_to_C_to_R(self, H, pc, hcr):
        tmp = self.rng.multinomial(H, [pc, hcr, 1 - (pc + hcr)])
        return tmp[0], tmp[1]

    def C_to_R_to_F(self, C, pcr, pd):
        tmp = self.rng.multinomial(C, [pcr, pd, 1 - (pcr + pd)])
        return tmp[0], tmp[1]

    def fit(self, method='normal'):
        """
        Compute best epidemic parameters values according to model's hyperparameters and the dataset
        """

        # Initial values of parameters:
        init_prm = (self.beta, self.sigma, self.gamma, self.hp,
                    self.hcr, self.pc, self.pd, self.pcr, self.s, self.t)
        # Bounds
        bds = [(self.beta_min, self.beta_max), (self.sigma_min, self.sigma_max),
               (self.gamma_min, self.gamma_max),
               (self.hp_min, self.hp_max), (self.hcr_min, self.hcr_max),
               (self.pc_min, self.pc_max),
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
                           # options={'eps': self.step_size},
                           args=(method),
                           bounds=bds)
        else:
            if self.optimizer == 'COBYLA':
                res = minimize(self.objective, np.asarray(init_prm),
                               method='COBYLA',
                               args=(method),
                               constraints=cons)

        #if self.fit_display:
            # Print optimizer result
            #print(res)

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
        # Constraint checking:
        check = False
        if parameters[5] + parameters[4] >= 1:  # pc + hcr < 1
            check = True
        if parameters[2] + parameters[3] >= 1:  # gamma + hp < 1
            check = True
        if parameters[6] + parameters[7] >= 1:  # pcr + pd < 1
            check = True
        if check:
            return 20000
        # Put parameters into a tuple
        params = tuple(parameters[0:10])
        sensitivity = params[8]
        testing_rate = params[9]

        # Get an initial state:
        init_state = self.get_initial_state(sensib=sensitivity,
                                            test_rate=testing_rate,
                                            sigma=params[1])
        if self.basis_obj_display:
            print(params)
        # Make prediction
        if method == 'normal' or method == 'Normal':
            predictions = self.predict(duration=self.dataset.shape[0],
                                       parameters=params,
                                       initial_state=init_state)
        if method == 'stocha':
            # Predict with 200 simulations
            tmp = self.nb_simul
            self.nb_simul = 50
            # Get predictions matrix:
            prd_mat = self.stochastic_predic_sans_ev(
                duration=self.dataset.shape[0],
                parameters=params,
                init=init_state)
            self.nb_simul = tmp
            # Get mean vectors
            predictions = np.mean(prd_mat, axis=2)
        # Time to compare:
        start_t = 15
        end_t = self.dataset.shape[0]

        infections = [predictions[0][7]]
        for i in range(1, end_t):
            infections.append(predictions[i][7] - predictions[i - 1][7])

        # Compare with dataset and compute the likelyhood value
        error = 0.0
        for i in range(start_t, end_t):
            err1 = err2 = err3 = err4 = err5 = 0.0
            # ================================================ #
            # PART 1: Test number fitting
            # ================================================ #
            # Predictions: the number of contamination is multiply by the testing rate
            pred = infections[i] * testing_rate
            # Divide it by the positivity rate:
            pred /= (self.dataset[i][2] / self.dataset[i][1])
            # Evidences: the testing number of the day
            evid = self.dataset[i][2]
            # The variance of the distribution
            sigma_sq = np.fabs(self.var_w_1 * evid)
            # Difference between prediction and evidence
            dx = np.fabs(pred - evid)
            # Avoid the exact case
            if sigma_sq == 0:
                sigma_sq = 1
            prob_1 = tools.normal_density(sigma_sq, dx)
            # Add to the log probability
            if prob_1 < 0.00000000000000000001:
                err1 += 50 * self.w_1
            else:
                err1 -= np.log(prob_1) * self.w_1
            if self.full_obj_display:
                print(
                    'iter {} - prb_1 {} - sigma2 {} - dx {} - pred {} - ev {}'.format(
                        i, prob_1, sigma_sq,
                        dx, pred, evid))
            if np.log(prob_1) * self.w_1 > 0:
                print(
                    'iter {} - prb_1 {} - sigma2 {} - dx {} - pred {} - ev {}'.format(
                        i, prob_1, sigma_sq,
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
        if self.basis_obj_display:
            print('score = {}'.format(error))
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
            cumul_positive[i] += cumul_positive[i - 1]
        raw.insert(7, 'cumul_positive', cumul_positive)
        if self.smoothing:
            self.dataframe = dataframe_smoothing(raw)
        else:
            self.dataframe = raw
        self.dataset = self.dataframe.to_numpy()

    def set_scenario(self, scenario):

        # Lenght
        predict_length = scenario['duration']
        # Get scenarios items
        scenario_keys = scenario.keys()

        # Build the beta time line:
        self.beta_time_line = np.ones(predict_length)
        self.beta_time_line *= self.beta
        # Build the contacts matrix time line
        self.contacts_time_line = np.zeros((predict_length, 4, 5))
        i = 0
        for age_keys in self.gc_global.keys():
            j = 0
            for place_keys in self.gc_seignors.keys():
                self.contacts_time_line[:, i, j] = self.gc_global[age_keys][
                    place_keys]
                j += 1
            i += 1

        # School management:
        if 'close_schools' in scenario_keys:
            #print('--------------------------')
            #for i in range(0, 150):
                #print(self.contacts_time_line[i, 1, 1])
            # Get times:
            start, end = scenario['close_schools']
            # Modify time line contact matrix
            self.contacts_time_line[start:end, :, 1] = 0

        if 'case_isolation' in scenario_keys:
            """
            WARNING: only 50 - 70% of infected are symptomatic
            Infection rate with isolated reduce by :
                - 25% at home
                - 90% in communities
                - 100% at work and school
            """
            # Get times:
            start, end = scenario['case_isolation']
            # Modify time line contact matrix
            # At school
            self.contacts_time_line[start:end, :, 1] -= self.contacts_time_line[
                                                        start:end, :,
                                                        1] * 0.9 * 0.6
            # At work
            self.contacts_time_line[start:end, :, 2] -= self.contacts_time_line[
                                                        start:end, :,
                                                        2] * 0.9 * 0.6
            # At home
            self.contacts_time_line[start:end, :, 3] -= self.contacts_time_line[
                                                        start:end, :,
                                                        3] * 0.9 * 0.6 * 0.25
            # In communities
            self.contacts_time_line[start:end, :, 4] -= self.contacts_time_line[
                                                        start:end, :,
                                                        4] * 0.9 * 0.6 * 0.9

        if 'home_quarantine' in scenario_keys:
            """
            Exactly the same that case isolation but also apply to assymptomatics
            """
            # Get times:
            start, end = scenario['home_quarantine']
            # Modify time line contact matrix
            # At school
            self.contacts_time_line[start:end, :, 1] -= self.contacts_time_line[
                                                        start:end, :, 1] * 0.9
            # At work
            self.contacts_time_line[start:end, :, 2] -= self.contacts_time_line[
                                                        start:end, :, 2] * 0.9
            # At home
            self.contacts_time_line[start:end, :, 3] -= self.contacts_time_line[
                                                        start:end, :,
                                                        3] * 0.9 * 0.25
            # In communities
            self.contacts_time_line[start:end, :, 4] -= self.contacts_time_line[
                                                        start:end, :,
                                                        4] * 0.9 * 0.9

        if 'lock_down' in scenario_keys:
            """
            Reduce of
                - 100% schools
                - 75% communities
                - 75% work
            """
            # Get times:
            start, end = scenario['lock_down']
            # Modify time line contact matrix
            # At school
            self.contacts_time_line[start:end, :, 1] = 0
            # At work
            self.contacts_time_line[start:end, :, 2] -= self.contacts_time_line[
                                                        start:end, :,
                                                        2] * 0.9 * 0.75
            # At home
            self.contacts_time_line[start:end, :, 3] -= 0
            # In communities
            self.contacts_time_line[start:end, :, 4] -= self.contacts_time_line[
                                                        start:end, :,
                                                        4] * 0.9 * 0.75

        if 'social_dist' in scenario_keys:
            """
            Reduce of
                - 75 % at work and communities
                - Parameter start age: if we take junior or not
            """
            # Get times:
            start, end, age = scenario['social_dist']
            # Modify time line contact matrix
            if age < 6:
                for i in range(start, end):
                    # At school
                    self.contacts_time_line[i, :, 1] -= self.contacts_time_line[
                                                        i, :, 1] * 0.9 * 0.75
                    # At work
                    self.contacts_time_line[i, :, 2] -= self.contacts_time_line[
                                                        i, :, 2] * 0.9 * 0.75
                    # At home
                    self.contacts_time_line[i, :, 3] -= 0
                    # In communities
                    self.contacts_time_line[i, :, 4] -= self.contacts_time_line[
                                                        i, :, 4] * 0.9 * 0.75
            if age >= 6:
                for i in range(start, end):
                    # At school
                    #self.contacts_time_line[i, 1:4,
                    #1] -= self.contacts_time_line[i, 1:4, 1] * 0.9 * 0.75
                    # At work
                    self.contacts_time_line[i, 1:4,
                    2] -= self.contacts_time_line[i, 1:4, 2] * 0.9 * 0.75
                    # At home
                    self.contacts_time_line[i, 1:4, 3] -= 0
                    # In communities
                    self.contacts_time_line[i, 1:4,
                    4] -= self.contacts_time_line[i, 1:4, 4] * 0.9 * 0.75

        if 'wearing_mask' in scenario_keys:
            """
            Decrease of:
                - 20%
            """
            # Get times:
            start, end = scenario['wearing_mask']
            # Modify time line contact matrix
            # At school
            #self.contacts_time_line[start:end, :, 1] -= self.contacts_time_line[
            #                                            start:end, :,
            #                                            1] * 0.9 * 0.2
            # At work
            self.contacts_time_line[start:end, :, 2] -= self.contacts_time_line[
                                                        start:end, :,
                                                        2] * 0.9 * 0.2
            # At home
            self.contacts_time_line[start:end, :, 3] -= 0
            # In communities
            self.contacts_time_line[start:end, :, 4] -= self.contacts_time_line[
                                                        start:end, :,
                                                        4] * 0.9 * 0.2

        # Apply to the beta time line vector

        # Get the original number of contacts:
        a = self.contacts_time_line[0, :, 0] * self.contacts_time_line[0, :, 1]
        b = self.contacts_time_line[0, :, 0] * self.contacts_time_line[0, :, 2]
        c = self.contacts_time_line[0, :, 0] * self.contacts_time_line[0, :, 3]
        d = self.contacts_time_line[0, :, 0] * self.contacts_time_line[0, :, 4]
        e = a + b + c + d
        total_contact = np.sum(e)

        total_contact = np.ones(predict_length) * total_contact
        new_contacts = np.zeros(predict_length)
        for i in range(0, predict_length):
            # Get the original number of contacts:
            a = self.contacts_time_line[i, :, 0] * self.contacts_time_line[i, :,
                                                   1]
            b = self.contacts_time_line[i, :, 0] * self.contacts_time_line[i, :,
                                                   2]
            c = self.contacts_time_line[i, :, 0] * self.contacts_time_line[i, :,
                                                   3]
            d = self.contacts_time_line[i, :, 0] * self.contacts_time_line[i, :,
                                                   4]
            e = a + b + c + d
            new_contacts[i] = np.sum(e)
        # Apply the ratio to beta time line vector:
        #for i in range(0, 150):
            #print('{} - {}'.format(total_contact[i], new_contacts[i]))
        self.beta_time_line = self.beta_time_line * (
                    new_contacts / total_contact)

        #for i in range(0, predict_length):
            #print(self.beta_time_line[i])

    def stochastic_mean(self, duration):
        sp = self.stochastic_predic(duration=duration)
        sp_S = sp[:, 0, :]
        sp_E = sp[:, 1, :]
        sp_I = sp[:, 2, :]
        sp_R = sp[:, 3, :]
        sp_H = sp[:, 4, :]
        sp_C = sp[:, 5, :]
        sp_D = sp[:, 6, :]

        mean = [np.mean(sp_S, axis=1), np.mean(sp_E, axis=1), np.mean(sp_I, axis=1),
                np.mean(sp_R, axis=1), np.mean(sp_H, axis=1), np.mean(sp_C, axis=1),
                np.mean(sp_D, axis=1)]

        sto_hq = [np.mean(sp_S, axis=1) + (2 * np.std(sp_S, axis=1)),
                  np.mean(sp_E, axis=1) + (2 * np.std(sp_E, axis=1)),
                  np.mean(sp_I, axis=1) + (2 * np.std(sp_I, axis=1)),
                  np.mean(sp_R, axis=1) + (2 * np.std(sp_R, axis=1)),
                  np.mean(sp_H, axis=1) + (2 * np.std(sp_H, axis=1)),
                  np.mean(sp_C, axis=1) + (2 * np.std(sp_C, axis=1)),
                  np.mean(sp_D, axis=1) + (2 * np.std(sp_D, axis=1))]

        sto_lq = [np.mean(sp_S, axis=1) - (2 * np.std(sp_S, axis=1)),
                  np.mean(sp_E, axis=1) - (2 * np.std(sp_E, axis=1)),
                  np.mean(sp_I, axis=1) - (2 * np.std(sp_I, axis=1)),
                  np.mean(sp_R, axis=1) - (2 * np.std(sp_R, axis=1)),
                  np.mean(sp_H, axis=1) - (2 * np.std(sp_H, axis=1)),
                  np.mean(sp_C, axis=1) - (2 * np.std(sp_C, axis=1)),
                  np.mean(sp_D, axis=1) - (2 * np.std(sp_D, axis=1))]

        return mean, sto_hq, sto_lq, sp_S, sp_E, sp_I, sp_R, sp_H, sp_C, sp_D

    def plot(self, filename, type, duration=0, plot_conf_inter=False,
             global_view=False, plot_param=False):
        """
        @param filename(@type String): name of file to save plot in
        @param type(@type String): type of curves to plot
        @param duration(@type int): duration of predictions
        @param plot_conf_inter(@type bool): plot confidence range
        @param global_view(@type bool): plot all stochastic curves
        @return:
        """
        plot_dataset(self, filename, type, duration, plot_conf_inter,
                     global_view, plot_param)


