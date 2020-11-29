import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from scipy.stats import binom as binom
import tools
from plot import plot_dataset
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
        self.pd = 0.1           # Critical mortality rate
        self.pcr = 0.3          # Critical recovery rate

        # Only for fit_part_2
        self.I_out = None       # Sum of probability to leave I each day

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
        self.fit_2_display = True


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

    def set_parameters_from_bf(self, df):
        """
        This method initialize the values of parameters from a one row dataframe
        at the format who is given by the bruteforce process
        """
        self.beta = df['beta_final']
        self.sigma = df['sigma_final']
        self.gamma = df['gamma_final']
        self.hp = df['hp_final']
        self.hcr = df['hcr_final']
        self.pc = df['pc_final']
        self.pd = df['pd_final']
        self.pcr = df['pcr_final']
        self.s = df['s_final']
        self.t = df['t_final']
        self.I_0 = df['I_0']

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

    def stochastic_predic(self, time):

        output = np.zeros((len(time), 9))
        # Initial state:
        init_state = self.get_initial_state()
        output[0][0] = int(init_state[0])                       #s
        output[0][1] = int(init_state[1])                       #E
        output[0][2] = int(init_state[2])                       #i
        output[0][3] = int(init_state[3])                       #R
        output[0][4] = int(init_state[4])                       #H
        output[0][5] = int(init_state[5])                       #C
        output[0][6] = int(init_state[6])                       #F
        output[0][7] = int(init_state[7])                       #CI
        output[0][8] = int(init_state[8])                       #CH

        N = 1000000

        #params = (self.beta, self.sigma, self.gamma, self.sensitivity)


        for i in range(1, len(time)):

            S_to_E = np.random.multinomial(output[i-1][0], [self.beta * output[i-1][2] / N, 1-(self.beta * output[i-1][2] / N)])[0]
            #print(S_to_E)
            E_to_I = np.random.multinomial(output[i-1][1], [self.sigma, 1-self.sigma])[0]
            #print(E_to_I)

            I_to_R = np.random.multinomial(output[i-1][2], [self.gamma, self.hp, 1-(self.gamma+self.hp)])[0]

            I_to_H = np.random.multinomial(output[i-1][2], [self.gamma, self.hp, 1-(self.gamma+self.hp)])[1]

            H_to_C = np.random.multinomial(output[i-1][4], [self.pc, self.hcr, 1-(self.pc+self.hcr)])[0]
            H_to_R = np.random.multinomial(output[i-1][4], [self.pc, self.hcr, 1-(self.pc+self.hcr)])[1]

            C_to_R = np.random.multinomial(output[i-1][5], [self.pcr, self.pd, 1-(self.pcr+self.pd)])[0]
            C_to_F = np.random.multinomial(output[i-1][5], [self.pcr, self.pd, 1-(self.pcr+self.pd)])[1]


            # Update states:

            output[i][0] = output[i-1][0] - S_to_E                        #S
            output[i][1] = output[i-1][1] + S_to_E - E_to_I               #E


            output[i][2] = output[i-1][2] + E_to_I - I_to_R - I_to_H      #I
            # WARNING: if the epidemic die (infected <= 0) we consider a population of 0 infected
            if output[i][2] < 0:
                """
                Remove the number of people going out I compartiment
                who doesn't exist (when output value <0)
                """
                I_to_R -= abs(output[i][2])*(self.gamma/(self.gamma+self.hp))
                I_to_H -= abs(output[i][2])*(self.hp/(self.gamma+self.hp))
                """
                DIY to work well
                """
                if ((I_to_R*10) % 5) == 0:
                    I_to_H += 0.1
                I_to_H = np.round(I_to_H)
                I_to_R = np.round(I_to_R)
                output[i][2] = 0


            output[i][3] = output[i-1][3] + I_to_R + C_to_R + H_to_R      #R


            output[i][4] = output[i-1][4] + I_to_H - H_to_R - H_to_C      #H
            # WARNING:
            if output[i][4] < 0:
                """
                Remove the number of people going out H compartiment
                who doesn't exist (when output value <0)
                """
                H_to_R -= abs(output[i][4])*self.pc/self.pc+self.hcr
                H_to_C -= abs(output[i][4])*self.hcr/self.pc+self.hcr
                """
                DIY to work well
                """
                if ((H_to_R*10) % 5) == 0:
                    H_to_R += 0.1

                H_to_R = np.round(H_to_R)
                H_to_C = np.round(H_to_C)

                output[i][4] = 0

            output[i][5] = output[i-1][5] + H_to_C - C_to_R - C_to_F      #C
            # WARNING:
            if output[i][5] < 0:
                """
                Remove the number of people going out C compartiment
                who doesn't exist (when output value <0)
                """
                C_to_R -= abs(output[i][5])*self.pcr/self.pcr+self.pd
                C_to_F -= abs(output[i][5])*self.pd/self.pcr+self.pd
                """
                DIY to work well
                """
                if ((C_to_F*10) % 5) == 0:
                    C_to_F += 0.1

                C_to_R = np.round(C_to_R)
                C_to_F = np.round(C_to_F)

                output[i][5] = 0


            output[i][6] = output[i-1][6] + C_to_F                        #F
            output[i][7] = output[i-1][7] + np.random.binomial(E_to_I, self.s)      #CI
            output[i][8] = output[i-1][8] + I_to_H     #CH


        return output

    def stochastic_mean(self, time, nb_simul):
        '''
        Used to predict the stochastical model based on the mean of an important number of simulations

        Parameters
        ----------
        time: vector(int)
            vector of time to evaluate the stochastic prediction
        nb_simul: (int)
            number of simulation to evaluate the mean on

        Returns
        -------


        '''

        result_S = np.zeros((len(time), nb_simul))
        result_E = np.zeros((len(time), nb_simul))
        result_I = np.zeros((len(time), nb_simul))
        result_R = np.zeros((len(time), nb_simul))
        result_H = np.zeros((len(time), nb_simul))
        result_C = np.zeros((len(time), nb_simul))
        result_F = np.zeros((len(time), nb_simul))

        result_Conta = np.zeros((len(time), nb_simul))
        for i in range(0, nb_simul):

            pred = self.stochastic_predic(time)
            for j in range(0, len(time)):
                result_S[j][i] = pred[j][0]
                result_E[j][i] = pred[j][1]
                result_I[j][i] = pred[j][2]
                result_R[j][i] = pred[j][3]
                result_H[j][i] = pred[j][4]
                result_C[j][i] = pred[j][5]
                result_F[j][i] = pred[j][6]
                result_Conta[j][i] = pred[j][7]

        mean = np.zeros((len(time), 8))
        hquant = np.zeros((len(time), 8))
        lquant = np.zeros((len(time), 8))
        std = np.zeros((len(time), 8))

        n_std = 2

        for i in range(0, len(time)):
            mean[i][0] = np.mean(result_S[i, :])
            mean[i][1] = np.mean(result_E[i, :])
            mean[i][2] = np.mean(result_I[i, :])
            mean[i][3] = np.mean(result_R[i, :])
            mean[i][4] = np.mean(result_H[i, :])
            mean[i][5] = np.mean(result_C[i, :])
            mean[i][6] = np.mean(result_F[i, :])
            mean[i][7] = np.mean(result_Conta[i, :])

            std[i][0] = np.std(result_S[i, :])
            std[i][1] = np.std(result_E[i, :])
            std[i][2] = np.std(result_I[i, :])
            std[i][3] = np.std(result_R[i, :])
            std[i][4] = np.std(result_H[i, :])
            std[i][5] = np.std(result_C[i, :])
            std[i][6] = np.std(result_F[i, :])
            std[i][7] = np.std(result_Conta[i, :])

            # WARNING: 70% confidence interval
            hquant[i][0] = np.mean(result_S[i, :]) + n_std * std[i][0]
            hquant[i][1] = np.mean(result_E[i, :]) + n_std * std[i][1]
            hquant[i][2] = np.mean(result_I[i, :]) + n_std * std[i][2]
            hquant[i][3] = np.mean(result_R[i, :]) + n_std * std[i][3]
            hquant[i][4] = np.mean(result_H[i, :]) + n_std * std[i][4]
            hquant[i][5] = np.mean(result_C[i, :]) + n_std * std[i][5]
            hquant[i][6] = np.mean(result_F[i, :]) + n_std * std[i][6]
            hquant[i][7] = np.mean(result_Conta[i, :]) + n_std * std[i][7]

            lquant[i][0] = np.mean(result_S[i, :]) - n_std * std[i][0]
            lquant[i][1] = np.mean(result_E[i, :]) - n_std * std[i][1]
            lquant[i][2] = np.mean(result_I[i, :]) - n_std * std[i][2]
            lquant[i][3] = np.mean(result_R[i, :]) - n_std * std[i][3]
            lquant[i][4] = np.mean(result_H[i, :]) - n_std * std[i][4]
            lquant[i][5] = np.mean(result_C[i, :]) - n_std * std[i][5]
            lquant[i][6] = np.mean(result_F[i, :]) - n_std * std[i][6]
            lquant[i][7] = np.mean(result_Conta[i, :]) - n_std * std[i][7]

        return mean, hquant, lquant, std, result_S, result_E, result_I, result_R, result_H, result_C, result_F, result_Conta


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
            # Predictions: the number of contamination is multiply by the testing rate
            pred = infections[i] * testing_rate
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

    def objective_part_2(self, parameters):


        params = (self.beta, self.sigma, self.gamma, self.hp, parameters[0], parameters[1],
                  parameters[2], parameters[3], self.s, self.t)

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

        # if method == 'bruteforce':
        #    start_t = 7
        #    end_t = 35
        # Uncumul tests predictions:
        infections = [predictions[0][7]]
        for i in range(1, end_t):
            infections.append(predictions[i][7] - predictions[i - 1][7])

        # Compare with dataset and compute the likelyhood value
        error = 0.0
        for i in range(start_t, end_t):
            err3 = err4 = err5 = 0.0

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

            error += err3 + err4 + err5

        return error

    def objective_cumul_hospit(self, gamma, hp):
        """
        Compute the score for the curent model only on cumul hospit
        """
        # Put parameters into a tuple
        params = self.beta, self.sigma, gamma, hp, self.hcr, self.pc, self.pd, self.pcr, self.s, self.t

        # Get an initial state:
        init_state = self.get_initial_state(sensib=self.s, test_rate=self.t, sigma=params[1])
        # Make prediction
        predictions = self.predict(duration=self.dataset.shape[0],
                            parameters=params,
                            initial_state=init_state)
        # Time to compare:
        start_t = 3
        end_t = self.dataset.shape[0]

        # Compare with dataset
        error = 0.0
        for i in range(start_t, end_t):
            err = 0.0
            # ================================================ #
            # PART 1: Fit on hospitalized data CUMUL
            # ================================================ #
            # the predicted value is multiply by
            pred = predictions[i][8]
            evid = self.dataset[i][4]
            # Standardize and use normal distribution:
            sigma_sq = np.fabs(self.var_w_2 * evid)
            dx = np.fabs(pred - evid)
            if sigma_sq == 0:
                sigma_sq = 1
            prob = tools.normal_density(sigma_sq, dx)
            # Add to the log probability
            if prob < 0.00000000000000000001:
                err += 50 * self.w_3
            else:
                err -= np.log(prob) * self.w_3

            error += err
        return error


    def fit_part_2(self):

        # Initial value of parameters:
        init_prm = (self.hcr, self.pc, self.pd, self.pcr)
        # Bounds
        bds = [(0, 0.5), (0, 0.5), (0, 0.5), (0.1, 0.5)]
        # Minimize
        res = minimize(self.objective_part_2, np.asarray(init_prm),
                       method='L-BFGS-B',
                       bounds=bds)
        if self.fit_2_display:
            print(res)

        # Update parameters:
        self.hcr = res.x[0]
        self.pc = res.x[1]
        self.pd = res.x[2]
        self.pcr = res.x[3]

        return res.fun

    def fit_gamma_hp_rate(self, plot=True):

        gamma = self.gamma
        hp = self.hp
        hs_sum = gamma + hp

        gamma_range = np.linspace(0, hs_sum, 200)
        hp_range = np.zeros(len(gamma_range))
        for i in range(0, len(gamma_range)):
            hp_range[i] = gamma_range[-(i+1)]
        proportion_range = np.linspace(0, 1, 200)
        best = (math.inf, 0, 0)
        error = []
        for i in range(0, len(gamma_range)):

            err = self.objective_cumul_hospit(gamma_range[i], hp_range[i])
            error.append(err)
            if err < best[0]:
                best = (err, gamma_range[i], hp_range[i])

        if plot:

            plt.plot(proportion_range, error, color='red')
            plt.title('Evolution of the error in fit_gamma_hp_rate')
            plt.show()

        self.gamma = best[1]
        self.hp = best[2]



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

    def plot_fit_cumul(self, duration=0, plot_conf_inter=False,
                       global_view=False, plot_param=False):
        """
        See. self.plot()
        """
        self.plot(filename='fit_on_cum_num_pos',
                  type='--ds-cum_num_pos --ds-num_pos --det-+CC --sto-+CC',
                  duration=duration,
                  plot_conf_inter=plot_conf_inter,
                  global_view=global_view,
                  plot_param=plot_param)

    def plot_fit_hosp(self, duration=0, plot_conf_inter=False,
                      global_view=False, plot_param=False):
        """
        See. self.plot()
        """
        self.plot(filename='fit_on_cum_hospitalized',
                  type='--ds-num_cum_hospit --det-+CH',
                  duration=duration,
                  plot_conf_inter=plot_conf_inter,
                  global_view=global_view,
                  plot_param=plot_param)

    def plot_fit_crit(self, duration=0, plot_conf_inter=False,
                      global_view=False, plot_param=False):
        """
        See. self.plot()
        """
        self.plot(filename='fit_on_criticals',
                  type='--ds-num_crit --det-C --sto-C',
                  duration=duration,
                  plot_conf_inter=plot_conf_inter,
                  global_view=global_view,
                  plot_param=plot_param)

    def plot_fit_death(self, duration=0, plot_conf_inter=False,
                       global_view=False, plot_param=False):
        """
        See. self.plot()
        """
        self.plot(filename='fit_on_death',
                  type='--ds-num_fatal --det-D --sto-D',
                  duration=duration,
                  plot_conf_inter=plot_conf_inter,
                  global_view=global_view,
                  plot_param=plot_param)


    def ploter(self):

        self.plot(filename="Sto(I,E,H,C,F)",
                   type='--sto-I --sto-E --sto-H --sto-C --sto-F',
                   duration=200,
                   global_view=True)

        self.plot(filename="Sto(S,R)",
                   type='--sto-S --sto-R',
                   duration=200,
                   global_view=True)

        self.plot(filename="Compare_stocha_and_deter(I,E,H,C,F)",
                   type='--sto-I --sto-E --sto-H --sto-C --sto-F' +
                        '--det-I --det-E --det-H --det-C --det-F' ,
                   duration=200,
                   plot_conf_inter=True)


        self.plot(filename="Compare_stocha_and_deter(S,R)",
                   type='--sto-S --sto-R' +
                        '--det-S --det-R' ,
                   duration=200,
                   plot_conf_inter=True)

        self.plot_fit_cumul(plot_conf_inter=True)
        self.plot_fit_hosp(plot_conf_inter=True)
        self.plot_fit_crit(plot_conf_inter=True)
        self.plot_fit_death(plot_conf_inter=True)

        '''
        model.plot(filename="SEIR-MODEL(E,I,H,C,D).pdf",
                   type='--sto-E --sto-I --sto-H --sto-C --sto-D',
                   duration=200,
                   global_view=True)
    
        model.plot(filename="SEIR-MODEL-determinist(S,E,I,R).pdf",
                   type='--det-S --det-E --det-I --det-R',
                   duration=200,
                   global_view=True)
        '''