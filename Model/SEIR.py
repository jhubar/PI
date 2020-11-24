import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from scipy.stats import binom as binom

from smoothing import dataframe_smoothing
from plot import plot_dataset


class SEIR():

    def __init__(self):

        # ========================================== #
        #           Model parameters
        # ========================================== #
        self.beta = 0.545717         # Contamination rate
        self.sigma = 0.778919        # Incubation rate
        self.gamma = 0.2158      # Recovery rate
        self.hp = 0.196489          # Hospit rate
        self.hcr = 0.0514          # Hospit recovery rate
        self.pc = 0.075996           # Critical rate
        self.pd = 0.0458608           # Critical mortality
        self.pcr = 0.293681          # Critical recovery rate
        self.s = 0.765          # Sensitivity
        self.t = 0.75           # Testing rate in symptomatical

        # Learning set
        self.dataframe = None
        self.dataset = None

        # Initial state
        self.I_0 = 0                                    # Infected
        self.E_0 = 0                                    # Exposed
        self.R_0 = 0       # Recovered
        self.S_0 = 0      # Sensible
        self.H_0 = 0
        self.C_0 = 0
        self.D_0 = 0
        self.CT_0 = 0                # Contamined
        self.CH_0 = 0

        # ========================================== #
        #        Hyperparameters dashboard:
        # ========================================== #

        # Importance given to each curve during the fitting process
        self.w_1 = 0.7          # Weight of cumulative positive data
        self.w_2 = 0.3          # Weight of positive data
        self.w_3 = 0.7          # Weight of hopit data
        self.w_4 = 0.3          # Weight of cumul hospit data
        self.w_5 = 1            # Weight àf critical data
        self.w_6 = 1            # Weight of fatalities data

        # Value to return if log(binom.pmf(k,n,p)) = - infinity
        self.overflow = - 1000

        # Smoothing data or not
        self.smoothing = True

        # Binomial smoother: ex: if = 2: predicted value *= 2 and p /= 2 WARNING: only use integer
        self.b_s_1 = 6
        self.b_s_2 = 2
        self.b_s_3 = 6
        self.b_s_4 = 2
        self.b_s_5 = 4
        self.b_s_6 = 4

        # Optimizer step size
        self.opti_step = 0.05

        # Optimizer constraints
        self.beta_min = 0.1
        self.beta_max = 0.9
        self.sigma_min = 1/5
        self.sigma_max = 1
        self.gamma_min = 1/10
        self.gamma_max = 1/4
        self.hp_min = 0.01
        self.hp_max = 0.5
        self.hcr_min = 0.01
        self.hcr_max = 0.4
        self.pc_min = 0.01
        self.pc_max = 0.4
        self.pd_min = 0.01
        self.pd_max = 0.5
        self.pcr_min = 0.01
        self.pcr_max = 0.4
        self.s_min = 0.7
        self.s_max = 0.85
        self.t_min = 0.5
        self.t_max = 1

        # Optimizer choise:
        self.cobyla = False
        self.LBFGSB = False
        self.slsqp = True
        self.auto = False

        self.import_dataset()

    def stochastic_predic(self, time):

        output = np.zeros((len(time), 9))
        # Initial state:
        output[0][0] = self.S_0                       #s
        output[0][1] = self.E_0                       #E
        output[0][2] = self.I_0                       #i
        output[0][3] = self.R_0                       #R
        output[0][4] = self.H_0                       #H
        output[0][5] = self.C_0                       #C
        output[0][6] = self.D_0                       #F
        output[0][7] = self.I_0                       #CI
        output[0][8] = self.H_0                       #CH

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

    def get_parameters(self):

        prm = (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr, self.s, self.t)
        return prm

    def get_initial_state(self):

        init = (self.S_0, self.E_0, self.I_0, self.R_0, self.H_0, self.C_0, self.D_0, self.CT_0, self.CH_0)
        return init

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr,s,t):

        S, E, I, R, H, C, D, CT, CH = state

        dS = -(beta * S * I) / (S + I + E + R + H + C + D)
        dE = ((beta * S * I) / (S + I + E + R + H + C + D)) - (sigma * E)
        dI = (sigma * E) - (gamma * I) - (hp * I)
        dH = (hp * I) - (hcr * H) - (pc * H)
        dC = (pc * H) - (pd * C) - (pcr * C)
        dD = (pd * C)
        dR = (gamma * I) + (hcr * H) + (pcr * C)

        dCT = sigma * E
        dCH = hp * I

        return dS, dE, dI, dR, dH, dC, dD, dCT, dCH

    def predict(self, duration, initial_state=None, parameters=None):

        # Time vector:
        time = np.arange(duration)
        # Parameters to use
        prm = parameters
        if prm is None:
            prm = self.get_parameters()
        # Initial state to use:
        init = initial_state
        if init is None:
            init = self.get_initial_state()

        # Make prediction:
        predict = odeint(func=self.differential,
                         y0=init,
                         t=time,
                         args=(tuple(prm)))
        return predict

    def fit(self):

        # Initial values of parameters:
        init_prm = (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr, self.s, self.t)
        # Initial state:
        init_state = self.get_initial_state()
        # Time vector:
        time = self.dataset[:, 0]
        # Bounds
        bds = [(self.beta_min, self.beta_max), (self.sigma_min, self.sigma_max), (self.gamma_min, self.gamma_max),
               (self.hp_min, self.hp_max), (self.hcr_min, self.hcr_max), (self.pc_min, self.pc_max),
               (self.pd_min, self.pd_max), (self.pcr_min, self.pcr_max), (self.s_min, self.s_max),
               (self.t_min, self.t_max)]
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
        if self.LBFGSB:
            res = minimize(self.objective, np.asarray(init_prm),
                           method='L-BFGS-B',
                           args=('method_1'),
                           bounds=bds)
        else:
            if self.cobyla:
                res = minimize(self.objective, np.asarray(init_prm),
                               method='COBYLA',
                               args=('method_1'),
                               constraints=cons)
            elif self.slsqp:
                res = minimize(self.objective, np.asarray(init_prm),
                               method='SLSQP',
                               args=('method_1'),
                               options={'eps': 0.05},
                               constraints=cons)
            else:   # Auto
                res = minimize(self.objective, np.asarray(init_prm),
                               constraints=cons,
                               options={'eps': self.opti_step},
                               bounds=bds)


        # Print optimizer result
        # print(res)
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

    def objective(self, parameters, method, print_details=False):

        if method == 'method_1':
            # Here we try to maximise the probability of each observations
            # Make predictions:
            params = tuple(parameters)
            pred = self.predict(duration=self.dataset.shape[0],
                                parameters=params)
            # Uncumul contaminations
            conta = []
            conta.append(pred[0][7])
            for i in range(0, pred.shape[0]):
                conta.append(pred[i][7] - pred[i-1][7])


            # Compare with dataset:
            prb = 0
            # print(params)

            for i in range(0, pred.shape[0]):
                p_k1 = p_k2 = p_k3 = p_k4 = p_k5 = p_k6 = self.overflow
                # ======================================= #
                # PART 1: Fit on cumul positive test
                # ======================================= #
                pa = params[8] * params[9]
                n = np.around(pred[i][7] * pa)
                k = self.dataset[i][7]

                p = 1 / self.b_s_1
                if k < 0 and n < 0:   # a vérifer k<0

                    k *= -1
                    n *= -1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k < 0:
                    n += k + 1
                    k = 1
                n *= self.b_s_1
                prob = binom.pmf(k=k, n=n, p=p)
                if prob > 0:
                    p_k1 = np.log(binom.pmf(k=k, n=n, p=p))
                else:
                    p_k1 = self.overflow
                prb -= p_k1 * self.w_1

                # ======================================= #
                # PART 2: Fit on positive test
                # ======================================= #
                pa = params[8] * params[9]
                n = np.around(conta[i] * pa)
                k = self.dataset[i][1]
                p = 1 / self.b_s_2
                if k < 0 and n < 0:
                    print("hello 2")
                    k *= -1
                    n *= -1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k < 0:
                    n += k + 1
                    k = 1
                n *= self.b_s_2
                prob = binom.pmf(k=k, n=n, p=p)
                if prob > 0:
                    p_k2 = np.log(binom.pmf(k=k, n=n, p=p))
                else:
                    p_k2 = self.overflow
                prb -= p_k2 * self.w_2

                # ======================================= #
                # PART 3: Fit on hospit
                # ======================================= #
                n = np.around(pred[i][4])
                k = self.dataset[i][3]
                p = 1 / self.b_s_3
                if k < 0 and n < 0:

                    k *= -1
                    n *= -1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k < 0:

                    n += k + 1
                    k = 1
                n *= self.b_s_3
                prob = binom.pmf(k=k, n=n, p=p)
                if prob > 0:
                    p_k3 = np.log(binom.pmf(k=k, n=n, p=p))
                else:
                    p_k3 = self.overflow
                prb -= p_k3 * self.w_3

                # ======================================= #
                # PART 4: Fit on cumul hospit
                # ======================================= #
                n = np.around(pred[i][8])
                k = self.dataset[i][4]
                p = 1 / self.b_s_4
                if k < 0 and n < 0:

                    k *= -1
                    n *= -1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k < 0:

                    n += k + 1
                    k = 1
                n *= self.b_s_4
                prob = binom.pmf(k=k, n=n, p=p)
                if prob > 0:
                    p_k4 = np.log(binom.pmf(k=k, n=n, p=p))
                else:
                    p_k4 = self.overflow
                prb -= p_k4 * self.w_4

                # ======================================= #
                # Part 5: Fit on Critical
                # ======================================= #
                n = np.around(pred[i][5])
                k = self.dataset[i][5]
                p = 1 / self.b_s_5

                if k < 0 and n < 0:

                    k *= -1
                    n *= -1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k < 0:

                    n += k + 1
                    k = 1
                n *= self.b_s_5
                prob = binom.pmf(k=k, n=n, p=p)
                if prob > 0:
                    p_k5 = np.log(binom.pmf(k=k, n=n, p=p))
                else:
                    p_k5 = self.overflow
                prb -= p_k5 * self.w_5

                # ======================================= #
                # Part 6: Fit on Fatalities
                # ======================================= #
                n = np.around(pred[i][6])
                k = self.dataset[i][6]
                p = 1 / self.b_s_6
                if k < 0 and n < 0:

                    k *= -1
                    n *= -1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k < 0:

                    n += k + 1
                    k = 1
                n *= self.b_s_6
                prob = binom.pmf(k=k, n=n, p=p)
                if prob > 0:
                    p_k6 = np.log(binom.pmf(k=k, n=n, p=p))
                else:
                    p_k6 = self.overflow
                prb -= p_k6 * self.w_6

                if print_details:
                    print('iter {}: {} - {} - {} - {} - {} - {}'.format(i, p_k1, p_k2, p_k3, p_k4, p_k5, p_k6))
                    print('test+ cumul: {} - {}'.format(np.around(pred[i][7] * params[8] * params[9]), self.dataset[i][7]))
                    print('test+: {} - {}'.format(np.around(conta[i] * params[8] * params[9]), self.dataset[i][1]))
                    print('hospit: {} - {}'.format(np.around(pred[i][4]), self.dataset[i][3]))
                    print('hospit cumul: {} - {}'.format(np.around(pred[i][8]), self.dataset[i][4]))
                    print('critical: {} - {}'.format(np.around(pred[i][5]), self.dataset[i][5]))
                    print('Fatalities: {} - {}'.format(np.around(pred[i][6]), self.dataset[i][6]))

            # print(prb)

            return prb

    def import_dataset(self):

        url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
        # Import the dataframe:
        raw = pd.read_csv(url, sep=',', header=0)
        raw['num_positive'][0] = 1
        # Ad a new column at the end with cumulative positive cases at the right
        cumul_positive = raw['num_positive'].to_numpy()
        raw.insert(7, 'cumul_positive', cumul_positive)
        if self.smoothing:
            self.dataframe = dataframe_smoothing(raw)
        else: self.dataframe = raw
        self.dataset = self.dataframe.to_numpy()

        self.I_0 = int(self.dataset[0][1] / (self.s * self.t))
        self.E_0 = int(self.I_0 * 5)
        self.R_0 = int(0)
        self.S_0 = int(1000000 - self.I_0 - self.E_0)
        self.H_0 = 0
        self.C_0 = 0
        self.D_0 = 0
        self.CT_0 = self.I_0  # Contamined
        self.CH_0 = int(self.dataset[0][4])

    def sensib_finder(self):

        range_size = 50
        sensib_range = np.linspace(0.1, 1, range_size)
        print(sensib_range)

        accuracies = []

        init_state = self.get_initial_state()
        best = (math.inf, 0)

        for j in range(0, range_size):

            self.s = sensib_range[j]
            self.fit()

            # Compute error:
            error = 0.0
            params = (self.beta, self.sigma, self.gamma, self.s)
            pred = self.predict(duration=self.dataset.shape[0],
                                parameters=params)
            # Compare with dataset:
            prb = 0
            print(params)
            for i in range(0, pred.shape[0]):
                p = params[-1]
                p /= 2
                n = round(pred[i][4] * 2)
                k = round(self.dataset[i][1])


                p_k = np.log(binom.pmf(k=k, n=n, p=p))
                if p_k == - math.inf or math.isnan(p_k):
                    p_k = -1000
                prb += p_k
            accuracies.append(-prb)
            if -prb < best[0]:
                best = (-prb, sensib_range[j])
            print('iter {} / {}'.format(j, range_size))

        plt.plot(sensib_range, accuracies, c='blue')
        plt.show()
        print('best sensib = {} with error = {}'.format(best[1], best[0]))
        self.s = best[1]

    def plot(self, filename, type, duration=0, plot_conf_inter=False, global_view=False):
        plot_dataset(self, filename , type, duration, plot_conf_inter, global_view)


if __name__ == "__main__":

    # Create the model:
    model = SEIR()

    # Fit:
    # model.fit()

    # params = model.get_parameters()
    # #model.objective(params, 'method_1', print_details=True)
    #
    # # Make a prediction:
    # prd = model.predict(model.dataset.shape[0])
    # for i in range(0, prd.shape[0]):
    #     prd[i][3] = prd[i][3] * model.s * model.t
    #
    # print('=== For cumul positif: ')
    # for i in range(0, 10):
    #     print('dataset: {}, predict = {}'.format(model.dataset[i, 7], prd[i][7]))
    # print('=== For hospit: ')
    # for i in range(0, 10):
    #     print('dataset: {}, predict = {}'.format(model.dataset[i, 3], prd[i][4]))
    # print('===  E values: ')
    # print(prd[:, 1])

    model.plot('testtest.pdf',
               '--sto-I --sto-E --sto-H --sto-C --sto-F' +
               '--det-I --det-E --det-H --det-C --det-F' ,
               duration=200,
               plot_conf_inter=True)

    model.plot('testtest2.pdf',
               '--sto-S --sto-R --det-S --det-R',
               duration=200,
               plot_conf_inter=True)
