import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from scipy.signal import savgol_filter
from scipy.stats import binom as binomS
from scipy import optimize
from scipy.special import binom as binomP



class SEIR():

    def __init__(self):

        # Model parameters
        self.beta = 0.1     # Contamination rate
        self.sigma = 0.7        # Incubation rate
        self.gamma = 0.22725    # Recovery rate
        self.hp = 0.05          # Hospit rate
        self.hcr = 0.2          # Hospit recovery rate
        self.pc = 0.1           # Critical rate
        self.pd = 0.2           # Critical recovery rate
        self.pcr = 0.3          # Critical mortality
        self.s = 0.765          # Sensitivity
        self.t = 0.75           # Testing rate in symptomatical

        # Learning set
        self.dataframe = None
        self.dataset = None

        # Initial state
        self.I_0 = 2                        # Infected
        self.E_0 = 3                        # Exposed
        self.R_0 = 0                        # Recovered
        self.S_0 = 1000000 - self.I_0 - self.E_0      # Sensible
        self.H_0 = 0
        self.C_0 = 0
        self.D_0 = 0
        self.CT_0 = self.I_0                # Contamined

    def get_parameters(self):

        prm = (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr, self.s, self.t)
        return prm

    def get_initial_state(self):

        I_0 = self.dataset[0][1]
        H_0 = self.dataset[0][3]
        E_0 = I_0
        D_0 = 0
        C_0 = 0
        S_0 = 1000000 - I_0 - H_0 - E_0
        R_0 = 0
        CT_0 = I_0
        CH_0 = H_0
        init = (S_0, E_0, I_0, R_0, H_0, C_0, D_0, CT_0, CH_0)
        return init

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr, s, t):

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
        # Constraint on parameters:
        cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + 0.5},  # Beta max
                {'type': 'ineq', 'fun': lambda x: -x[1] + 1},  # Maximal sigma value
                {'type': 'ineq', 'fun': lambda x: -x[2] + 1 / 3},  # Max gamma value
                {'type': 'ineq', 'fun': lambda x: -x[3] + 0.2},  # Hospitalization rate
                {'type': 'ineq', 'fun': lambda x: -x[4] + 0.5},  # Hospital recovery rate max
                {'type': 'ineq', 'fun': lambda x: -x[5] + 0.2},  # Proba max de critical
                {'type': 'ineq', 'fun': lambda x: -x[6] + 0.5},  # Mortality max
                {'type': 'ineq', 'fun': lambda x: -x[7] + 0.5},  # Recover rate from critical max
                {'type': 'ineq', 'fun': lambda x: -x[8] + 0.85},  # Max sensitivity
                {'type': 'ineq', 'fun': lambda x: -x[9] + 0.9},  # Max testing rate
                {'type': 'ineq', 'fun': lambda x: x[0] - 0.1},
                {'type': 'ineq', 'fun': lambda x: x[1] - 0.3},  # Min sigma value
                {'type': 'ineq', 'fun': lambda x: x[2] - 1 / 7},  # Min gamma value
                {'type': 'ineq', 'fun': lambda x: x[3] - 0.01},  # Min hospit rate
                {'type': 'ineq', 'fun': lambda x: x[4] - 0.01},  # Min hospital recovery rate
                {'type': 'ineq', 'fun': lambda x: x[5] - 0.01},  # Proba min de critical
                {'type': 'ineq', 'fun': lambda x: x[6] - 0.1},  # Mortality min
                {'type': 'ineq', 'fun': lambda x: x[7] - 0.0},  # Min recover rate from critical
                {'type': 'ineq', 'fun': lambda x: x[8] - 0.70},  # Min sensitivity
                {'type': 'ineq', 'fun': lambda x: x[9] - 0.5})  # Min testing rate

        """
                cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + 0.5},     # Beta max
                {'type': 'ineq', 'fun': lambda x: -x[1] + 1},       # Maximal sigma value
                {'type': 'ineq', 'fun': lambda x: -x[2] + 1/3},     # Max gamma value
                {'type': 'ineq', 'fun': lambda x: -x[3] + 0.2},     # Hospitalization rate
                {'type': 'ineq', 'fun': lambda x: -x[4] + 0.5},     # Hospital recovery rate max
                {'type': 'ineq', 'fun': lambda x: -x[5] + 0.2},     # Proba max de critical
                {'type': 'ineq', 'fun': lambda x: -x[6] + 0.5},     # Mortality max
                {'type': 'ineq', 'fun': lambda x: -x[7] + 0.5},     # Recover rate from critical max
                {'type': 'ineq', 'fun': lambda x: -x[8] + 0.85},    # Max sensitivity
                {'type': 'ineq', 'fun': lambda x: -x[9] + 0.8},       # Max testing rate
                {'type': 'ineq', 'fun': lambda x: x[0] - 0.1},
                {'type': 'ineq', 'fun': lambda x: x[1] - 0.8},      # Min sigma value
                {'type': 'ineq', 'fun': lambda x: x[2] - 1/7},      # Min gamma value
                {'type': 'ineq', 'fun': lambda x: x[3] - 0.01},     # Min hospit rate
                {'type': 'ineq', 'fun': lambda x: x[4] - 0.01},      # Min hospital recovery rate
                {'type': 'ineq', 'fun': lambda x: x[5] - 0.01},     # Proba min de critical
                {'type': 'ineq', 'fun': lambda x: x[6] - 0.0},      # Mortality min
                {'type': 'ineq', 'fun': lambda x: x[7] - 0.0},      # Min recover rate from critical
                {'type': 'ineq', 'fun': lambda x: x[8] - 0.70},     # Min sensitivity
                {'type': 'ineq', 'fun': lambda x: x[9] - 0.5})      # Min testing rate
        """

        # Optimizer
        res = minimize(self.objective, np.asarray(init_prm), constraints=cons, # method='COBYLA',
                       args=('method_3'),
                       options={'maxiter': 20000})



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


    def fit_visual(self):

        initial_state = self.S_0, self.I_0, self.R_0

        range_size = 100
        beta_range = np.linspace(0, 1, range_size)
        gamma_range = np.linspace(0, 1, range_size)

        best = (math.inf, 0, 0)
        error_map = np.zeros((range_size, range_size))
        for b in range(0, len(beta_range)):
            for g in range(0, len(gamma_range)):
                params = (beta_range[b], gamma_range[g], self.s)
                error = self.objective(params, 'method_2')
                error_map[b][g] = error
                if error < best[0]:
                    best = (error, beta_range[b], gamma_range[g])
            print('iter {} / {}'.format(b+1, range_size))
        print('best value: error = {}, beta = {}, gamma = {}'.format(best[0], best[1], best[2]))
        X, Y = np.meshgrid(beta_range, gamma_range)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, error_map)
        #ax.set_zscale('log')
        ax.view_init(15, 60)
        plt.show()

        self.beta = best[1]
        self.gamma = best[2]

    def objective(self, parameters, method):

        if method == 'method_1':

            # Make predictions:
            pred = self.predict(duration=self.dataset.shape[0],
                                parameters=parameters)
            # Compare with dataset:
            error = 0.0
            for i in range(0, pred.shape[0]):
                error += (self.dataset[i][1] - pred[i][3] * parameters[2]) ** 2
            print(error)
            return error

        if method == 'method_2':
            # Here we try to maximise the probability of each observations
            # Make predictions:
            params = (parameters[0], parameters[1], parameters[2], self.s)
            pred = self.predict(duration=self.dataset.shape[0],
                                parameters=params)
            # Compare with dataset:
            prb = 0
            #print(params)
            for i in range(0, pred.shape[0]):
                p = params[-1]
                n = round(pred[i][3])
                k = round(self.dataset[i][1])
                if p < 0.1:
                    prb += - 1000
                    continue
                if k == 0:
                    k = 1
                if n == 0:
                    n = 1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                p_k = np.log(binomS.pmf(k=k, n=n, p=p))
                if p_k == - math.inf or math.isnan(p_k):
                    p_k = -1000
                prb += p_k
            print(-prb)
            return -prb

        if method == 'method_3':
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
            print(params)
            for i in range(0, pred.shape[0]):

                # PART 1: Fit on cumul positive test
                p = params[8] * params[9] * 0.5
                n = pred[i][7] * 2
                k = self.dataset[i][7]

                p_k = 0.0
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k == 0 or n == 0:
                    p_k = - 1000
                else:
                    p_k = np.log(binomS.pmf(k=k, n=n, p=p))
                if p_k == - math.inf or math.isnan(p_k):
                    p_k = - 1000
                prb -= p_k *2

                # PART 2: Fit on non_cumul positive test
                p = params[8] * params[9] * 0.5
                n = conta[i] * 2
                k = self.dataset[i][1]

                p_k = 0.0
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k == 0 or n == 0:
                    p_k = - 1000
                else:
                    p_k = np.log(binomS.pmf(k=k, n=n, p=p))
                if p_k == - math.inf or math.isnan(p_k):
                    p_k = -400
                prb -= p_k *2

                # PART 3: Fit on hospit
                n = pred[i][4] * 5
                k = self.dataset[i][3]
                p = 0.2

                p_k = 0.0
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k == 0 or n == 0:
                    p_k = - 1000
                else:
                    p_k = np.log(binomS.pmf(k=k, n=n, p=p))
                if p_k == - math.inf or math.isnan(p_k):
                    p_k = -1000
                prb -= p_k * 2


                # PART 4: Fit on hospit Cumul
                n = pred[i][8] * 5
                k = self.dataset[i][4]
                p = 0.2

                p_k = 0.0
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k == 0 or n == 0:
                    p_k = - 1000
                else:
                    p_k = np.log(binomS.pmf(k=k, n=n, p=p))
                if p_k == - math.inf or math.isnan(p_k):
                    p_k = -1000
                prb -= p_k

                # Part 5: Fit on Critical
                n = pred[i][5] * 5
                k = self.dataset[i][5]
                p = 0.2

                p_k = 0.0
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k == 0 or n == 0:
                    p_k = - 1000
                else:
                    p_k = np.log(binomS.pmf(k=k, n=n, p=p))
                if p_k == - math.inf or math.isnan(p_k):
                    p_k = -1000
                prb -= p_k

                # Part 5: Fit on Fatalities
                n = pred[i][6] * 5
                k = self.dataset[i][6]
                p = 0.2

                p_k = 0.0
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k == 0 or n == 0:
                    p_k = - 1000
                else:
                    p_k = np.log(binomS.pmf(k=k, n=n, p=p))
                if p_k == - math.inf or math.isnan(p_k):
                    p_k = -1000
                prb -= p_k

            print(prb)

            return prb


    def dataframe_smoothing(self, df):
        # From andreas NRMAS
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
        # Perform smoothing for each attributes
        for i in range(1, np_df.shape[1]):
            smoothed[:, i] = own_NRMAS(np_df[:, i], 7)

        # Write new values in a dataframe
        new_df = pd.DataFrame(smoothed, columns=df.columns)

        return new_df

    def import_dataset(self):

        url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
        # Import the dataframe:
        raw = pd.read_csv(url, sep=',', header=0)
        raw['num_positive'][0] = 1
        # Ad a new column at the end with cumulative positive cases at the right
        cumul_positive = raw['num_positive'].to_numpy()
        raw.insert(7, 'cumul_positive', cumul_positive)
        self.dataframe = self.dataframe_smoothing(raw)
        self.dataset = self.dataframe.to_numpy()

        self.I_0 = self.dataset[0][1] / self.s
        self.E_0 = self.I_0 * 5
        self.R_0 = 0
        self.S_0 = 1000000 - self.I_0 - self.E_0

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


                p_k = np.log(binomS.pmf(k=k, n=n, p=p))
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



def own_NRMAS_index(vector, window, index):
    smoothed_value = 0
    nb_considered_values = 0
    max_size = (window - 1) / 2
    smoothing_window = np.arange(-max_size, max_size + 1, 1)

    for j in range(window):

        sliding_index = int(index + smoothing_window[j])

        if (sliding_index >= 0) and (sliding_index <= len(vector) - 1):
            smoothed_value += vector[sliding_index]
            nb_considered_values += 1

    return smoothed_value / nb_considered_values


def own_NRMAS(vector, window):
    smoothed_vector = np.zeros(len(vector))

    if (window % 2) == 0:
        print("Error window size even")
        return

    for i in range(len(vector)):
        smoothed_vector[i] = own_NRMAS_index(vector, window, i)

    return smoothed_vector


if __name__ == "__main__":

    # Create the model:
    model = SEIR()
    # Import dataset:
    model.import_dataset()

    # Fit:
    model.fit()

    # Make a prediction:
    prd = model.predict(model.dataset.shape[0])
    for i in range(0, prd.shape[0]):
        prd[i][3] = prd[i][3] * model.s * model.t

    print('=== For cumul positif: ')
    for i in range(0, 10):
        print('dataset: {}, predict = {}'.format(model.dataset[i, 7], prd[i][7]))
    print('=== For hospit: ')
    for i in range(0, 10):
        print('dataset: {}, predict = {}'.format(model.dataset[i, 3], prd[i][4]))
    print('===  E values: ')
    print(prd[:, 1])

    # Plot
    plt.scatter(model.dataset[:, 0], model.dataset[:, 7], c='blue', label='testing data')
    plt.scatter(model.dataset[:, 0], model.dataset[:, 3], c='green', label='hospit')
    plt.plot(model.dataset[:, 0], prd[:, 4], c='yellow', label='hospit pred')
    plt.plot(model.dataset[:, 0], prd[:, 7], c='red', label='predictions')
    plt.legend()
    plt.show()




