import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from scipy.signal import savgol_filter
from scipy.stats import binom

class SEIR():

    def __init__(self):

        self.beta = 0.38709
        self.sigma = 0.35330
        self.gamma = 0.22725

        self.sensitivity = 1

        self.dataframe = None
        self.dataset = None

    def differential(self, state, time, beta, sigma, gamma, sensitivity):

        S, E, I, R, conta = state

        dS = -(beta * S * I) / (S + E + I + R)
        dE = (beta * S * I) / (S + E + I + R) - E * sigma
        dI = (E * sigma) - (gamma * I)
        dR = (gamma * I)

        dConta = (beta * S * I) / (S + E + I + R)

        return dS, dE, dI, dR, dConta

    def predict(self, time):

        params = (self.beta, self.sigma, self.gamma, self.sensitivity)

        initial_state = self.get_initial_state()

        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=params)
        return predict

    def stochastic_predic(self, time):

        output = np.zeros((len(time), 5))
        # Initial state:
        output[0][2] = self.dataset[0][1]
        output[0][1] = 4 * output[0][2]
        output[0][0] = 1000000 - 5 * output[0][2]
        N = 1000000

        params = (self.beta, self.sigma, self.gamma, self.sensitivity)

        for i in range(1, len(time)):

            S_to_E = np.random.binomial(output[i-1][2], self.beta * output[i-1][0] / N)
            E_to_I = np.random.binomial(output[i-1][1], self.sigma)
            I_to_R = np.random.binomial(output[i-1][2], self.gamma)

            # Update staes:
            output[i][0] = output[i-1][0] - S_to_E
            output[i][1] = output[i-1][1] + S_to_E - E_to_I
            output[i][2] = output[i-1][2] + E_to_I - I_to_R
            output[i][3] = output[i-1][3] + I_to_R
            output[i][4] = E_to_I * self.sensitivity

        return output

    def stochastic_mean(self, time, nb_simul):

        result_S = np.zeros((len(time), nb_simul))
        result_E = np.zeros((len(time), nb_simul))
        result_I = np.zeros((len(time), nb_simul))
        result_R = np.zeros((len(time), nb_simul))
        result_Conta = np.zeros((len(time), nb_simul))
        for i in range(0, nb_simul):

            pred = self.stochastic_predic(time)
            for j in range(0, len(time)):
                result_S[j][i] = pred[j][0]
                result_E[j][i] = pred[j][1]
                result_I[j][i] = pred[j][2]
                result_R[j][i] = pred[j][3]
                result_Conta[j][i] = pred[j][4]

        mean = np.zeros((len(time), 5))
        std = np.zeros((len(time), 5))
        for i in range(0, len(time)):
            mean[i][0] = np.mean(result_S[i, :])
            mean[i][1] = np.mean(result_E[i, :])
            mean[i][2] = np.mean(result_I[i, :])
            mean[i][3] = np.mean(result_R[i, :])
            mean[i][4] = np.mean(result_Conta[i, :])
            std[i][0] = np.std(result_S[i, :])
            std[i][1] = np.std(result_E[i, :])
            std[i][2] = np.std(result_I[i, :])
            std[i][3] = np.std(result_R[i, :])
            std[i][4] = np.std(result_Conta[i, :])


        return mean, std

    def get_initial_state(self):

        I_0 = 5
        E_0 = 10 * I_0
        R_0 = 0
        S_0 = 1000000 - I_0 - E_0 - R_0

        Conta_0 = I_0

        return S_0, E_0, I_0, R_0, Conta_0

    def fit(self):

        #Prefit
        #for i in range(0, 2):
        #    for j in range(0, 4):
        #        self.manual_fit(param_index=j, prt=True, method='method_3')

        start_values = [self.beta, self.sigma, self.gamma, self.sensitivity]
        bnd = [(0., 0.5), (0.3, 1.), (0., 0.5), (0.7, 0.85)]

        res = minimize(self.objective, np.asarray(start_values), args=('method_4'), bounds=bnd)

        print(res)

        self.beta = res.x[0]
        self.sigma = res.x[1]
        self.gamma = res.x[2]
        self.sensitivity = res.x[3]



    def manual_fit(self, param_index=0, method='method_2', prt=False, name='Fitting'):

        rg_size = 200
        param_rg = np.linspace(0, 1, rg_size)
        params = [self.beta, self.sigma, self.gamma, self.sensitivity]
        error = []
        best = (math.inf, 0)
        for i in range(0, rg_size):
            params[param_index] = param_rg[i]
            err = self.objective(tuple(params), method=method)
            error.append(err)
            if err < best[0]:
                best = (err, param_rg[i])
        if param_index == 0:
            self.beta = best[1]
        if param_index == 1:
            self.sigma = best[1]
        if param_index == 2:
            self.gamma = best[1]
        if param_index == 3:
            self.sensitivity = best[1]
        if prt:
            prm_name = ['beta', 'sigma', 'gamma', 'sensitivity']
            plt.plot(param_rg, error, c='blue', label=prm_name[param_index])
            plt.title('Fit {}, best value: {}'.format(prm_name[param_index], best[1]))
            plt.legend()
            plt.show()




    def objective(self, parameters, method):


        if method == 'method_1':
            # Make predictions
            time = self.dataset[:, 0]
            initial_state = self.get_initial_state()
            params = tuple(parameters)

            # Make predictions:
            predict = odeint(func=self.differential,
                             y0=initial_state,
                             t=time,
                             args=params)
            # Compute new cases
            test_pred = []
            test_pred.append(predict[0][4])
            for i in range(1, predict.shape[0]):
                test_pred.append(predict[i][4] - predict[i-1][4])
            for i in range(0, predict.shape[0]):
                test_pred[i] *= params[3]
            # Use sum of square
            sse = 0
            for i in range(0, predict.shape[0]):
                sse += (self.dataset[i][1] - predict[i][4]) ** 2
            return sse

        if method == 'method_2':
            # Use binomial

            # Make predictions
            time = self.dataset[:, 0]
            initial_state = self.get_initial_state()
            params = tuple(parameters)

            # Make predictions:
            predict = odeint(func=self.differential,
                             y0=initial_state,
                             t=time,
                             args=params)
            # Compute new cases
            test_pred = []
            test_pred.append(predict[0][4])
            for i in range(1, predict.shape[0]):
                test_pred.append(predict[i][4] - params[3] * predict[i - 1][4])

            proba = np.zeros(1, dtype=np.float64)
            for i in range(0, len(time)):
                k = self.dataset[i][1]
                n = int(test_pred[i])
                #print('k= {}, n= {}'.format(k/params[3], n))
                if k > n:
                    proba += 0
                else:
                    #proba += (math.factorial(k)/(math.factorial(k)*math.factorial(n-k))) * (params[3] ** k) * (1 - params[3]) ** (n - k)
                    proba += binom.pmf(k=k, n=n, p=params[3])
            print(- proba)
            return - proba

        if method == 'method_3':
            # on stochastic mean
            save_params = (self.beta, self.sigma, self.gamma, self.sensitivity)
            self.beta = parameters[0]
            self.sigma = parameters[1]
            self.gamma = parameters[2]
            self.sensitivity = parameters[3]

            pred_mean, pred_var = self.stochastic_mean(self.dataset[:, 1], 200)

            proba = 0.0
            for t in range(0, pred_mean.shape[0]):
                if pred_var[t][4] == 0:
                    pred_var[t][4] = 0.0001
                proba += (1 / (np.sqrt(pred_var[t][4]*2*math.pi))) * np.exp(((-1)/2) * ((pred_mean[t][4] - self.dataset[t][1]/self.sensitivity)**2)/pred_var[t][4])

            print(- proba)
            return - proba

        if method == 'method_4':
            # on stochastic mean
            save_params = (self.beta, self.sigma, self.gamma, self.sensitivity)
            self.beta = parameters[0]
            self.sigma = parameters[1]
            self.gamma = parameters[2]
            self.sensitivity = parameters[3]

            pred_mean, pred_var = self.stochastic_mean(self.dataset[:, 1], 200)

            sse = 0.0
            for t in range(0, pred_mean.shape[0]):
                sse += (self.dataset[t][1] - pred_mean[t][4])**2

            print(sse)
            return sse


    def inport_dataset(self):

        url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
        # Import the dataframe:
        raw = pd.read_csv(url, sep=',', header=0)
        raw['num_positive'][0] = 1
        self.dataframe = raw
        self.dataset = raw.to_numpy()



if __name__ == "__main__":

    model = SEIR()

    model.inport_dataset()

    model.fit()

    time = model.dataset[:, 0]

    predictions_mean, predictions_std = model.stochastic_mean(time, 200)

    data = model.dataset[:, 1]
    var = np.zeros(predictions_mean.shape[0])
    for i in range(len(var)):
        print(predictions_std[i][4])
        var[i] = predictions_mean[i][4] + predictions_std[i][4] ** 2

    plt.scatter(time, data, color='blue', label='original/sensit')
    plt.plot(time, predictions_mean[:, 4], c='red', label='predictions')
    plt.plot(time, var, c='orange', label='var')
    plt.legend()
    plt.show()



