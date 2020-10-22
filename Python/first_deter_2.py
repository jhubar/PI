import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import requests # to dowload csv file in github
from numpy import asarray
from numpy import savetxt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"

# Import datas from github
# Trouver un moyen d'importer le fichier à partir de github

class SIR_model():

    def __init__(self):
        """
        Init the model
        """
        self.beta = 0
        self.gamma = 0

    def set_beta(self, beta_value):
        """
        Manualy define the value of beta
        """
        self.beta = beta_value

    def set_gamma(self, gamma_value):
        """
        Manually define the value of set_gamma
        """
        self.gamma = gamma_value

    def fit(self, dataset, beta_min, beta_max, gamma_min, gamma_max, range_size):
        """
        Find optimal value of beta and gamma parameters for the given dataset
        """
        # Make cumulative observed data
        # Revient à avoir I + R
        cumul_data = []
        tmp = 0
        for item in dataset[:, 1]:
            tmp += item
            cumul_data.append(tmp)

        beta_interval = (beta_max - beta_min) / range_size
        gamma_interval = (gamma_max - gamma_min) / range_size
        beta_range = [beta_interval + beta_min]
        gamma_range = [gamma_interval + gamma_min]
        for i in range(1, range_size):
            beta_range.append(beta_range[i-1] + beta_interval)
            gamma_range.append(gamma_range[i-1] + gamma_interval)
        # Create SSE matrix:
        MSE = np.zeros((range_size, range_size))

        best_tuple = (9999999, 0, 0)

        # Make simulations:
        for b in range(0, range_size):         # Navigate in beta
            for g in range(0, range_size):      # Navigate in gamma
                S, I, R, t = self.predict(999999, 1, 0, dataset[0][0], dataset[len(cumul_data)-1][0], beta_range[b], gamma_range[g])
                # Make cumul:
                SI = []
                for i in range(0, len(I)):
                    SI.append(I[i] + R[i])
                # Make mean square error:
                sse = 0.0
                for i in range(0, len(cumul_data)):
                    sse += (SI[i] - cumul_data[i])**2
                MSE[b][g] = sse / len(cumul_data)
                # is best?
                if MSE[b][g] <= best_tuple[0]:
                    best_tuple = (MSE[b][g], b, g)
            print("test {} sur {}".format(b, range_size))

        print("Minimal value")
        print("SEE = {}".format(best_tuple[0]))
        print("beta = {}".format(beta_range[best_tuple[1]]))
        print("gamma = {}".format(gamma_range[best_tuple[2]]))

        X, Y = np.meshgrid(beta_range, gamma_range)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, MSE)
        ax.view_init(15, 60)
        plt.show()

        pass

    def fit_beta(self, dataset, beta_min, beta_max, range_size):
        # Make cumulative observed data
        # Revient à avoir I + R
        cumul_data = []
        tmp = 0
        for item in dataset[:, 1]:
            tmp += item
            cumul_data.append(tmp)

        beta_interval = (beta_max - beta_min) / range_size
        beta_range = [beta_interval + beta_min]
        for i in range(1, range_size):
            beta_range.append(beta_range[i-1] + beta_interval)
        # Create SSE vector:
        SSE = []

        optimal_beta = (99999999999999, 0)
        for b in range(0, range_size):
            S, I, R, t = self.predict(999999, 1, 0, dataset[0][0], dataset[len(cumul_data)-1][0], beta_range[b])
            tmp_sse = 0
            for i in range(0, len(cumul_data)):
                tmp_sse += (I[i] - cumul_data[i])**2
            SSE.append(tmp_sse)
            if tmp_sse < optimal_beta[0]:
                optimal_beta = (tmp_sse, beta_range[b])
        self.beta = optimal_beta[1]
        plt.plot(beta_range, SSE)
        plt.show()
        print("optimal value of bet = {} with SSE of {}".format(optimal_beta[1], optimal_beta[0]))

    def fit_gamma(self, dataset, gamma_min, gamma_max, range_size):
        # Make cumulative observed data
        # Revient à avoir I + R
        data = []
        tmp = 0
        for item in dataset[:, 1]:
            data.append(item)

        gamma_interval = (gamma_max - gamma_min) / range_size
        gamma_range = [gamma_min]
        for i in range(1, range_size):
            gamma_range.append(gamma_range[i-1] + gamma_interval)
        # Create SSE vector:
        SSE = []

        optimal_gamma = (9999999, 0)
        for b in range(0, range_size):
            contaminations = self.predict(999999, 1, 0, dataset[0][0], dataset[len(data)-1][0], gamma=gamma_range[b], show_conta=True)
            tmp_sse = 0
            for i in range(0, len(contaminations)):
                tmp_sse += (contaminations[i] - data[i])**2
            SSE.append(tmp_sse)
            if tmp_sse < optimal_gamma[0]:
                optimal_gamma = (tmp_sse, gamma_range[b])
        self.gamma = optimal_gamma[1]
        plt.plot(gamma_range, SSE)
        plt.show()
        print("optimal value of gamma = {} with SSE of {}".format(optimal_gamma[1], optimal_gamma[0]))

    def predict(self, S0, I0, R0, t0, t1, beta=-1, gamma=-1, show_conta=False):
        # If beta or gama are given
        if beta == -1:
            beta_val = self.beta
        else:
            beta_val = beta
        if gamma == -1:
            gamma_val = self.gamma
        else:
            gamma_val = gamma
        # Repartition of population
        N = S0 + R0 + I0
        S = S0
        R = R0
        I = I0
        SS = [S0]
        RR = [R0]
        II = [I0]
        tt = [t0]
        t = t0
        contaminations = []
        while t <= t1:
            if I > 0.0001:
                dS = float(-(beta_val * S * I)/N)
                dR = float(gamma_val * I)
                dI = beta_val * S * I / N - gamma_val*I
                contaminations.append((beta_val * S * I)/N)
            else:
                dS = 0
                dR = 0
                dI = 0
            S += dS
            I += dI
            R += dR
            SS.append(S)
            II.append(I)
            RR.append(R)
            t += 1
            tt.append(t)
        if show_conta:
            return contaminations

        return SS, II, RR, tt

def covid_20():
    """
    Fit and predict on covid 20
    """
    # Import datas
    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
    data = pd.read_csv(url, sep=',', header=0)
    print(data)
    data_matrix = data.to_numpy()
    """
    make first sir pred:
    """
    # Store datas:
    t_0 = 0
    t_f = len(data_matrix[:, 0])
    I_0 = 1
    S_0 = 999999
    R_0 = 0
    model = SIR_model()

    model.fit_beta(data_matrix, beta_min=0.35, beta_max=0.55, range_size=200)
    model.fit_gamma(data_matrix, gamma_min=0, gamma_max=0.8, range_size=200)


    simul_conta_curve = model.predict(S_0, I_0, R_0, t_0, t_f-1, show_conta=True)
    plt.plot(data_matrix[:, 0], data_matrix[:, 1], c="red")
    plt.plot(data_matrix[:, 0], simul_conta_curve, c="green")
    plt.show()

    # Simulation sur le long terme:
    S, I, R, t = model.predict(S_0, I_0, R_0, t_0, 100)
    plt.plot(t, R, c="black")
    plt.plot(t, S, c="blue")
    plt.plot(t, I, c="red")
    plt.show()

def covid_19():
    """
    Test sur les données belges
    """
    df = pd.read_csv("https://epistat.sciensano.be/Data/COVID19BE_CASES_AGESEX.csv", sep=',')
    # Return a serie pandas with cases per date
    cases = df.groupby("DATE")["CASES"].sum()
    cases_np = cases.values
    time = []
    cases = []
    for i in range(0, len(cases_np)):
        time.append(i)
        cases.append(cases_np[i])
    # Plot data evolution
    plt.plot(time, cases)
    plt.show()

    """
    Fiter the model on the 20 first days
    """
    dataset = np.zeros((20, 2))
    for i in range(0, 20):
        dataset[i][0] = i + 1
        dataset[i][1] = cases_np[i]


    model = SIR_model()
    model.fit_beta(dataset, beta_min=0.01, beta_max=0.9, range_size=1000)
    model.fit_gamma(dataset, gamma_min=0.01, gamma_max=0.8, range_size=1000)
    model.fit_beta(dataset, beta_min=0.01, beta_max=0.9, range_size=1000)
    model.fit_gamma(dataset, gamma_min=0.01, gamma_max=0.8, range_size=1000)
    model.fit_beta(dataset, beta_min=0.01, beta_max=0.9, range_size=1000)
    model.fit_gamma(dataset, gamma_min=0.01, gamma_max=0.8, range_size=1000)
    model.fit_beta(dataset, beta_min=0.01, beta_max=0.9, range_size=1000)
    model.fit_gamma(dataset, gamma_min=0.01, gamma_max=0.8, range_size=1000)


    # Compare mmodèle et données:
    predict = model.predict(11000000, dataset[0][1], 0, dataset[0][0], dataset[dataset.shape[0]-1][0], show_conta=True)

    plt.plot(dataset[:, 0], dataset[:, 1], c="green")
    plt.plot(dataset[:, 0], predict, c="red")
    plt.show()



    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--ni")
    parser.add_argument("--tw")
    

    args = parser.parse_args()

    if args.ni:
        covid_19()
    elif args.tw:
        covid_20()


    pass
