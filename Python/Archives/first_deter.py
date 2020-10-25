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
        self.beta = 0.2
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

    def fit(self, dataset, beta_min=0, beta_max=1, gamma_min=0, gamma_max=1, range_size=100, pop_size=1000000):
        """
        Find optimal value of beta and gamma parameters for the given dataset
        """

        # Create a matrix to store all SSE values for each tested beta and gamma combinations
        SSE = np.zeros((range_size, range_size))
        # Set beta and gamma values to test
        gamma_range = []
        beta_range = []
        beta_interval = (beta_max - beta_min) / range_size
        gamma_interval = (gamma_max - gamma_min) / range_size
        # Store minimal value in tested values
        min = (99999999, 0, 0)
        tmp_beta = beta_min
        tmp_gamma = gamma_min
        #beta_range.append(tmp_beta)
        #gamma_range.append(tmp_gamma)
        # Fill the two vectors with values of beta and gamma to compute
        for i in range(0, range_size):
            tmp_beta += beta_interval
            tmp_gamma += gamma_interval
            gamma_range.append(tmp_gamma)
            beta_range.append(tmp_beta)
        # Performe each simulations
        for i in range(0, range_size):  # Beta pour les lignes
            for j in range(0, range_size):  # Gamma pour les colonnes
                # compute the number of people infected each days
                infect_in_day = self.predict(pop_size-1, 1, 0, 1, dataset.shape[0], conta_curve=True,
                                             beta=beta_range[i], gamma=gamma_range[j])
                for k in range(0, dataset.shape[0]):
                    SSE[i][j] += (infect_in_day[k] - dataset[k][1])**2
                if SSE[i][j] <= min[0]:
                    min = (SSE[i][j], i, j)
            # print(i)

        #Export matrix:
        savetxt('see_matrix.csv', SSE, delimiter=",")

        X, Y = np.meshgrid(beta_range, gamma_range)
        # Z = SSE.reshape((1, range_size**2))
        # print(X.shape)
        # print(Y.shape)



        print("Minimal value")
        print("SEE = {}".format(min[0]))
        print("beta = {}".format(beta_range[min[1]]))
        print("gamma = {}".format(gamma_range[min[2]]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, SSE)
        ax.view_init(15, 60)
        plt.show()

        # Find beta value with the best SEE for each gamma value
            # 1er colonne = valeur de gamma
            # 2e colonne = meilleure valeur de beta pour ce gamma
            # 3e col = SEE associé à cette combinaison beta gamma
        min_beta = np.zeros((len(gamma_range), 3))
        for i in range(0, len(gamma_range)):
            min_beta[i][0] = gamma_range[i]
            min_beta[i][2] = 99999999
            for j in range(0, len(beta_range)):
                if SSE[i][j] <= min_beta[i][2]:
                    min_beta[i][2] = SSE[i][j]
                    min_beta[i][1] = gamma_range[j]
        # plot results:
        plt.plot(min_beta[:, 0], min_beta[:, 1])
        plt.show()
        plt.plot(min_beta[:, 0], min_beta[:, 2])
        plt.show()
        for i in range(0, len(gamma_range)):
            print("Gamma= {}, best beta= {}, SEE={}".format(min_beta[i][0], min_beta[i][1], min_beta[i][2]))

        pass

    def predict(self, S0, I0, R0, t0, t1, conta_curve=False, beta=-1, gamma=-1):
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
        contaminations.append(0)
        while t <= t1:
            if I > 0.0001:
                dS = float(-(beta_val * S * I)/N)
                dR = float(gamma_val * I)
                dI = beta_val * S * I / N - gamma_val*I
                conta = beta_val * S * I / N
            else:
                dS = 0
                dR = 0
                dI = 0
                conta = 0
            contaminations.append(conta)
            S += dS
            I += dI
            R += dR
            SS.append(S)
            II.append(I)
            RR.append(R)
            t += 1
            tt.append(t)

        # If we want contamination curve:
        if conta_curve:
            return contaminations
        return SS, II, RR, tt




if __name__ == "__main__":

    # Import datas
    data = pd.read_csv('git_data.csv', sep=',', header=0)
    data_matrix = data.to_numpy()
    """
    make first sir pred:
    """
    # Store datas:
    t_0 = 0
    t_f = 100
    I_0 = 1
    S_0 = 99
    R_0 = 0
    model = SIR_model()
    # Make predictions:
    S, I, R, t = model.predict(S_0, I_0, R_0, t_0, t_f)
    DDI = model.predict(S_0, I_0, R_0, t_0, t_f, conta_curve=True)

    plt.plot(t, I, c="red")
    plt.plot(t, R, c="blue")
    plt.plot(t, S, c="green")
    plt.show()

    #model.fit(data_matrix, beta_min=0, beta_max=0.5, gamma_min=0.02, gamma_max=0.3, range_size=200)



    pass
