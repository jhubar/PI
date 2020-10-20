import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import requests # to dowload csv file in github
from numpy import asarray
from numpy import savetxt
from mpl_toolkits.mplot3d import Axes3D


url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"

# Import datas from github
# Trouver un moyen d'importer le fichier à partir de github

class SIR_model():

    def __init__(self):
        """
        Init the model
        """
        self.beta = 1
        self.gamma = 0.69

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

    def fit(self, dataset):
        """
        Find optimal value of beta and gamma parameters for the given dataset
        """
        positives = dataset[:, 1]
        t_zero = dataset[0][0]
        t_final = dataset[dataset.shape[0]-1][0]
        range_size = 100
        pop_size = 1000000
        # Creation d'une matrice avec toutes les valeurs possibles:
        SSE = np.zeros((range_size, range_size))
        gamma_range = []
        beta_range = []
        interval = 1 / range_size
        min = (99999999, 0, 0)
        value = 0
        for i in range(0, range_size):
            value += interval
            gamma_range.append(value)
            beta_range.append(value)

        for i in range(0, range_size):
            for j in range(0, range_size):
                # make predictions:
                DDI = self.predict_perso(pop_size, 1, 0, 1, dataset.shape[0] - 1, beta_range[i], gamma_range[j])

                for k in range(0, len(DDI)):
                    SSE[i][j] += (DDI[k] - dataset[k][1])**2
                if SSE[i][j] <= min[0]:
                    min = (SSE[i][j], i, j)

        #Export matrix:
        savetxt('see_matrix.csv', SSE, delimiter=",")

        X, Y = np.meshgrid(beta_range, gamma_range)
        # Z = SSE.reshape((1, range_size**2))
        print(X.shape)
        print(Y.shape)



        print("Minimal value")
        print("SEE = {}".format(min[0]))
        print("beta = {}".format(beta_range[min[1]]))
        print("gamma = {}".format(gamma_range[min[2]]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, SSE)
        plt.show()


        pass

    def predict(self, S0, I0, R0, t0, t1):
        N = S0 + R0 + I0;
        S = S0;
        R = R0;
        I = I0;
        SS = [S0];
        RR = [R0];
        II = [I0];
        tt = [t0];
        dt = 1;
        t = t0
        DDI = []
        DDI.append(0)
        while t <= t1:
            dS = -self.beta * S * I / N
            dI = self.beta * S * I / N - self.gamma * I
            dR = self.gamma * I
            DDI.append(self.beta * S * I / N)
            S = S + dt * dS;
            I = I + dt * dI;
            R = R + dt * dR;
            SS.append(S);
            II.append(I);
            RR.append(R)
            t = t + dt;
            tt.append(t)
        return (SS, II, RR, tt, DDI)

    def predict_perso(self, S0, I0, R0, t0, t1, beta, gamma):
        N = S0 + R0 + I0;
        S = S0;
        R = R0;
        I = I0;
        SS = [S0];
        RR = [R0];
        II = [I0];
        tt = [t0];
        dt = 1;
        t = t0
        DDI = []
        while t <= t1:
            dS = -beta * S * I / N
            dI = beta * S * I / N - gamma * I
            DDI.append(beta * S * I / N)
            dR = gamma * I
            S = S + dt * dS;
            I = I + dt * dI;
            R = R + dt * dR;
            SS.append(S);
            II.append(I);
            RR.append(R)
            t = t + dt;
            tt.append(t)
        return DDI

    def predict_old(self, S0, I0, R0, t0, t1):
        """
        SIR model:
        """
        N = S0 + R0 + I0
        S = S0
        R = R0
        I = I0
        # Output list
        SS = [S0]
        RR = [R0]
        II = [I0]
        tt = [t0]
        t = t0
        while t <= t1:
            dS = - (self.beta * (I * S)/N) * S
            dR = self.gamma * I
            dI = (-dS) - dR
            print("time {}, I = {}, S= {}, R={}".format(t, I, S, R))
            S = S + dS
            I = I + dI
            R = R + dR
            SS.append(S)
            II.append(I)
            RR.append(R)
            t = t + 1
            tt.append(t)
        return SS,II,RR,tt


if __name__ == "__main__":

    # Import datas
    data = pd.read_csv('git_data.csv', sep=',', header=0)
    data_matrix = data.to_numpy()
    """
    make first sir pred:
    """
    # Store datas:
    t_0 = 0
    t_f = 500
    I_0 = 1
    S_0 = 999999
    R_0 = 0
    model = SIR_model()
    # Make predictions:
    S, I, R, t, DDI = model.predict(S_0, I_0, R_0, t_0, t_f)

    plt.plot(t, S, c="green")
    plt.plot(t, I, c="red")
    plt.plot(t, R, c="blue")
    plt.show()

    plt.plot(t, DDI)
    plt.show()

    model.fit(data_matrix)

    for i in range(0, 25):
        print(DDI[i])
        print(data_matrix[i][1])


    pass














