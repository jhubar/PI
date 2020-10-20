import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import requests # to dowload csv file in github

url = "https://github.com/ADelau/proj0016-epidemic-data/blob/main/data.csv"

# Import datas from github
# Trouver un moyen d'importer le fichier à partir de github

class SIR_model():

    def __init__(self):
        """
        Init the model
        """
        self.beta = 0.01
        self.gamma = 0.04

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
        pass

    def predict(self, S0, I0, R0, t0, t1):
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
    """
    make first sir pred:
    """
    # Store datas:
    t_0 = 0
    t_f = 50
    I_0 = 1
    S_0 = 999
    R_0 = 0
    model = SIR_model()
    # Make predictions:
    S, I, R, t = model.predict(S_0, I_0, R_0, t_0, t_f)

    plt.plot(t, S, c="green")
    plt.show()
    plt.plot(t, I, c="red")
    plt.show()
    plt.plot(t, R, c="blue")
    plt.show()

    print(S)
    print(I)
    print(R)


    pass














