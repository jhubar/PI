import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import requests # to dowload csv file in github

url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"

# Import datas from github
# Trouver un moyen d'importer le fichier à partir de github

class SIR_model():

    def __init__(self):
        """
        Init the model
        """
        self.beta = 0.58
        self.gamma = 0.48

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
        N = S0 + R0 + I0;
        S = S0;
        R = R0;
        I = I0;
        SS = [S0];
        RR = [R0];
        II = [I0];
        tt = [t0];
        dt = 0.1;
        t = t0
        while t <= t1:
            dS = -self.beta * S * I / N
            dI = self.beta * S * I / N - self.gamma * I
            dR = self.gamma * I
            S = S + dt * dS;
            I = I + dt * dI;
            R = R + dt * dR;
            SS.append(S);
            II.append(I);
            RR.append(R)
            t = t + dt;
            tt.append(t)
        return (SS, II, RR, tt)

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
    """
    make first sir pred:
    """
    # Store datas:
    t_0 = 0
    t_f = 500
    I_0 = 1
    S_0 = 999
    R_0 = 0
    model = SIR_model()
    # Make predictions:
    S, I, R, t = model.predict(S_0, I_0, R_0, t_0, t_f)

    plt.plot(t, S, c="green")
    plt.plot(t, I, c="red")
    plt.plot(t, R, c="blue")
    plt.show()

    print(data)


    pass














