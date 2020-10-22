import matplotlib.pyplot as plt
import matplotlib
from numpy import matrix
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score





# Import datas from github
# Trouver un moyen d'importer le fichier à partir de github

class SIR_model():

    def __init__(self):
        """
        Init the model
        """
        self.beta = 1
        self.gamma = 0.59

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

def add_total_positive_atm(df, spreading_period):

    new_c = []

    for i in range(len(df['num_positive'])):

        tot = 0

        for j in range(0, spreading_period ):

            if(i - j >= 0):
                tot += df['num_positive'][i-j]

        new_c.append(tot)

    df['Total_positive_atm'] = new_c

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"

    # Import datas
    data = pd.read_csv(url, sep=',', header=0)
    """
    make first sir pred:
    """
    # Store datas:
    t_0 = 0
    t_f = 16
    I_0 = 1
    S_0 = 999999
    R_0 = 0
    model = SIR_model()
    # Make predictions:
    S, I, R, t = model.predict(S_0, I_0, R_0, t_0, t_f)
    spreading_period = 14
    add_total_positive_atm(data, spreading_period)
    # plt.plot(t, S, c="green")
    plt.plot(data['Day'],data['Total_positive_atm'])
    plt.plot(t, I, c="red")
    # plt.plot(t, R, c="blue")
    # plt.show()
    plt.savefig('img/first_deter.png')

    # print(data)


    pass
