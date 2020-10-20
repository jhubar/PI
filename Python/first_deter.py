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
        self.beta = 0.5
        self.gamma = 0.5

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

    def SIR(self,S0,I0,R0,t0, t1):
        """
        SIR model:
        """
        N=S0+R0+I0
        S=S0; R=R0; I=I0
        SS=[S0];RR=[R0]; II=[I0]
        tt=[t0]
        dt=0.1
        t=t0
        while t <= t1:
            dS=-self.beta*S*I/N
            dI=self.beta*S*I/N-self.gamma*I
            dR=self.gamma*I
            S=S+dt*dS
            I=I+dt*dI
            R=R+dt*dR
            SS.append(S); II.append(I); RR.append(R)
            t=t+dt
            tt.append(t)
        return(SS,II,RR,tt)


if __name__ == "__main__":
    """
    make first sir pred:
    """
    model = SIR_model()

    pass












data = pd.read_csv('git_data.csv', sep=',', header=0)
print(data)
