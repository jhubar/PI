import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array as ar


url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"

# Import datas from github
# Trouver un moyen d'importer le fichier à partir de github

class SIR_model():

    def __init__(self):
        """
        Init the model
        """
        self.beta = 5
        self.gamma = 4.8

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

    def fit(self, df):
        """
        Find optimal value of beta and gamma parameters for the given dataset
        """
        num_popositive = df['num_positive']

        t_zero = 0
        t_final = df.shape[0] - 1

        population = 1000000

        # Creation d'une matrice avec toutes les valeurs possibles:
        gamma_range = np.linspace(0., 1, 1000)
        beta_range = np.linspace(0, 1, 1000)

        SSE = np.zeros((len(beta_range), len(gamma_range)))

        min_ms = (float('inf'), 0, 0)

        for i in range(len(gamma_range)):
            for j in range(len(beta_range)):

                # make predictions:
                contaminations = self.predict_perso(S0=999999,
                                         I0=1,
                                         R0=0,
                                         t0=df['Day'][0],
                                         t1=len(df['Day']),
                                         beta=gamma_range[i],
                                         gamma=beta_range[j])
                
                for k in range(len(contaminations)):
                    SSE[i][j] += (contaminations[k] - df['num_positive'][k]) ** 2

                if(SSE[i][j] < min_ms[0]):
                    min_ms = (SSE[i][j], i, j)

        

        print("Minimal value")
        print("SEE = {}".format(min_ms[0]))
        print("beta = {}".format(beta_range[min_ms[1]]))
        print("gamma = {}".format(gamma_range[min_ms[2]]))

        pass

    def predict(self, S0, I0, R0, t0, t1):

        N = S0 + R0 + I0
        S = S0
        R = R0
        I = I0

        SS = [S0]
        RR = [R0]
        II = [I0]
        tt = [t0]

        dt = 1
        t = t0 + 1
        contaminations = [0]
        
        while (t <= t1):

            dS = -self.beta * S * I / N
            dI = self.beta * S * I / N - self.gamma * I
            dR = self.gamma * I
            
            S += dS
            I += dI
            R += dR
            t += dt

            contaminations.append(self.beta * S * I / N)
            SS.append(S)
            II.append(I)
            RR.append(R)
            tt.append(t)

        return SS, II, RR, tt, contaminations

    def predict_perso(self, S0, I0, R0, t0, t1, beta, gamma):

        N = S0 + R0 + I0
        S = S0
        R = R0
        I = I0

        SS = [S0]
        RR = [R0]
        II = [I0]
        tt = [t0]

        dt = 1
        t = t0 + 1
        contaminations = [0]

        while (t <= t1):

            dS = -beta * S * I / N
            dI = (beta * S * I / N) - gamma * I
            dR = gamma * I

            S = S + dt * dS
            I = I + dt * dI
            R = R + dt * dR
            t = t + dt

            contaminations.append(beta * S * I / N)
            SS.append(S)
            II.append(I)
            RR.append(R)
            tt.append(t)

        return contaminations

    def plot_model(df, contaminations):

        t = ar.array('i', range(0, df.shape[0]))

        plt.plot(t, df['num_positive'], label='num_positive')
        plt.plot(t, DDI, label='DDI')

        plt.legend()
        plt.show()



if __name__ == "__main__":

    # Import datas
    data = pd.read_csv(url, error_bad_lines=False)
    data_matrix = data.to_numpy()
    """
    make first sir pred:
    """
    # Store datas:
    t_0 = data['Day'][0]
    t_f = len(data['Day'])
    I_0 = 1
    S_0 = 999999
    R_0 = 0

    model = SIR_model()
    # Make predictions:
    S, I, R, t, contaminations = model.predict(S_0, I_0, R_0, t_0, t_f)

    a = contaminations

    plt.plot(t, S, c="green")
    plt.plot(t, I, c="red")
    plt.plot(t, R, c="blue")
    plt.show()

    plt.plot(t, data['num_positive'], label='num_positive')
    plt.plot(t, contaminations, label='contaminations')
    plt.legend()
    plt.show()

    model.fit(data)

    #model.plot_model(data, contaminations)

    pass














