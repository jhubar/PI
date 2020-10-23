import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array as ar

from mpl_toolkits.mplot3d import Axes3D


url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"

class SIR_model():

    def __init__(self):

        self.beta = 0.7578
        self.gamma = 0.6367

    def set_beta(self, beta_value):

        self.beta = beta_value

    def set_gamma(self, gamma_value):

        self.gamma = gamma_value

    def fit(self, df):

        gamma_range = np.linspace(0.001, 0.7, 100)
        beta_range = np.linspace(0.001, 0.8, 100)

        SSE = np.zeros((len(beta_range), len(gamma_range)))

        min_ms = (float('inf'), 0, 0)

        # test of beta and gamma
        for i in range(len(gamma_range)):
            for j in range(len(beta_range)):

                # make predictions:
                contaminations = self.predict(S0=0.99999,
                                              I0=0.00001,
                                              R0=0,
                                              t0=df['Day'][0],
                                              t1=len(df['Day']),
                                              conta_curve=True,
                                              beta=gamma_range[i],
                                              gamma=beta_range[j])
                # least mean square
                for k in range(len(contaminations)):
                    SSE[i][j] += (contaminations[k] - df['num_positive_prop'][k]) ** 2

                if(SSE[i][j] < min_ms[0]):
                    min_ms = (SSE[i][j], i, j)

        print("Minimal value")
        print("SEE = {}".format(min_ms[0]))
        print("beta = {}".format(beta_range[min_ms[1]]))
        print("gamma = {}".format(gamma_range[min_ms[2]]))
        print("")

        # plot 3d graph
        fig = plt.figure(figsize=(25, 20))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(beta_range, gamma_range)
        ax.scatter(X, Y, SSE)
        ax.invert_xaxis()
        ax.set_title("Representation of least mean square error\n" +
                     " depending on the parameters beta and gamma",
                     fontsize=30)
        ax.set_xlabel('beta', fontsize=30)
        ax.set_ylabel('gamma', fontsize=30)
        ax.set_zlabel('Least mean square error', fontsize=30)
        fig.savefig("LMS.png")

        pass

    def predict(self, S0, I0, R0, t0, t1, conta_curve=False, beta=-1, gamma=-1):

        if (beta == -1):
            beta_ = self.beta
        else:
            beta_ = beta

        if (gamma == -1):
            gamma_ = self.gamma
        else:
            gamma_ = gamma


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

            if(I > 0.00000000001):
                dS = - beta_ * S * I / N
                dI = (beta_ * S * I / N) - gamma_ * I
                dR = gamma_ * I
                contaminations.append(beta_ * S * I / N)

            else:
                dS = 0
                dI = 0
                dR = 0
                contaminations.append(0)

            S += dS
            I += dI
            R += dR
            t += dt


            SS.append(S)
            II.append(I)
            RR.append(R)
            tt.append(t)

        if conta_curve:
            return contaminations

        return SS, II, RR, tt

    def plot_fitting(self, t, df, contaminations):

        fig = plt.figure(figsize=(25, 20))

        plt.plot(t, df['num_positive_prop'], label='num_positive_prop')
        plt.plot(t, contaminations, label='contaminations')
        
        plt.ylabel('Proportion of the population', fontsize=30)
        plt.xlabel('Time (days)', fontsize=30)
        plt.title('Daily contamined with beta=' + str(self.beta) 
                  + ' and gamma=' + str(self.gamma) + 'compared to our data',
                  fontsize=30)
        plt.legend(fontsize=30)
        fig.savefig('fitting.png')
        
        
    def plot_SIR(self, t, S, I, R):
        
        plt.clf()
        fig = plt.figure(figsize=(25, 20))
        plt.plot(t, S, label='susceptible')
        plt.plot(t, I, label='Infected')
        plt.plot(t, R, label='Recover')
        
        plt.ylabel('Proportion of the population', fontsize=30)
        plt.xlabel('Time (days)', fontsize=30)
        plt.title('SIR model with beta=' + str(self.beta) + ' and gamma=' + 
                  str(self.gamma), fontsize=30)
        plt.legend(fontsize=30)
        fig.savefig('SIR.png')

    def add_total_positive_atm(self, df, spreading_period):

        new_c = []

        for i in range(len(df['num_positive'])):

            tot = 0

            for j in range(0, int(spreading_period)):

                if(i - j >= 0):
                    tot += df['num_positive'][i-j]

            new_c.append(tot)

        df['Total_positive_atm'] = new_c

    def add_num_positive_prop(self, df, population):

        get_prop = lambda x: x / population
        df['num_positive_prop'] = df.num_positive.apply(get_prop)

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
    I_0 = 0.000001
    S_0 = 0.999999
    R_0 = 0

    model = SIR_model()
    
    # Make predictions:
    S, I, R, t = model.predict(S_0, I_0, R_0, t_0, t_f)
    model.plot_SIR(t, S, I, R)
    
    contaminations = model.predict(S_0, I_0, R_0, t_0, t_f, True)


    model.add_total_positive_atm(data, 1/model.gamma)
    model.add_num_positive_prop(data, 1000000)

    '''
    plt.scatter(data['Day'],data['Total_positive_atm'], c = "orange", label='Cumulative infected')
    plt.plot(data['Day'],data['num_positive'], label='Positives cases')
    # plt.plot(t, S, c="blue", label='Susceptible')
    plt.scatter(t, I, c="red", label='Infected')

    plt.plot(t, R, c="green", label='Recovered')
    plt.legend()
    plt.savefig('img/deter_ad.png')
    '''
    
    model.plot_fitting(t, data, contaminations)

    model.fit(data)

    pass
