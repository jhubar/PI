
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math


class SEIR():

    def __init__(self):
        # Parameter's values
        self.beta = 0.57
        self.gamma = 0.2
        self.sigma = 0.2

    def differential_VERSION_SEIR(self, state, time, beta, gamma, sigma):
        """
        Differential equations of the model
        """
        S, E, I, R = state

        dS = -(beta * S * I) / (S + I + R)
        dE = (beta * S * I) / (S + I + R) - (E * sigma)
        dI = (E * sigma) - (gamma * I)
        dR = gamma * I

        return dS, dE, dI, dR

    def differential(self, state, time, beta, gamma, sigma):
        """
        Differential equations of the model
        """
        S, E, I, R = state

        dS = -(beta * S * I) / (S + I + R)
        dE = (beta * S * I) / (S + I + R) - (E * sigma)
        dI = (beta * S * I) / (S + I + R) - (gamma * I)
        dR = gamma * I

        return dS, dE, dI, dR

    def predict(self, S_0, E_0, I_0, R_0, duration):
        """
        Predict epidemic curves
        """
        # Initialisation vector:
        initial_state = (S_0, E_0, I_0, R_0)
        # Time vector:
        time = np.arange(duration)
        # Parameters vector
        parameters = (self.beta, self.gamma, self.sigma)
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=parameters)

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3])).T

    def fit_sequential(self, dataset, args="hospit", method='fit_on_R', range_size=2000):
        """
        Args:
            - hospit = fit on cumulative hospitalisations
            - show = plot
        Method:
            - Fit on R: try to fit the R compartiment
        """
        if "hospit" in args:
            time = dataset[:, 0]
            data = dataset[:, 3]

        # Initialize model's parameters
        self.beta = 0.5
        self.gamma = 0.5
        self.sigma = 0.5
        # Set initial state:
        initial_state = (1000000 - data[0], 0, data[0], 0)
        # Store the sum of each parameters variation
        dt = math.inf
        it = 0
        while it < 5:
            dt = 0
            # store precedent value of beta and fit beta
            beta_0 = self.beta
            self.fit_on_beta(time, data, initial_state, method, range_size, show=True)
            dt += math.fabs(beta_0 - self.beta)
            # store precedent value of gamma and fit on gamma
            gamma_0 = self.gamma
            self.fit_on_gamma(time, data, initial_state, method, range_size, show=True)
            dt += math.fabs(gamma_0 - self.gamma)
            # store precedent value of sigma and fit on sigma
            sigma_0 = self.sigma
            self.fit_on_sigma(time, data, initial_state, method, range_size, show=True)
            dt += math.fabs(sigma_0 - self.sigma)
            it += 1
            print(dt)

    def fit_bruteforce(self, dataset, args="hospit", method='fit_on_R', range_size=200, print_space=False):
        """
        This method will search the optimal value while testing a certain number of combinations
        Search only for SIR model, so with sigma = 1
        Args:
            - hospit = fit on cumulative hospitalisations
            - show = plot
        Method:
            - Fit on R: try to fit the R compartiment
        """
        if "hospit" in args:
            time = dataset[:, 0]
            data = dataset[:, 3]

        beta_range = np.linspace(0, 1, range_size)
        gamma_range = np.linspace(0, 1, range_size)
        self.sigma = 1
        # Set initial state:
        initial_state = (1000000 - data[0], 0, data[0], 0)
        # Store each SSE
        SSE = np.zeros((range_size, range_size))
        # Best value:
        best = (math.inf, 0, 0)
        # Compute SSE
        for b in range(0, range_size):
            for g in range(0, range_size):
                params = (beta_range[b], gamma_range[g], self.sigma)
                # compute SSE:
                sse = self.SSE(time=time, data=data, initial_state=initial_state, parameters=params, method=method)
                SSE[b][g] = sse
                if best[0] > sse:
                    best = (sse, b, g)
            print("Fitting: {} / {}".format(b, range_size))
        print("best parameters: beta={}, gamma={}, sigma={}".format(self.beta, self.gamma, self.sigma))
        if print_space:
            self.plot_sse_space(SSE, beta_range, gamma_range)

    def fit_scipy(self, dataset, args="hospit", method='fit_on_R'):

        if "hospit" in args:
            time = dataset[:, 0]
            data = dataset[:, 3]
        # Set initial state:
        initial_state = (1000000 - data[0], 0, data[0], 0)
        start_values = [0.4, 0.3, 0.3]
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, data, 'fit_on_R'), method='L-BFGS-B', bounds=[(0, 1), (0, 1), (0, 1)])
        print(res)
        self.beta = res.x[0]
        self.gamma = res.x[1]
        self.sigma = res.x[2]

    def plot_sse_space(self, SSE, beta_range, gamma_range):

        X, Y = np.meshgrid(beta_range, gamma_range)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, SSE)
        ax.set_zscale('log')
        ax.view_init(15, 60)
        plt.show()


    def fit_on_beta(self, time, data, initial_state, method='fit_on_R', range_size=2000, show=False):
        """
        Compute the best val of beta with the two others actual parameters
        """
        beta_range = np.linspace(0, 1, range_size)
        # Store SSE
        SSE = []
        best = (math.inf, 0)
        # Compute SSE values
        for beta in beta_range:
            params = (beta, self.gamma, self.sigma)
            sse = self.SSE(time=time, data=data, initial_state=initial_state, parameters=params, method=method)
            SSE.append(sse)
            if sse <= best[0]:
                best = (sse, beta)
        # set beta to the best value:
        self.beta = best[1]
        if show:
            print("best value of beta = {}".format(self.beta))
            plt.plot(beta_range, SSE)
            plt.title("Evolution of SSE with the value of beta for gamma={} "
                      "and sigma={}".format(self.gamma, self.sigma))
            plt.show()

    def fit_on_gamma(self, time, data, initial_state, method='fit_on_R', range_size=2000, show=False):
        """
        Compute the best val of gamma with the two others actual parameters
        """
        gamma_range = np.linspace(0, 1, range_size)
        # Store SSE
        SSE = []
        best = (math.inf, 0)
        # Compute SSE values
        for gamma in gamma_range:
            params = (self.beta, gamma, self.sigma)
            sse = self.SSE(time=time, data=data, initial_state=initial_state, parameters=params, method=method)
            SSE.append(sse)
            if sse <= best[0]:
                best = (sse, gamma)
        # set beta to the best value:
        self.gamma = best[1]
        if show:
            print("best value of gamma = {}".format(self.gamma))
            plt.plot(gamma_range, SSE)
            plt.title("Evolution of SSE with the value of gamma for beta={} "
                      "and sigma={}".format(self.beta, self.sigma))
            plt.show()

    def fit_on_sigma(self, time, data, initial_state, method='fit_on_R', range_size=2000, show=False):
        """
        Compute the best val of sigma with the two others actual parameters
        """
        sigma_range = np.linspace(0, 1, range_size)
        # Store SSE
        SSE = []
        best = (math.inf, 0)
        # Compute SSE values
        for sigma in sigma_range:
            params = (self.beta, self.gamma, sigma)
            sse = self.SSE(time=time, data=data, initial_state=initial_state, parameters=params, method=method)
            SSE.append(sse)
            if sse <= best[0]:
                best = (sse, sigma)
        # set beta to the best value:
        self.sigma = best[1]
        if show:
            print("best value of sigma = {}".format(self.beta))
            plt.plot(sigma_range, SSE)
            plt.title("Evolution of SSE with the value of sigma for beta={} "
                      "and gamma={}".format(self.beta, self.gamma))
            plt.show()

    def SSE(self, parameters, initial_state, time, data, method='fit_on_R'):
        """
        Compute the sum of squared errors
        """
        params = (parameters[0], parameters[1], parameters[2])
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=params)
        if method == 'fit_on_R':
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i] - predict[i][3])**2
            return sse




    def import_dataset(self, args='covid_20 - df'):
        """
        Import testing datas
        Args: must contain:
            - covid_20 for covid 20 dataset
            - np to return on np matrix format
            - df to return a pandas dataframe format
        """
        if 'covid_20' in args:
            url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
            # load pandas dataframe and convert to nparray:
            df = pd.read_csv(url, sep=",", header=0)
            if "df" in args:
                return df
            if "np" in args:
            # Convert to numpy array
                np_df = df.to_numpy()
                return np_df



    def plot_curves(self, seir_matrix, args="show", title="Plot", f_name="plot"):
        """
        Plot epidemic curves from a matrix:
            - 1th row: time index
            - others rows: S, E, I and R
        Args: must contain
            - show : to show the graph
            - save : to save the graph in a file
        """
        plt.plot(seir_matrix[:, 0], seir_matrix[:, 1], c='green', label='S')
        plt.plot(seir_matrix[:, 0], seir_matrix[:, 2], c='orange', label='E')
        plt.plot(seir_matrix[:, 0], seir_matrix[:, 3], c='red', label='I')
        plt.plot(seir_matrix[:, 0], seir_matrix[:, 4], c='blue', label='R')
        plt.xlabel('Time (days)')
        plt.ylabel('Number of peoples')
        plt.title(title)
        if "save" in args:
            plt.savefig(fname="fig/{}.pdf".format(f_name))
        if "show" in args:
            plt.show()

    def compare_with_dataset(self, data, method="cumul_hospit"):
        """
        Compare model's result with given dataset
        """

        if method == "cumul_hospit":
            # Make predictions:
            predictions = self.predict(S_0=1000000 - data[0][3], E_0=0, I_0=data[0][3], R_0=0, duration=data.shape[0])
            plt.scatter(predictions[:, 0], data[:, 3], c="blue")
            plt.plot(predictions[:, 0], predictions[:, 4], c="red")
            plt.show()


def covid_20():

    # Create the model
    model = SEIR()
    # Load the testing dataset
    data = model.import_dataset(args="covid_20, np")
    # Fit the model:
    model.fit_scipy(dataset=data, args='hospit', method='fit_on_R')
    # Make predictions:
    predictions = model.predict(S_0=999999, E_0=0, I_0=1, R_0=0, duration=300)
    # Plot predictions:
    model.plot_curves(predictions, args="show save", title="Predi with beta={}, gamma={}"
                                                           "sigma={}".format(model.beta, model.gamma, model.sigma),
                      f_name="plot_after_fitting")
    model.compare_with_dataset(data, "cumul_hospit")








if __name__ == "__main__":

    covid_20()