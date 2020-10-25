
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math


class SIR():

    def __init__(self):
        # Parameter's values
        self.beta = 0.57
        self.gamma = 0.2

    def differential(self, state, time, beta, gamma):
        """
        Differential equations of the model
        """
        S, I, R = state

        dS = -(beta * S * I) / (S + I + R)
        dI = (beta * S * I) / (S + I + R) - (gamma * I)
        dR = gamma * I

        return dS, dI, dR

    def predict(self, S_0, I_0, R_0, duration):
        """
        Predict epidemic curves
        """
        # Initialisation vector:
        initial_state = (S_0, I_0, R_0)
        # Time vector:
        time = np.arange(duration)
        # Parameters vector
        parameters = (self.beta, self.gamma)
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=parameters)

        return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2])).T

    def fit_sequential(self, dataset, args="hospit", method='fit_on_R', range_size=2000):
        """
        This method try to fit the beta and gamma parameters of the model by doing the following steps:
            a. Start from an initial value of beta and gamma.
            b. Continue this loop while the beta and gamma variation is more than a precision boundary:
                - Enumerate a range_size length sequence of beta values to test for the actual value of gamma
                - Set as parameter the value of beta who return the smallest squared error.
                - Do the same thing for the parameter gamma
                - back to the start of the loop.
        After a small number of execution, the model converge to the optimal value.
        ----------------
        1. Args: take the type of input data to fit. Can be:
            - hospit: to fit on the cumulative number of hospitalisations
            - positive: to fit on the cumulative number of positive tests. Note: make the cumul himself.
        2. method: the parameter of the SIR model to fit:
            - fit_on_R: try to fit on the R curve (default)
            - fit_on_RI: try to fit on the sum of the cures R and I
        """
        if "hospit" in args:
            time = dataset[:, 0]
            data = dataset[:, 3]

        if "positive" in args:
            time = dataset[:, 0]
            data = dataset[:, 1]
            # Make sure we don't have a zero at the first time step
            if data[0] == 0:
                data[0] == 1
            for i in range(1, len(data)):
                data[i] += data[i - 1]

        # Initialize model's parameters
        self.beta = 0.5
        self.gamma = 0.5
        # Set initial state:
        initial_state = (1000000 - data[0], data[0], 0)
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
            it += 1
            print(dt)

    def fit_bruteforce(self, dataset, args="hospit", method='fit_on_R', range_size=200, print_space=False):
        """
        Try to find the best value of beta and gamma by enumerating the most combinations of beta and gamma.
        This method return the combination who have the best sum of square error.
        1. Args: take the type of input data to fit. Can be:
            - hospit: to fit on the cumulative number of hospitalisations
            - positive: to fit on the cumulative number of positive tests. Note: make the cumul himself.
        2. method: the parameter of the SIR model to fit:
            - fit_on_R: try to fit on the R curve (default)
            - fit_on_RI: try to fit on the sum of the cures R and I
        """
        if "hospit" in args:
            time = dataset[:, 0]
            data = dataset[:, 3]

        if "positive" in args:
            time = dataset[:, 0]
            data = dataset[:, 1]
            # Make sure we don't have a zero at the first time step
            if data[0] == 0:
                data[0] == 1
            for i in range(1, len(data)):
                data[i] += data[i-1]

        beta_range = np.linspace(0, 1, range_size)
        gamma_range = np.linspace(0, 1, range_size)
        self.sigma = 1
        # Set initial state:
        initial_state = (1000000 - data[0], data[0], 0)
        # Store each SSE
        SSE = np.zeros((range_size, range_size))
        # Best value:
        best = (math.inf, 0, 0)
        # Compute SSE
        for b in range(0, range_size):
            for g in range(0, range_size):
                params = (beta_range[b], gamma_range[g])
                # compute SSE:
                sse = self.SSE(time=time, data=data, initial_state=initial_state, parameters=params, method=method)
                SSE[b][g] = sse
                if best[0] > sse:
                    best = (sse, b, g)
            print("Fitting: {} / {}".format(b, range_size))
        print("best parameters: beta={}, gamma={}".format(self.beta, self.gamma))
        if print_space:
            self.plot_sse_space(SSE, beta_range, gamma_range)

    def fit_scipy(self, dataset, args="hospit", method='fit_on_R'):
        """
        Use scipy.optimize.minimize to search the optimal combination of
        beta and gamma
        1. Args: take the type of input data to fit. Can be:
            - hospit: to fit on the cumulative number of hospitalisations
            - positive: to fit on the cumulative number of positive tests. Note: make the cumul himself.
        2. method: the parameter of the SIR model to fit:
            - fit_on_R: try to fit on the R curve (default)
            - fit_on_RI: try to fit on the sum of the cures R and I
        """
        if "hospit" in args:
            time = dataset[:, 0]
            data = dataset[:, 3]

        if "positive" in args:
            time = dataset[:, 0]
            data = dataset[:, 1]
            # Make sure we don't have a zero at the first time step
            if data[0] == 0:
                data[0] == 1
            for i in range(1, len(data)):
                data[i] += data[i - 1]

        # Set initial state:
        initial_state = (1000000 - data[0], data[0], 0)
        start_values = [0.4, 0.3]
        res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, data, method), method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
        print(res)
        self.beta = res.x[0]
        self.gamma = res.x[1]

    def plot_sse_space(self, SSE, beta_range, gamma_range):
        """
        Draw a 3D map of SSE in fuction of the two parameters beta and gamma
        """
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
            params = (beta, self.gamma)
            sse = self.SSE(time=time, data=data, initial_state=initial_state, parameters=params, method=method)
            SSE.append(sse)
            if sse <= best[0]:
                best = (sse, beta)
        # set beta to the best value:
        self.beta = best[1]
        if show:
            print("best value of beta = {}".format(self.beta))
            plt.plot(beta_range, SSE)
            plt.title("Evolution of SSE with the value of beta for gamma={}".format(self.gamma))
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
            params = (self.beta, gamma)
            sse = self.SSE(time=time, data=data, initial_state=initial_state, parameters=params, method=method)
            SSE.append(sse)
            if sse <= best[0]:
                best = (sse, gamma)
        # set beta to the best value:
        self.gamma = best[1]
        if show:
            print("best value of gamma = {}".format(self.gamma))
            plt.plot(gamma_range, SSE)
            plt.title("Evolution of SSE with the value of gamma for beta={}".format(self.beta))
            plt.show()

    def SSE(self, parameters, initial_state, time, data, method='fit_on_R'):
        """
        Compute the sum of squared errors
        """
        params = (parameters[0], parameters[1])
        # Solve differential equations:
        predict = odeint(func=self.differential,
                         y0=initial_state,
                         t=time,
                         args=params)
        """
        1 st method: Fit on R
        Try to fit on the R curve by computing the difference
        between data's and R curve
        """
        if method == 'fit_on_R':
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i] - predict[i][2])**2
            return sse

        """
        2st method: Fit on RI: 
        Try to fit with the R + I curve
        """
        if method == 'fit_on_RI':
            sse = 0.0
            for i in range(0, len(time)):
                sse += (data[i] - predict[i][1] - predict[i][2])**2
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
        plt.plot(seir_matrix[:, 0], seir_matrix[:, 2], c='red', label='I')
        plt.plot(seir_matrix[:, 0], seir_matrix[:, 3], c='blue', label='R')
        plt.xlabel('Time (days)')
        plt.ylabel('Number of peoples')
        plt.title(title)
        if "save" in args:
            plt.savefig(fname="fig/{}.pdf".format(f_name))
        if "show" in args:
            plt.show()

    def compare_with_dataset(self, data, curve_choise="fit_on_R", data_choise="hospit"):
        """
        Compare model's result with given dataset
        Parameters:
            1. curve_choise: The curve of the model that we want to compare
                - if fit_on_R: draw the R line (default)
                - if fit_on_RI: draw the R + I line
            2. data_choise: the data that we have choise:
                - if hospit: draw the points of the cumul hospitalisation data
        """

        if data_choise == "hospit":
            # Make predictions:
            predictions = self.predict(S_0=1000000 - data[0][3], I_0=data[0][3], R_0=0, duration=data.shape[0])
            if curve_choise == "fit_on_R":
                plt.scatter(predictions[:, 0], data[:, 3], c="blue")
                plt.plot(predictions[:, 0], predictions[:, 3], c="red")
            if curve_choise == "fit_on_RI":
                Y = predictions[:, 2] + predictions[:, 3]
                plt.scatter(predictions[:, 0], data[:, 3], c="blue")
                plt.plot(predictions[:, 0], Y, c="red")
            plt.show()

        if data_choise == "positive":
            # Make cumul:
            cumul = [data[0][1]]
            if cumul[0] == 0:
                cumul[0] = 1
            for i in range(1, data.shape[0]):
                cumul.append(data[i][1] + cumul[i-1])
            # Make predictions:
            predictions = self.predict(S_0=1000000 - cumul[0], I_0=cumul[0], R_0=0, duration=data.shape[0])
            if curve_choise == "fit_on_R":
                plt.scatter(predictions[:, 0], cumul, c="blue")
                plt.plot(predictions[:, 0], predictions[:, 3], c="red")
            if curve_choise == "fit_on_RI":
                Y = predictions[:, 2] + predictions[:, 3]
                plt.scatter(predictions[:, 0], cumul, c="blue")
                plt.plot(predictions[:, 0], Y, c="red")
            plt.show()



def covid_20(fit_method="scipy"):
    """
    Utilisation du prametre fit_method: peut contenir:
        1. Choix de la façon de fitter:
            - scipy pour fitter avc scipy
            - sequential pour fitter avec la méthode séquentielle
            - bruteforce pour énumérer les possibilités et pouvoir imprimer l'espace
        2. Choix de la valeur à fitter:
            - fit_on_R : essaie de fitter la courbe R sur les données (valeur par def.)
            - fit_on_RI: essaie de fitter la courbe R + I
        3. Choix des données covid_20 à utiliser pour le fitting:
            - hospit : (valeur par default) : fit le cumul des hospitalisations
            - positive : fite the cumul of positive tests

    """

    # Create the model
    model = SIR()
    # Load the testing dataset
    data = model.import_dataset(args="covid_20, np")
    # Fit the model:
    method = "fit_on_R"
    if "fit_on_RI" in fit_method:
        method = "fit_on_RI"
    data_to_fit = "hospit"  # Default value
    if "positive" in fit_method:
        data_to_fit = "positive"

    if "scipy" in fit_method:
        model.fit_scipy(dataset=data, args=data_to_fit, method=method)
    if "sequential" in fit_method:
        model.fit_sequential(data, args=data_to_fit, method=method, range_size=2000)
    if "bruteforce" in fit_method:
        model.fit_bruteforce(data, args=data_to_fit, method=method, range_size=200)
    # Make predictions:
    predictions = model.predict(S_0=999999, I_0=1, R_0=0, duration=300)
    # Plot predictions:
    model.plot_curves(predictions, args="show save",
                      title="Predi with beta={}, gamma={}".format(model.beta, model.gamma),
                      f_name="plot_after_fitting")
    model.compare_with_dataset(data, curve_choise=method, data_choise=data_to_fit)


if __name__ == "__main__":

    # covid_20(fit_method="sequential")
    # covid_20(fit_method="scipy")
    # covid_20(fit_method="bruteforce")
    covid_20(fit_method="scipy hospit fit_on_RI")