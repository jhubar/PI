import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from sklearn.linear_model import LinearRegression

from scipy.signal import savgol_filter


class positifs_reg():

    def __init__(self, degree):

        self.model = LinearRegression(degree)
        self.degree = degree
        self.score = None

    def fit(self, original_data, y):

        data = np.zeros((len(original_data), self.degree + 1))

        for i in range(0, len(original_data)):
            for d in range(0, self.degree + 1):
                data[i][d] = original_data[i] ** d

        self.model.fit(data, y)

        self.score = self.model.score(data, y)

    def predict(self, original_x):

        x = np.zeros((len(original_x), self.degree + 1))

        for i in range(0, len(original_x)):
            for d in range(0, self.degree + 1):
                x[i][d] = original_x[i] ** d

        return self.model.predict(x)


def dataframe_smoothing(df):
    # From andreas NRMAS
    # Convert the dataframe to a numpy array:
    np_df = df.to_numpy()
    # Smoothing period = 7 days
    smt_prd = 7
    smt_vec = np.ones(smt_prd)
    smt_vec /= smt_prd
    # Sore smoothed data in a new matrix:
    smoothed = np.copy(np_df)
    # How many smothing period can we place in the dataset:
    nb_per = math.floor(np_df.shape[0] / smt_prd)
    # Perform smoothing for each attributes
    for i in range(1, np_df.shape[1]):
        smoothed[:, i] = own_NRMAS(np_df[:, i], 7)

    # Write new values in a dataframe
    new_df = pd.DataFrame(smoothed, columns=df.columns)

    return new_df

def own_NRMAS_index(vector, window, index):
    smoothed_value = 0
    nb_considered_values = 0
    max_size = (window - 1) / 2
    smoothing_window = np.arange(-max_size, max_size + 1, 1)

    for j in range(window):

        sliding_index = int(index + smoothing_window[j])

        if (sliding_index >= 0) and (sliding_index <= len(vector) - 1):
            smoothed_value += vector[sliding_index]
            nb_considered_values += 1

    return smoothed_value / nb_considered_values


def own_NRMAS(vector, window):
    smoothed_vector = np.zeros(len(vector))

    if (window % 2) == 0:
        print("Error window size even")
        return

    for i in range(len(vector)):
        smoothed_vector[i] = own_NRMAS_index(vector, window, i)

    return smoothed_vector

if __name__ == "__main__":

    # a color map:
    cmap = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
    # Import the dataframe:
    raw_dataset = pd.read_csv(url, sep=',', header=0)

    np_data = raw_dataset.to_numpy()

    """
    -------------------------------------------------------
    Partie regression polynomiale
    -------------------------------------------------------
    """
    plt.scatter(np_data[:, 0], np_data[:, 1])
    idx = 0
    for m in range(3, 7):
        md = positifs_reg(m)
        md.fit(np_data[:, 0], np_data[:, 1])

        pred = md.predict(np_data[:, 0])

        plt.plot(np_data[:, 0], pred, c=cmap[idx], label='m = {}, score = {}'.format(m, md.score))
        idx += 1

    plt.legend()
    plt.show()

    """
    Mieux = polynôme de degré 2.
    """
    md = positifs_reg(2)
    md.fit(np_data[:, 0], np_data[:, 1])
    reg_coef = (md.model.coef_[1], md.model.coef_[2])

    print("coeff = ")
    print(md.model.coef_)

    """
    -------------------------------------------------------
    Partie smoothing
    -------------------------------------------------------
    """

    smoothed_data_frame = dataframe_smoothing(raw_dataset)

    smoothed_data = smoothed_data_frame.to_numpy()

    # make a matrix for positive counter
    positive_set = np.zeros((smoothed_data.shape[0], 11))
    # Indexations:
    # 0: time
    # 1: positif totday
    # 2: real positif in tested borne sup
    # 3: real positif in tested borne inf
    # 4: real positif in tested mean
    # 5: real positif in sympto borne sup
    # 6: real positif in sympto borne inf
    # 7: real positif in sympto mean
    # 8: real borne + normalized
    # 9: real borne - normalized
    # 10: real mean normalized

    # Fill the positive set
    for i in range(0, positive_set.shape[0]):
        # Time
        positive_set[i][0] = smoothed_data[i][0]
        # Evidences
        positive_set[i][1] = smoothed_data[i][1]
        # sensitivity effect
        positive_set[i][2] = positive_set[i][1] / 0.7       # div par borne inf de sensibilité
        positive_set[i][3] = positive_set[i][1] / 0.85      # div par borne sup de la sensibilité
        positive_set[i][4] = (positive_set[i][2] + positive_set[i][3]) / 2  # Make the mean
        # Proportion of symptomatic tested effect:
        positive_set[i][5] = positive_set[i][2] / 0.5       # div par borne inf de testing rate
        positive_set[i][6] = positive_set[i][3] / 1         # div par borne sup de testing rate
        positive_set[i][7] = (positive_set[i][5] + positive_set[i][6]) / 2  # Make the mean

    # print intervals:
    plt.scatter(np_data[:, 0], np_data[:, 1], c="red", label='original data')
    plt.plot(np_data[:, 0], positive_set[:, 2], c="orange", label="sensib borne sup")
    plt.plot(np_data[:, 0], positive_set[:, 3], c='orange', label='sensib borne inf')
    plt.plot(np_data[:, 0], positive_set[:, 5], c='blue', label='total borne sup')
    plt.plot(np_data[:, 0], positive_set[:, 6], c='blue', label='total borne inf')
    plt.plot(np_data[:, 0], positive_set[:, 7], c='green', label='total mean')
    plt.legend()
    plt.show()

    """
    -------------------------------------------------------
    Normalisation des données: 
    Vu que le nombre de positif mesuré est déterminé par le mobre réel de symptomatiques qui ont le corona
    multiplié par un "facteur de détection" qui est en fait la la sensibilité multiplié par le pourcentage de 
    symptomatiques testés. Donc, ce facteur de détections suit une gaussienne. Pour obtenir 
    sa déveation standard et la moyenne, on doit dabord nomaliser en divisant chaque courbe par les le nombre de test 
    positif observé
    -------------------------------------------------------
    """
    for i in range(0, positive_set.shape[0]):
        # Borne sup
        positive_set[i][8] = positive_set[i][1] / positive_set[i][5]
        # Borne inf
        positive_set[i][9] = positive_set[i][1] / positive_set[i][6]
        # Mean:
        positive_set[i][10] = (positive_set[i][8] + positive_set[i][9]) / 2

    plt.plot(positive_set[:, 0], positive_set[:, 8], c='blue', label='borne sup norm')
    plt.plot(positive_set[:, 0], positive_set[:, 9], c='blue', label='borne inf norm')
    plt.plot(positive_set[:, 0], positive_set[:, 10], c='red', label='mean norm')
    plt.show()

    # Trouver la variance à partir de l'interval de confience pour t0:
    mn = np.mean(positive_set[:, 10])
    borne = np.mean(positive_set[:, 8])
    triple_sig = borne - mn
    sigma = abs(triple_sig / 3)




    """
    -------------------------------------------------------
    MONTECARLO
    -------------------------------------------------------
    """
    nb_real = 10000
    realisations = np.zeros((nb_real, positive_set.shape[0]))



    for i in range(0, nb_real):

        # Starting value
        realisations[i][0] = 1
        for t in range(1, positive_set.shape[0]):

            sensib = np.random.normal(0.775, (0.075 / 3), 1)
            test_ratio = np.random.normal(0.75, 0.25/3, 1)
            tmp = sensib * test_ratio
            realisations[i][t] = positive_set[t][1] / (sensib * test_ratio)


    # compute mean and std over realisations
    mean = []
    var = []
    meanvarup = []
    meanvardown = []

    for i in range(0, positive_set.shape[0]):
        mean.append(np.mean(realisations[:, i]))
        var.append(np.var(realisations[:, i]))
        meanvarup.append(mean[i] + var[i])
        meanvardown.append(mean[i] - var[i])


    plt.plot(positive_set[:, 0], mean, c="blue", label='mean')
    plt.plot(positive_set[:, 0], var, c='red', label='mean + var')
    #plt.plot(positive_set[:, 0], meanvardown, c='red', label='mean - var')
    plt.legend()
    plt.show()


