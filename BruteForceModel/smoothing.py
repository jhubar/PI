import numpy as np
import pandas as pd
import math


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