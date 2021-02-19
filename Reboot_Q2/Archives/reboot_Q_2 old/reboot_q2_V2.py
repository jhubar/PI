import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools

if __name__ == "__main__":

    # --------------------------- Create the model --------------------------- #

    # Create the model:
    model = SEIR()

    # Load the dataset
    model.import_dataset()

    # Load best models file:
    result = pd.read_csv('FINAL_MODEL.csv', header=0, sep=';', index_col=0)
    # Numpy version
    npr = result.to_numpy()
    # Chose index 0:
    i = 0
    # Print the selected row
    # row = result.loc[i, :]
    # print(row)
    # Load parameters
    model.beta = npr[i][0]
    model.sigma = npr[i][1]
    model.gamma = npr[i][2]
    model.hp = npr[i][3]
    model.hcr = npr[i][4]
    model.pc = npr[i][5]
    model.pd = npr[i][6]
    model.pcr = npr[i][7]
    model.s = npr[i][8]
    model.t = npr[i][9]
    model.I_0 = npr[i][23]

    # Load parameters
    model.beta = npr[i][0]
    model.sigma = npr[i][1]
    model.gamma = npr[i][2]
    model.hp = npr[i][3]
    model.hcr = npr[i][4]
    model.pc = npr[i][5]
    model.pd = npr[i][6]
    model.pcr = npr[i][7]
    model.s = npr[i][8]
    model.t = npr[i][9]
    model.I_0 = npr[i][23]
    # Get time vector:
    time = np.arange(90)

    #model.fit()


    # Make 70 day predictions
    n = model.dataset.shape[0]
    pred_A = model.stochastic_predic(n, nb_simul=200)
    # Get the mean
    pred_A_mean = np.mean(pred_A, axis=2)
    # Get deterministic predictions
    pred_det = model.predict(90)

    # Plot testing data

    for i in range(0, pred_A.shape[2]):
        plt.plot(range(0, n), pred_A[:, 7, i]*model.t*model.s, color='limegreen', linewidth=0.1)

    plt.plot(range(0, n), pred_A_mean[:, 7]*model.t*model.s, color='forestgreen', label='Test predictions')
    plt.scatter(range(0, n), model.dataset[:, 7], label='Testing data')
    plt.legend()
    plt.show()
    plt.close()

    # plot hopit data

    for i in range(0, pred_A.shape[2]):
        plt.plot(range(0, n), pred_A[:, 4, i], color='red', linewidth=0.1)
    plt.plot(range(0, n), pred_A_mean[:, 4], color='darkred', label='Hospit predictions')
    plt.scatter(range(0, n), model.dataset[:, 3], label='Hospit data')
    plt.legend()
    plt.show()
    plt.close()

    # plot critical data

    for i in range(0, pred_A.shape[0]):
        plt.plot(range(0, n), pred_A[:, 5, i], color='dodgerblue', linewidth=0.1 )
    plt.plot(range(0, n), pred_A_mean[:, 5], color='blue', label='Critical predictions')
    plt.scatter(range(0, n), model.dataset[:, 5], label='Critical data')
    plt.legend()
    plt.show()
    plt.close()







