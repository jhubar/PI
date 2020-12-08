
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools


if __name__ == "__main__":

    # =============================================================================================================== #
    #                                                                                                                 #
    #                                               COVID Bitches'                                                    #
    #                                                                                                                 #
    # =============================================================================================================== #

    # Create the model:
    model = SEIR()
    # Load the dataset
    model.import_dataset()

    # =============================================================================================================== #
    # Load best model from Bruteforcing model selection                                                               #
    # =============================================================================================================== #

    # Load best models file:
    result = pd.read_csv('FINAL_MODEL.csv', header=0, sep=';', index_col=0)
    # Numpy version
    npr = result.to_numpy()
    # Chose index 0:
    i = 0
    # Print the selected row
    row = result.loc[i, :]
    print(row)
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
    time = np.arange(model.dataset.shape[0] + 10)
    # Make deterministic predictions:
    deter_pred = model.predict(duration=len(time))
    # Set the number of realizations to perform for the stochastic model:
    model.nb_simul = 200
    # Make stochastic predictions without taking account of tests values
    stocha_pred_basis = model.stochastic_predic_sans_ev(len(time))
    # Make stochastic predictions while taking account of tests values
    stocha_pred_ev = model.stochastic_predic(len(time))
    # Compute mean predictions
    mean_stocha_basis = np.mean(stocha_pred_basis, axis=2)
    mean_stocha_ev = np.mean(stocha_pred_ev, axis=2)

    # Contamination cumul plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('Contaminations cumul curves')
    ax1.plot(time, deter_pred[:, 7], c='blue', label='Deter pred')
    ax1.scatter(model.dataset[:, 0], model.dataset[:, 7]/(model.s*model.t), c='black', label='Dataset')
    ax1.plot(time, mean_stocha_basis[:, 7], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, stocha_pred_basis[:, 7, i], c='green', linewidth=0.1)
    ax1.plot(time, stocha_pred_basis[:, 7, model.nb_simul - 1], c='green', linewidth=0.1, label='Stochastic conta')
    ax1.legend()

    ax2.plot(time, deter_pred[:, 7], c='blue', label='Deter pred')
    ax2.scatter(model.dataset[:, 0], model.dataset[:, 7]/(model.s*model.t), c='black', label='Dataset')
    ax2.plot(time, mean_stocha_ev[:, 7], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax2.plot(time, stocha_pred_ev[:, 7, i], c='green', linewidth=0.1)
    ax2.plot(time, stocha_pred_ev[:, 7, model.nb_simul - 1], c='green', linewidth=0.1, label='Stochastic conta')
    ax2.legend()
    fig.show()

    # Infection curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('Infections curves')
    ax1.plot(time, deter_pred[:, 2], c='blue', label='Deter pred')
    ax1.plot(time, mean_stocha_basis[:, 2], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, stocha_pred_basis[:, 2, i], c='green', linewidth=0.1)
    ax1.plot(time, stocha_pred_basis[:, 2, model.nb_simul - 1], c='green', linewidth=0.1, label='Stochastic')
    ax1.legend()

    ax2.plot(time, deter_pred[:, 2], c='blue', label='Deter pred')
    ax2.plot(time, mean_stocha_ev[:, 2], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax2.plot(time, stocha_pred_ev[:, 2, i], c='green', linewidth=0.1)
    ax2.plot(time, stocha_pred_ev[:, 2, model.nb_simul - 1], c='green', linewidth=0.1, label='Stochastic ')
    ax2.legend()
    fig.show()

    # Hospit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('Hospit curves')
    ax1.plot(time, deter_pred[:, 4], c='blue', label='Deter pred')
    ax1.plot(time, mean_stocha_basis[:, 4], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, stocha_pred_basis[:, 4, i], c='yellow', linewidth=0.1)
    ax1.plot(time, stocha_pred_basis[:, 4, model.nb_simul - 1], c='yellow', linewidth=0.1, label='Stochastic')
    ax1.legend()

    ax2.plot(time, deter_pred[:, 4], c='blue', label='Deter pred')
    ax2.plot(time, mean_stocha_ev[:, 4], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax2.plot(time, stocha_pred_ev[:, 4, i], c='yellow', linewidth=0.1)
    ax2.plot(time, stocha_pred_ev[:, 4, model.nb_simul - 1], c='yellow', linewidth=0.1, label='Stochastic')
    ax2.legend()
    fig.show()

    # Critical
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('Critical curves')
    ax1.plot(time, deter_pred[:, 5], c='blue', label='Deter pred')
    ax1.plot(time, mean_stocha_basis[:, 5], c='red', label='stocha mean')
    ax1.scatter(model.dataset[:, 0], model.dataset[:, 5] / (model.s * model.t), c='black', label='Dataset')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, stocha_pred_basis[:, 5, i], c='yellow', linewidth=0.1)
    ax1.plot(time, stocha_pred_basis[:, 5, model.nb_simul - 1], c='yellow', linewidth=0.1, label='Stochastic')
    ax1.legend()

    ax2.plot(time, deter_pred[:, 5], c='blue', label='Deter pred')
    ax2.plot(time, mean_stocha_ev[:, 5], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax2.plot(time, stocha_pred_ev[:, 5, i], c='yellow', linewidth=0.1)
    ax2.plot(time, stocha_pred_ev[:, 5, model.nb_simul - 1], c='yellow', linewidth=0.1, label='Stochastic')
    ax2.scatter(model.dataset[:, 0], model.dataset[:, 5] / (model.s * model.t), c='black', label='Dataset')
    ax2.legend()
    fig.show()

    # Fatalities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('Fatalities curves')
    ax1.plot(time, deter_pred[:, 6], c='blue', label='Deter pred')
    ax1.plot(time, mean_stocha_basis[:, 6], c='red', label='stocha mean')
    ax1.scatter(model.dataset[:, 0], model.dataset[:, 6] / (model.s * model.t), c='black', label='Dataset')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, stocha_pred_basis[:, 6, i], c='yellow', linewidth=0.1)
    ax1.plot(time, stocha_pred_basis[:, 6, model.nb_simul - 1], c='yellow', linewidth=0.1, label='Stochastic')
    ax1.legend()

    ax2.plot(time, deter_pred[:, 6], c='blue', label='Deter pred')
    ax2.plot(time, mean_stocha_ev[:, 6], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax2.plot(time, stocha_pred_ev[:, 6, i], c='yellow', linewidth=0.1)
    ax2.plot(time, stocha_pred_ev[:, 6, model.nb_simul - 1], c='yellow', linewidth=0.1, label='Stochastic')
    ax2.scatter(model.dataset[:, 0], model.dataset[:, 6] / (model.s * model.t), c='black', label='Dataset')
    ax2.legend()
    fig.show()







