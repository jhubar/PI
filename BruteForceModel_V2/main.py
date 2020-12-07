
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools


def main_a():

    # Create a model
    model = SEIR()
    # Import the dataset
    model.import_dataset()
    # Find an optimal value for initial state
    #init_I = tools.initial_infected_estimator(model.dataset)

    #model.fit()
    predictions = model.predict(duration=model.dataset.shape[0])

    print(model.get_parameters())

    uncumul = []
    uncumul.append(predictions[0][7])
    for j in range(1, predictions.shape[0]):
        uncumul.append(predictions[j][7] - predictions[j - 1][7])

    # Plot:
    time = model.dataset[:, 0]
    # Adapt test + with sensit and testing rate
    for j in range(0, len(time)):
        uncumul[j] = uncumul[j] * model.s * model.t

    # Plot cumul positive
    plt.scatter(time, model.dataset[:, 1], c='blue', label='test+')
    plt.plot(time, uncumul, c='blue', label='test+')
    # Plot hospit
    plt.scatter(time, model.dataset[:, 3], c='red', label='hospit pred')
    plt.plot(time, predictions[:, 4], c='red', label='pred hopit')
    plt.legend()
    plt.show()

    # Plot critical
    plt.scatter(time, model.dataset[:, 5], c='green', label='critical data')
    plt.plot(time, predictions[:, 5], c='green', label='critical pred')
    plt.scatter(time, model.dataset[:, 6], c='black', label='fatalities data')
    plt.plot(time, predictions[:, 6], c='black', label='fatalities pred')
    plt.legend()
    plt.show()

    # Smoothing test:
    unsmooth_data = model.dataset

    # Import a smoothed dataset:
    model.smoothing = True
    model.import_dataset()
    smooth_data = model.dataset

    # plot the data
    plt.scatter(smooth_data[:, 0], smooth_data[:, 1], color='blue', label='smoothed testing data')
    plt.scatter(smooth_data[:, 0], unsmooth_data[:, 1], color='green', label='unsmoothed testing data')
    plt.legend()
    plt.show()
    # Print initial data:
    for i in range(0, 15):
        print('Time: {} - smoothed: {} - original: {}'.format(i, smooth_data[i][1], unsmooth_data[i][1]))

    # Check best initial number of infected:
    tools.initial_infected_estimator(smooth_data)


def main_b():

    # Create a model
    model = SEIR()
    # Import the dataset
    model.import_dataset()

    # Find an optimal value for initial state
    # init_I = tools.initial_infected_estimator(model.dataset)

    # Set good parameters:
    model.beta = 0.401739
    model.sigma = 0.849249
    model.gamma = 0.27155
    model.hp = 0.0143677
    model.hcr = 0.0505969
    model.pc = 0.0281921
    model.pcr = 0.105229
    model.pd = 0.0489863
    model.s = 0.799943
    model.t = 0.946585
    model.I_0 = 28.6756

    # Time vector
    time = model.dataset[:, 0]

    # Make stochastic predictions
    prd = model.stochastic_predic(len(time))

    mean_pred = np.mean(prd, axis=2)
    print(mean_pred.shape)

    uncumul = []
    uncumul.append(mean_pred[0][7])
    for j in range(1, mean_pred.shape[0]):
        uncumul.append(mean_pred[j][7] - mean_pred[j - 1][7])
        print('iter {} - conta: {}'.format(j, uncumul[j]))

    # Plot:
    time = model.dataset[:, 0]
    # Adapt test + with sensit and testing rate
    for j in range(0, len(time)):
        uncumul[j] = uncumul[j] * model.s * model.t

    # Plot cumul positive
    plt.scatter(time, model.dataset[:, 1], c='blue', label='test+')
    plt.plot(time, uncumul, c='blue', label='test+')
    # Plot hospit
    plt.scatter(time, model.dataset[:, 3], c='red', label='hospit pred')
    plt.plot(time, mean_pred[:, 4], c='red', label='pred hopit')
    plt.legend()
    plt.show()

    # Plot critical
    plt.scatter(time, model.dataset[:, 5], c='green', label='critical data')
    plt.plot(time, mean_pred[:, 5], c='green', label='critical pred')
    plt.scatter(time, model.dataset[:, 6], c='black', label='fatalities data')
    plt.plot(time, mean_pred[:, 6], c='black', label='fatalities pred')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # main_a()

    main_b()