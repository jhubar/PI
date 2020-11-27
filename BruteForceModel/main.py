
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools




if __name__ == "__main__":

    # Create a model
    model = SEIR()
    # Import the dataset
    model.import_dataset()
    # Find an optimal value for initial state
    #init_I = tools.initial_infected_estimator(model.dataset)

    model.fit()
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