import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools
import bruteforce


def test_1(df, store_def=True):

    # Create the model:
    mdl = SEIR()
    mdl.import_dataset()

    # Read parameter row:
    row_A = df.loc[0, :]

    # Configure parameters:
    mdl.set_parameters_from_bf(row_A)

    # Make the little bruteforce
    #bruteforce.little_bruteforce(mdl)

    # Better fit on gamma and hp:
    mdl.fit_gamma_hp_rate(plot=True)

    # Fit the rest
    mdl.fit_part_2()
    mdl.get_parameters()
    print('new parameters:')
    print(mdl.get_parameters())
    # Make predictions:
    predictions = mdl.predict(duration=mdl.dataset.shape[0])

    # Uncumul contaminations data
    uncumul = []
    uncumul.append(predictions[0][7])
    for j in range(1, predictions.shape[0]):
        uncumul.append(predictions[j][7] - predictions[j - 1][7])

    # Plot:
    time = mdl.dataset[:, 0]
    # Adapt test + with sensit and testing rate
    for j in range(0, len(time)):
        uncumul[j] = uncumul[j] * mdl.s * mdl.t

    # Plot cumul positive
    plt.scatter(time, mdl.dataset[:, 1], c='blue', label='test+')
    plt.plot(time, uncumul, c='blue', label='test+')
    # Plot hospit
    plt.scatter(time, mdl.dataset[:, 3], c='red', label='hospit data')
    plt.plot(time, predictions[:, 4], c='red', label='hospit pred')
    plt.legend()
    plt.title('test_1')
    plt.show()

    plt.scatter(time, mdl.dataset[:, 5], c='green', label='Critical data')
    plt.plot(time, predictions[:, 5], c='green', label='Critical predictions')
    plt.scatter(time, mdl.dataset[:, 6], c='black', label='Fatalities data')
    plt.plot(time, predictions[:, 6], c='black', label='Fatalities predictions')
    plt.legend()
    plt.title('test_1')
    plt.show()

    if store_def:
        # Store model parameters in the definitive file
        final_param = mdl.get_parameters()
        h_param = mdl.get_hyperparameters()
        # Make the string and store in file
        index = 0
        pre_string = [str(index)]
        for item in final_param:
            pre_string.append(str(item))
        for item in h_param:
            pre_string.append(str(item))

        string = ';'.join(pre_string)

        # Open and write in the file
        file = open('FINAL_MODEL.csv', 'a')
        file.write(string)
        file.write('\n')
        file.close()


def reader():

    # Import file in a dataframe:
    result = pd.read_csv('FINAL_MODEL.csv', header=0, sep=';', index_col=0)
    # Numpy version
    npr = result.to_numpy()

    # Create a model:
    model = SEIR()
    # Import the dataset
    model.import_dataset()

    while True:

        print("===========================Enter the index of the row to analyse=================================")
        i = int(input())

        # Print the selected row
        row = result.loc[i, :]
        print(row)

        # Load parameters:
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
        # Make predictions
        predictions = model.predict(duration=model.dataset.shape[0])

        # Uncumul contaminations data
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
        plt.scatter(time, model.dataset[:, 3], c='red', label='hospit data')
        plt.plot(time, predictions[:, 4], c='red', label='hospit pred')
        plt.legend()
        plt.xlabel('Time (days)')
        plt.ylabel('Number of people')
        # plt.title('Index {}'.format(i))
        plt.show()

        plt.scatter(time, model.dataset[:, 5], c='green', label='Critical data')
        plt.plot(time, predictions[:, 5], c='green', label='Critical predictions')
        plt.scatter(time, model.dataset[:, 6], c='black', label='Fatalities data')
        plt.plot(time, predictions[:, 6], c='black', label='Fatalities predictions')
        plt.legend()
        plt.xlabel('Time (days)')
        plt.ylabel('Number of people')
        # plt.title('index {}'.format(i))
        plt.show()

        model.stocha_perso()


if __name__ == "__main__":

    # Import best result file
    best_result = pd.read_csv('result_analysis/best.csv', sep=';', index_col=0)

    #test_1(best_result)

    # Reader
    reader()
