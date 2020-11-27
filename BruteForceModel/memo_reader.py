import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SEIR import SEIR
import os







if __name__ == "__main__":


    result = pd.read_csv('result/memo_values.csv', header=0, index_col=0, sep=';')

    # Sort:
    result.sort_values(by=['score'], inplace=True, ignore_index=True, ascending=True)
    print(result)

    npr = result.to_numpy()

    for i in range(0, npr.shape[0]):

        # Create a model:
        model = SEIR()
        model.fit_display = True
        model.basis_obj_display = False
        model.full_obj_display = False
        # Load parameters:
        model.beta = npr[i][1]
        model.sigma = npr[i][2]
        model.gamma = npr[i][3]
        model.hp = npr[i][4]
        model.hcr = npr[i][5]
        model.pc = npr[i][6]
        model.pd = npr[i][7]
        model.pcr = npr[i][8]
        model.s = npr[i][9]
        model.t = npr[i][10]

        # Import dataset:
        model.import_dataset()

        # Fit the model:
        #model.fit()

        # Make predictions:
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
        plt.title('Index {}'.format(i))
        plt.show()

        plt.scatter(time, model.dataset[:, 5], c='green', label='Critical data')
        plt.plot(time, predictions[:, 5], c='green', label='Critical predictions')
        plt.scatter(time, model.dataset[:, 6], c='black', label='Fatalities data')
        plt.plot(time, predictions[:, 6], c='black', label='Fatalities predictions')
        plt.legend()
        plt.title('index {}'.format(i))
        plt.show()

        print('---------------------------------------------------------')

        row = result.loc[i, :]

        print(row)

        print("<Press enter/return to continue>")
        input()