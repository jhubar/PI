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
    time = np.arange(90)

    # Fit the model:

    model.fit(method="stocha")

    # Make stochastic predictions:
    pred_sto = model.stochastic_predic_sans_ev(duration=model.dataset.shape[0])
    # Get the mean
    pred_sto_mean = np.mean(pred_sto, axis=2)
    # Make deterministic predictions:
    pred_det = model.predict(duration=model.dataset.shape[0])

    # Plot results for cumulative conta:
    time = range(0, model.dataset.shape[0])
    for i in range(0, 200):
        plt.plot(time, pred_sto[:, 7, i]*model.t*model.s, color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 7]*model.t*model.s, color='green', label='Stochastic conta')
    plt.plot(time, pred_det[:, 7]*model.t*model.s, color='red', label='Deterministic conta')
    plt.scatter(time, model.dataset[:, 7], color='blue', label='Testing data')
    plt.legend()
    plt.title('Cumulative testing data vs pred')
    plt.show()
    plt.close()

    # For hospit
    for i in range(0, 200):
        plt.plot(time, pred_sto[:, 4, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 4], color='green', label='Hospit pred stocha')
    plt.plot(time, pred_det[:, 4], color='red', label='Hospit pred deter')
    plt.scatter(time, model.dataset[:, 3], color='blue', label='Hospit data')
    plt.legend()
    plt.title('Hospit: pred stocha/deter vs data')
    plt.show()
    plt.close()

    # Print parameters:
    prm = model.param_translater(method='dict')
    print(prm)



