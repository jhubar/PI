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
    time = np.arange(300)

    # --------------------------- Load assistant realization --------------------------- #

    url = 'https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/Cov_invaders.csv'
    # Import the dataframe:
    delau_data = pd.read_csv(url, sep=',', header=0)
    delau_np = delau_data.to_numpy()

    # --------------------------- 190 day with no measures ----------------------------- #

    # Get predictions
    pred_A = model.stochastic_predic(191, nb_simul=200)
    print(pred_A.shape)
    # Get mean predictions
    pred_A_mean = np.mean(pred_A, axis=2)

    # --------------------------- Compare between Delau and us -------------------------- #

    # Plot hospit
    plt.plot(delau_np[:, 0], pred_A_mean[:, 4], color="blue", label='pred hospit, no measures')
    for i in range(0, pred_A.shape[0]):
        plt.plot(pred_A[:, 4, i], color="green", linewidth=0.15)
    plt.plot(delau_np[:, 0], delau_np[:, 3], color="red", label='delau hospit, scenar')
    plt.legend()
    plt.show()
    plt.close()

    # --------------------------- 20 days of mask + social d ----------------------------- #

    scenario_1 = {
        'duration': 191,
        'social_dist': [76, 118, 6],
        'wearing_mask': [76, 118]
    }
    # Put the scenario in the model:
    model.set_scenario(scenario_1)
    # Make predictions:
    pred_B = model.stochastic_predic(duration=191, parameters=None,
                                     nb_simul=200, scenar=True)
    # Get the mean prediction
    pred_B_mean = np.mean(pred_B, axis=2)

    # --------------------------- Compare between Delau and us -------------------------- #

    # Plot hospit
    plt.plot(delau_np[:, 0], pred_B_mean[:, 4], color="blue", label='pred hospit')
    for i in range(0, pred_B.shape[0]):
        plt.plot(pred_B[:, 4, i], color="green", linewidth=0.15)
    plt.plot(delau_np[:, 0], delau_np[:, 3], color="red", label='delau hospit, scenar')
    plt.legend()
    plt.show()
    plt.title("Scenar vs Delau")
    plt.close()

    # --------------------------- Illustration rapport A------- -------------------------- #