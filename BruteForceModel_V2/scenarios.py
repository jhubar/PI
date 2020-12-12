
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools



def scenario_julien():
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

    # --------------------------- Build scenarios --------------------------- #

    # Choose scenario

    scenario_1 = {
        'duration': 300,
        'lock_down': [70, 100]
    }
    scenario_2 = {
        'duration': 300,
        'close_schools': [70, 100]
    }
    scenario_3 = {
        'duration': 300,
        'social_dist': [85, 120, 6]
    }
    scenario_4 = {
        'duration': 300,
        'wearing_mask': [85, 120]
    }
    scenario_5 = {
        'duration': 300,
        'wearing_mask': [85, 120],
        'close_schools': [70, 100]
    }
    scenario_6 = {
        'duration': 300,
        'lock_down': [70, 77]
    }
    scenario_7 = {
        'duration': 300,
        'lock_down': [70, 77],
        'wearing_mask': [77, 100]
    }

    # --------------------------- Create models --------------------------- #

    model.set_scenario(scenario_1)
    pred_scenar_1 = model.stochastic_predic(duration=300, parameters=None,
                                          nb_simul=200, scenar=True)
    pred_normal_1 = model.stochastic_predic(duration=300, parameters=None,
                                          nb_simul=200, scenar=False)

    model.set_scenario(scenario_2)
    pred_scenar_2 = model.stochastic_predic(duration=300, parameters=None,
                                            nb_simul=200, scenar=True)

    model.set_scenario(scenario_3)
    pred_scenar_3 = model.stochastic_predic(duration=300, parameters=None,
                                            nb_simul=200, scenar=True)

    model.set_scenario(scenario_4)
    pred_scenar_4 = model.stochastic_predic(duration=300, parameters=None,
                                            nb_simul=200, scenar=True)

    model.set_scenario(scenario_5)
    pred_scenar_5 = model.stochastic_predic(duration=300, parameters=None,
                                            nb_simul=200, scenar=True)

    model.set_scenario(scenario_6)
    pred_scenar_6 = model.stochastic_predic(duration=300, parameters=None,
                                            nb_simul=200, scenar=True)

    model.set_scenario(scenario_7)
    pred_scenar_7 = model.stochastic_predic(duration=300, parameters=None,
                                            nb_simul=200, scenar=True)

    # Compute mean predictions
    mean_scenar_1 = np.mean(pred_scenar_1, axis=2)
    mean_normal_1 = np.mean(pred_normal_1, axis=2)
    model.nb_simul = 200

    mean_scenar_2 = np.mean(pred_scenar_2, axis=2)

    mean_scenar_3 = np.mean(pred_scenar_3, axis=2)

    mean_scenar_4 = np.mean(pred_scenar_4, axis=2)

    mean_scenar_5 = np.mean(pred_scenar_5, axis=2)

    mean_scenar_6 = np.mean(pred_scenar_6, axis=2)

    mean_scenar_7 = np.mean(pred_scenar_7, axis=2)

    # --------------------------- Plot scenarios --------------------------- #

    # Hospit
    fig_H = plt.figure()
    ax_H = plt.subplot()
    ax_H.plot(time, mean_normal_1[:, 4], c='black', label='initial')
    ax_H.plot(time, mean_scenar_1[:, 4], label='scena 1')
    ax_H.plot(time, mean_scenar_2[:, 4], label='scena 2')
    ax_H.plot(time, mean_scenar_3[:, 4], label='scena 3')
    ax_H.plot(time, mean_scenar_4[:, 4], label='scena 4')
    ax_H.plot(time, mean_scenar_5[:, 4], label='scena 5')
    ax_H.plot(time, mean_scenar_6[:, 4], label='scena 6')
    ax_H.plot(time, mean_scenar_7[:, 4], label='scena 7')
    plt.axhline(1500, label="max hospit.")
    plt.legend()
    plt.title("Hospitalization")
    fig_H.savefig("img_andreas/Hospitalization.pdf")

    #Criticals
    fig_C = plt.figure()
    ax_C = plt.subplot()
    ax_C.plot(time, mean_normal_1[:, 5], c='black', label='initial')
    ax_C.plot(time, mean_scenar_1[:, 5], label='scena 1')
    ax_C.plot(time, mean_scenar_2[:, 5], label='scena 2')
    ax_C.plot(time, mean_scenar_3[:, 5], label='scena 3')
    ax_C.plot(time, mean_scenar_4[:, 5], label='scena 4')
    ax_C.plot(time, mean_scenar_5[:, 5], label='scena 5')
    ax_C.plot(time, mean_scenar_6[:, 5], label='scena 6')
    ax_C.plot(time, mean_scenar_7[:, 5], label='scena 7')
    plt.axhline(300, label='max criticals')
    plt.legend()
    plt.title("Criticals")
    fig_C.savefig('img_andreas/Criticals.pdf')
