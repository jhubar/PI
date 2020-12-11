
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools



def scenario_julien():
    # Create the model:
    model_1 = SEIR()
    model_2 = SEIR()
    model_3 = SEIR()
    model_4 = SEIR()
    model_5= SEIR()
    model_6 = SEIR()
    model_7 = SEIR()

    # Load the dataset
    model_1.import_dataset()
    model_2.import_dataset()
    model_3.import_dataset()
    model_4.import_dataset()
    model_5.import_dataset()
    model_6.import_dataset()
    model_7.import_dataset()

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
    model1.beta = npr[i][0]
    model1.sigma = npr[i][1]
    model1.gamma = npr[i][2]
    model.hp = npr[i][3]
    model.hcr = npr[i][4]
    model.pc = npr[i][5]
    model.pd = npr[i][6]
    model.pcr = npr[i][7]
    model.s = npr[i][8]
    model.t = npr[i][9]
    model.I_0 = npr[i][23]

    # Load parameters
    model1.beta = npr[i][0]
    model1.sigma = npr[i][1]
    model1.gamma = npr[i][2]
    model1.hp = npr[i][3]
    model1.hcr = npr[i][4]
    model1.pc = npr[i][5]
    model1.pd = npr[i][6]
    model1.pcr = npr[i][7]
    model1.s = npr[i][8]
    model1.t = npr[i][9]
    model1.I_0 = npr[i][23]
    # Get time vector:
    time = np.arange(300)

    # Build scenario
    scenario = {
        'duration': 300,
        # 'close_schools': [70, 100],
        # 'social_dist': [85, 120, 6],
        'lock_down': [70, 100]
    }
    model.set_scenario(scenario)
    # Make stochastic predictions according to scenario
    pred_scenar = model.stochastic_predic(duration=300, parameters=None, nb_simul=200, scenar=True)
    # Make normal stochastic predict
    pred_normal = model.stochastic_predic(duration=300, parameters=None, nb_simul=200, scenar=False)

    # Compute mean predictions
    mean_scenar = np.mean(pred_scenar, axis=2)
    mean_normal = np.mean(pred_normal, axis=2)
    model.nb_simul = 200

    # Hospit
    fig, ax1 = plt.subplots()
    fig.suptitle('Hospit curves')
    ax1.plot(time, mean_scenar[:, 4], c='green', label='stocha mean')
    ax1.plot(time, mean_normal[:, 4], c='green', label='stocha mean')


    #Fatalities
    # ax1.plot(time, mean_scenar[:, 6], c='red', label='stocha mean')
    # ax1.plot(time, mean_normal[:, 6], c='red', label='stocha mean')

    # Build scenario
    scenario1 = {
        'duration': 300,
        # 'close_schools': [70, 100],
        'social_dist': [70, 100, 6],
        # 'lock_down': [73, 100]
    }
    model1.set_scenario(scenario1)
    # Make stochastic predictions according to scenario
    pred_scenar1 = model1.stochastic_predic(duration=300, parameters=None, nb_simul=200, scenar=True)
    # Make normal stochastic predict
    pred_normal1 = model1.stochastic_predic(duration=300, parameters=None, nb_simul=200, scenar=False)

    # Compute mean predictions
    mean_scenar1 = np.mean(pred_scenar1, axis=2)
    mean_normal1 = np.mean(pred_normal1, axis=2)
    model1.nb_simul = 200
    # Hospit

    fig.suptitle('Hospit curves')
    ax1.plot(time, mean_scenar1[:, 4], c='green', label='stocha mean 1')
    ax1.plot(time, mean_normal1[:, 4], c='green', label='stocha mean 1')

    #Fatalities
    # ax1.plot(time, mean_scenar1[:, 6], c='red', label='stocha mean 1')
    # ax1.plot(time, mean_normal1[:, 6], c='red', label='stocha mean 1')
    plt.axhline(300)
    plt.axhline(3000)
    ax1.legend()
    plt.savefig('imgJulien/lockdonw.png')
    plt.close()
