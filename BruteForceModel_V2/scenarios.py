
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
    time = np.arange(150)

    # Build scenario
    scenario = {
        'duration': 150,
        'close_schools': [70, 100],
        # 'social_dist': [85, 120, 6]
    }
    model.set_scenario(scenario)
    # Make stochastic predictions according to scenario
    pred_scenar = model.stochastic_predic(duration=150, parameters=None, nb_simul=200, scenar=True)
    # Make normal stochastic predict
    pred_normal = model.stochastic_predic(duration=150, parameters=None, nb_simul=200, scenar=False)

    # Compute mean predictions
    mean_scenar = np.mean(pred_scenar, axis=2)
    mean_normal = np.mean(pred_normal, axis=2)
    model.nb_simul = 200

    plt.close()
    # I curve
    fig, ax1 = plt.subplots()
    fig.suptitle('I curves')
    ax1.plot(time, mean_scenar[:, 2], c='blue', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, pred_scenar[:, 2, i], c='green', linewidth=0.1)
    ax1.plot(time, pred_scenar[:, 2, model.nb_simul - 1], c='green', linewidth=0.1, label='Stochastic I')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, pred_normal[:, 2, i], c='blue', linewidth=0.1)
    ax1.plot(time, pred_normal[:, 2, model.nb_simul - 1], c='blue', linewidth=0.1, label='Stochastic I')
    ax1.legend()
    plt.savefig('imgJulien/I_curve.png')
    plt.close()

    # fig.show()

    # Hospit
    fig, ax1 = plt.subplots()
    fig.suptitle('Hospit curves')
    ax1.plot(time, mean_scenar[:, 4], c='black', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, pred_scenar[:, 4, i], c='green', linewidth=0.1)
    ax1.plot(time, pred_scenar[:, 4, model.nb_simul - 1], c='green', linewidth=0.1, label='Stochastic')


    ax1.plot(time, mean_normal[:, 4], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, pred_normal[:, 4, i], c='green', linewidth=0.1)
    ax1.plot(time, pred_normal[:, 4, model.nb_simul - 1], c='green', linewidth=0.1, label='Stochastic')



    # Critical

    ax1.plot(time, mean_scenar[:, 5], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, pred_scenar[:, 5, i], c='orange', linewidth=0.1)
    ax1.plot(time, pred_scenar[:, 5, model.nb_simul - 1], c='orange', linewidth=0.1, label='Stochastic')


    ax1.plot(time, mean_normal[:, 5], c='red', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, pred_normal[:, 5, i], c='orange', linewidth=0.1)
    ax1.plot(time, pred_normal[:, 5, model.nb_simul - 1], c='orange', linewidth=0.1, label='Stochastic')

    #Fatalities
    ax1.plot(time, mean_scenar[:, 6], c='black', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, pred_scenar[:, 6, i], c='red', linewidth=0.1)
    ax1.plot(time, pred_scenar[:, 6, model.nb_simul - 1], c='red', linewidth=0.1, label='Stochastic')


    ax1.plot(time, mean_normal[:, 6], c='black', label='stocha mean')
    for i in range(0, model.nb_simul - 1):
        ax1.plot(time, pred_normal[:, 6, i], c='red', linewidth=0.1)
    ax1.plot(time, pred_normal[:, 6, model.nb_simul - 1], c='red', linewidth=0.1, label='Stochastic')

    ax1.legend()
    plt.savefig('imgJulien/Hospit.png')
    plt.close()

def scenario_julien2():
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
    time = np.arange(150)

    # Build scenario
    scenario = {
        'duration': 150,
        # 'close_schools': [70, 100],
        'social_dist': [85, 120, 6]
    }
    model.set_scenario(scenario)
    # Make stochastic predictions according to scenario
    pred_scenar = model.stochastic_predic(duration=150, parameters=None, nb_simul=200, scenar=True)
    # Make normal stochastic predict
    pred_normal = model.stochastic_predic(duration=150, parameters=None, nb_simul=200, scenar=False)

    # Compute mean predictions
    mean_scenar = np.mean(pred_scenar, axis=2)
    mean_normal = np.mean(pred_normal, axis=2)
    model.nb_simul = 200

    # Hospit
    fig, ax1 = plt.subplots()
    fig.suptitle('Hospit curves')
    ax1.plot(time, mean_scenar[:, 4], c='green', label='stocha mean')
    ax1.plot(time, mean_normal[:, 4], c='green', label='stocha mean')
    # Critical

    ax1.plot(time, mean_scenar[:, 5], c='orange', label='stocha mean')
    ax1.plot(time, mean_normal[:, 5], c='orange', label='stocha mean')

    #Fatalities
    ax1.plot(time, mean_scenar[:, 6], c='red', label='stocha mean')
    ax1.plot(time, mean_normal[:, 6], c='red', label='stocha mean')


    ax1.legend()
    plt.savefig('imgJulien/Criticals.png')
    plt.close()


def scenario_julien3():
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
    time = np.arange(150)

    # Build scenario
    scenario = {
        'duration': 150,
        'close_schools': [70, 100],
        'social_dist': [85, 120, 6]
    }
    model.set_scenario(scenario)
    # Make stochastic predictions according to scenario
    pred_scenar = model.stochastic_predic(duration=150, parameters=None, nb_simul=200, scenar=True)
    # Make normal stochastic predict
    pred_normal = model.stochastic_predic(duration=150, parameters=None, nb_simul=200, scenar=False)

    # Compute mean predictions
    mean_scenar = np.mean(pred_scenar, axis=2)
    mean_normal = np.mean(pred_normal, axis=2)
    model.nb_simul = 200

    # Hospit
    fig, ax1 = plt.subplots()
    fig.suptitle('Hospit curves')
    ax1.plot(time, mean_scenar[:, 4], c='green', label='stocha mean')
    ax1.plot(time, mean_normal[:, 4], c='green', label='stocha mean')
    # Critical

    ax1.plot(time, mean_scenar[:, 5], c='orange', label='stocha mean')
    ax1.plot(time, mean_normal[:, 5], c='orange', label='stocha mean')

    #Fatalities
    ax1.plot(time, mean_scenar[:, 6], c='red', label='stocha mean')
    ax1.plot(time, mean_normal[:, 6], c='red', label='stocha mean')
    plt.hline()
    plt.hline()
    ax1.legend()
    plt.savefig('imgJulien/Criticals2.png')
    plt.close()



def scenario_julien3():
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
    time = np.arange(150)

    # Build scenario
    scenario = {
        'duration': 150,
        # 'close_schools': [70, 100],
        # 'social_dist': [85, 120, 6],
        'lock_down': [73, 128],
    }
    model.set_scenario(scenario)
    # Make stochastic predictions according to scenario
    pred_scenar = model.stochastic_predic(duration=150, parameters=None, nb_simul=200, scenar=True)
    # Make normal stochastic predict
    pred_normal = model.stochastic_predic(duration=150, parameters=None, nb_simul=200, scenar=False)

    # Compute mean predictions
    mean_scenar = np.mean(pred_scenar, axis=2)
    mean_normal = np.mean(pred_normal, axis=2)
    model.nb_simul = 200

    # Hospit
    fig, ax1 = plt.subplots()
    fig.suptitle('Hospit curves')
    ax1.plot(time, mean_scenar[:, 4], c='green', label='stocha mean')
    ax1.plot(time, mean_normal[:, 4], c='green', label='stocha mean')
    # Critical

    ax1.plot(time, mean_scenar[:, 5], c='orange', label='stocha mean')
    ax1.plot(time, mean_normal[:, 5], c='orange', label='stocha mean')

    #Fatalities
    ax1.plot(time, mean_scenar[:, 6], c='red', label='stocha mean')
    ax1.plot(time, mean_normal[:, 6], c='red', label='stocha mean')
    plt.hline(300)
    plt.hline(1500)
    ax1.legend()
    plt.savefig('imgJulien/lockdonw.png')
    plt.close()
