import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from SEIR import SEIR

if __name__ == "__main__":
    nb_sim = 100
    # --------------------------- Create the model --------------------------- #
    print('Phase 1: ')
    # Create the model:
    model = SEIR()

    # Load the dataset
    model.import_dataset()

    # Fit the model
    #model.fit(method='normal')

    # Fit starting state:
    #model.init_state_optimizer()

    # Make stochastic predictions:
    pred_sto = model.stochastic_predic(duration=model.dataset.shape[0], nb_simul=nb_sim)
    # Get the mean
    pred_sto_mean = np.mean(pred_sto, axis=2)
    # Make deterministic predictions:
    pred_det = model.predict(duration=model.dataset.shape[0])

    # Plot results for cumulative conta:
    time = range(0, model.dataset.shape[0])
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 7, i]*model.t*model.s, color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 7]*model.t*model.s, color='green', label='Stochastic conta')
    plt.plot(time, pred_det[:, 7]*model.t*model.s, color='red', label='Deterministic conta')
    plt.scatter(time, model.dataset[:, 7], color='blue', label='Testing data')
    plt.legend()
    plt.title('Cumulative testing data vs pred')
    #plt.show()
    plt.savefig('fig/cum_test_70.png')
    plt.close()

    # For hospit
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 4, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 4], color='green', label='Hospit pred stocha')
    plt.plot(time, pred_det[:, 4], color='red', label='Hospit pred deter')
    plt.scatter(time, model.dataset[:, 3], color='blue', label='Hospit data')
    plt.legend()
    plt.title('Hospit: pred stocha/deter vs data')
    #plt.show()
    plt.savefig('fig/hospit_70.png')
    plt.close()

    # For Criticals
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 5, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 5], color='green', label='Critical pred stocha')
    plt.plot(time, pred_det[:, 5], color='red', label='Critical pred deter')
    plt.scatter(time, model.dataset[:, 5], color='blue', label='Critical data')
    plt.legend()
    plt.title('Critical: pred stocha/deter vs data')
    #plt.show()
    plt.savefig('fig/critical_70.png')
    plt.close()

    # For Fatalities
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 6, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 6], color='green', label='Fatalities pred stocha')
    plt.plot(time, pred_det[:, 6], color='red', label='Fatalities pred deter')
    plt.scatter(time, model.dataset[:, 6], color='blue', label='Fatalities data')
    plt.legend()
    plt.title('Fatalities: pred stocha/deter vs data')
    # plt.show()
    plt.savefig('fig/fatal_70.png')
    plt.close()

    # Print parameters:
    prm = model.param_translater(method='dict')
    print(prm)
    print('I_0 = {}'.format(model.I_0))
    print('E_0 = {}'.format(model.E_0))

    # --------------------------- Load assistant realization --------------------------- #

    url = 'https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/Cov_invaders.csv'
    # Import the dataframe:
    delau_data = pd.read_csv(url, sep=',', header=0)
    # Get cumul positive test column
    cumul_positive = np.copy(delau_data['num_positive'].to_numpy())
    for i in range(1, len(cumul_positive)):
        cumul_positive[i] += cumul_positive[i - 1]
    delau_data.insert(7, 'cumul_positive', cumul_positive)
    delau_np = delau_data.to_numpy()


    # --------------------------- 190 day with no measures ----------------------------- #
    print('Phase 2: ')
    # Make stochastic predictions:

    pred_sto = model.stochastic_predic(duration=delau_np.shape[0], nb_simul=nb_sim)
    # Get the mean
    pred_sto_mean = np.mean(pred_sto, axis=2)
    # Make deterministic predictions:
    pred_det = model.predict(duration=delau_np.shape[0])

    # Plot results for cumulative conta:
    time = np.arange(delau_np.shape[0])
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 7, i]*model.t*model.s, color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 7]*model.t*model.s, color='green', label='Stochastic conta')
    plt.plot(time, pred_det[:, 7]*model.t*model.s, color='red', label='Deterministic conta')
    plt.scatter(time, delau_np[:, 7], color='blue', label='Testing data')
    plt.legend()
    plt.title('Cumulative testing data vs pred')
    #plt.show()
    plt.savefig('fig/cum_test_190.png')
    plt.close()

    # For hospit
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 4, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 4], color='green', label='Hospit pred stocha')
    plt.plot(time, pred_det[:, 4], color='red', label='Hospit pred deter')
    plt.scatter(time, delau_np[:, 3], color='blue', label='Hospit data')
    plt.legend()
    plt.title('Hospit: pred stocha/deter vs data')
    #plt.show()
    plt.savefig('fig/hospit_190.png')
    plt.close()

    # For Criticals
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 5, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 5], color='green', label='Critical pred stocha')
    plt.plot(time, pred_det[:, 5], color='red', label='Critical pred deter')
    plt.scatter(time, delau_np[:, 5], color='blue', label='Critical data')
    plt.legend()
    plt.title('Critical: pred stocha/deter vs data')
    #plt.show()
    plt.savefig('fig/critical_190.png')
    plt.close()

    # For Fatalities
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 6, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 6], color='green', label='Fatalities pred stocha')
    plt.plot(time, pred_det[:, 6], color='red', label='Fatalities pred deter')
    plt.scatter(time, delau_np[:, 6], color='blue', label='Fatalities data')
    plt.legend()
    plt.title('Fatalities: pred stocha/deter vs data')
    # plt.show()
    plt.savefig('fig/fatal_190.png')
    plt.close()

    # --------------------------- 20 days of mask + social d ----------------------------- #
    print('Phase 3: ')
    scenario_1 = {
        'duration': 191,
        'social_dist': [75, 128, 6],
        'wearing_mask': [75, 128]
    }
    # Put the scenario in the model:
    model.set_scenario(scenario_1)

    # Make stochastic predictions:
    pred_sto = model.stochastic_predic(duration=delau_np.shape[0], nb_simul=nb_sim, scenar=True)
    # Get the mean
    pred_sto_mean = np.mean(pred_sto, axis=2)
    # Make deterministic predictions:
    pred_det = model.predict(duration=delau_np.shape[0])

    # Plot results for cumulative conta:
    time = np.arange(delau_np.shape[0])
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 7, i]*model.t*model.s, color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 7]*model.t*model.s, color='green', label='Stochastic conta')
    plt.plot(time, pred_det[:, 7]*model.t*model.s, color='red', label='Deterministic conta')
    plt.scatter(time, delau_np[:, 7], color='blue', label='Testing data')
    plt.legend()
    plt.title('Cumulative testing data vs pred')
    #plt.show()
    plt.savefig('fig/scenar_1_cum_test_190.png')
    plt.close()

    # For hospit
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 4, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 4], color='green', label='Hospit pred stocha')
    plt.plot(time, pred_det[:, 4], color='red', label='Hospit pred deter')
    plt.scatter(time, delau_np[:, 3], color='blue', label='Hospit data')
    plt.legend()
    plt.title('Hospit: pred stocha/deter vs data')
    #plt.show()
    plt.savefig('fig/scenar_1_hospit_190.png')
    plt.close()

    # For Criticals
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 5, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 5], color='green', label='Critical pred stocha')
    plt.plot(time, pred_det[:, 5], color='red', label='Critical pred deter')
    plt.scatter(time, delau_np[:, 5], color='blue', label='Critical data')
    plt.legend()
    plt.title('Critical: pred stocha/deter vs data')
    #plt.show()
    plt.savefig('fig/scenar_1_critical_190.png')
    plt.close()

    # For Fatalities
    for i in range(0, nb_sim):
        plt.plot(time, pred_sto[:, 6, i], color='limegreen', linewidth=0.1)
    plt.plot(time, pred_sto_mean[:, 6], color='green', label='Fatalities pred stocha')
    plt.plot(time, pred_det[:, 6], color='red', label='Fatalities pred deter')
    plt.scatter(time, delau_np[:, 6], color='blue', label='Fatalities data')
    plt.legend()
    plt.title('Fatalities: pred stocha/deter vs data')
    # plt.show()
    plt.savefig('fig/scenar_1_fatal_190.png')
    plt.close()
