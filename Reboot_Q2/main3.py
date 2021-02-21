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



    # --------------------------- 20 days of mask + social d ----------------------------- #
    print('Phase 3: ')
    scenario_1 = {
        'duration': 191,
        'social_dist': [73, 190, 6],
        'wearing_mask': [73, 190]
    }
    # Put the scenario in the model:
    model.set_scenario(scenario_1)

    nb_sim = 200

    # Make stochastic predictions:
    pred_sto = model.stochastic_predic(duration=delau_np.shape[0], nb_simul=nb_sim, scenar=True)
    # Get the mean
    pred_sto_mean = np.mean(pred_sto, axis=2)
    # Get std
    pred_sto_std = np.std(pred_sto, axis=2)
    # Get higher and lower bound of confidence interval
    sto_hq = pred_sto_mean + (2 * pred_sto_std)
    sto_lq = pred_sto_mean - (2 * pred_sto_std)

    # Make deterministic predictions:
    pred_det = model.predict(duration=delau_np.shape[0])

    # Plot results for cumulative conta:
    time = np.arange(delau_np.shape[0])


    plt.fill_between(time, sto_hq[:, 7], sto_lq[:, 7], color='lavender', alpha=0.7)
    plt.plot(time, pred_sto_mean[:, 7]*model.t*model.s, color='black', label='Stochastic cumulative conta. Mean')
    plt.scatter(time, delau_np[:, 7], color='blue', label='Testing Cumulative data')
    plt.legend()
    plt.title('Cumulative testing: data vs predictions')
    plt.xlabel('Time in days')
    #plt.show()
    plt.savefig('fig/scenar_1_cum_test_190_confidence.png')
    plt.close()

    # For hospit
    plt.fill_between(time, sto_hq[:, 4], sto_lq[:, 4], color='lavender', alpha=0.7)
    plt.plot(time, pred_sto_mean[:, 4], color='black', label='Hospit pred stocha')
    plt.scatter(time, delau_np[:, 3], color='blue', label='Hospit data')
    plt.legend()
    plt.title('Hospitalizations: data vs predictions')
    plt.xlabel('Time in days')
    #plt.show()
    plt.savefig('fig/scenar_1_hospit_190_confidence.png')
    plt.close()

    # For Criticals
    plt.fill_between(time, sto_hq[:, 5], sto_lq[:, 5], color='lavender', alpha=0.7)
    plt.plot(time, pred_sto_mean[:, 5], color='green', label='Critical pred stocha')
    plt.scatter(time, delau_np[:, 5], color='blue', label='Critical data')
    plt.legend()
    plt.title('Critical: data vs prediction')
    plt.xlabel('Time in days')
    #plt.show()
    plt.savefig('fig/scenar_1_critical_190_confidence.png')
    plt.close()

    # For Fatalities
    plt.fill_between(time, sto_hq[:, 6], sto_lq[:, 6], color='lavender', alpha=0.7)
    plt.plot(time, pred_sto_mean[:, 6], color='green', label='Fatalities pred stocha')
    plt.scatter(time, delau_np[:, 6], color='blue', label='Fatalities data')
    plt.legend()
    plt.title('Fatalities: data vs predictions')
    plt.xlabel('Time in days')
    # plt.show()
    plt.savefig('fig/scenar_1_fatal_190_confidence.png')
    plt.close()
