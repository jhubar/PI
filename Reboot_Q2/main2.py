import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from SEIR import SEIR

"""
Adaptation du main de base pour ploter les graphes et interval de confiance sans mesures
"""

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

    model.plot('fig/base_pred_A', type='--det-I --sto-I --det-C --sto-C --det-H --sto-H --det-D --sto-D', duration=200, plot_conf_inter=True)

    model.plot('fig/base_pred_B', type='--det-I --sto-I --det-C --sto-C --det-H --sto-H --det-D --sto-D', duration=200,
               plot_conf_inter=False, global_view=True)

    model.plot('fig/base_pred_C', type='--det-S --sto-S --det-R --sto-R', duration=200, plot_conf_inter=True)

    model.plot('fig/base_pred_B', type='--det-S --sto-S --det-R --sto-R', duration=200,
               plot_conf_inter=False, global_view=True)

