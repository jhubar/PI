import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from scipy.stats import binom as binom

from smoothing import dataframe_smoothing
from plot import plot_dataset

from SEIR import SEIR
from bayesian import gaussian_inferences

if __name__ == "__main__":

    # Create the model:
    model = SEIR()
    gaussian_inferences(model.dataframe)
    #
    # model.plot(filename="Compare_stocha_and_deter.pdf",
    #            type='--sto-I --sto-E --sto-H --sto-C --sto-F' +
    #                 '--det-I --det-E --det-H --det-C --det-F' ,
    #            duration=200,
    #            plot_conf_inter=True)
    #
    # model.plot_fit_cumul(plot_conf_inter=True)
    # model.plot_fit_hosp(plot_conf_inter=True)
    # model.plot_fit_crit(plot_conf_inter=True)
    # model.plot_fit_death(plot_conf_inter=True)
    #
    # model.plot(filename="SEIR-MODEl(E,I,H,C,D).pdf",
    #            type='--sto-E --sto-I --sto-H --sto-C --sto-D',
    #            duration=200,
    #            global_view=True)
