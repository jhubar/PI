import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from scipy.integrate import odeint   # To integrate our equation
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import uncertainpy as un
import chaospy as cp                 # To create distributions
import json
import math
import random
from smooth import own_NRMAS_index, own_NRMAS
from plot import plot_current_data
from plot import cumul_positif_comp
from plot import cumul_hospit_comp
from plot import non_cum_hospit_comp
from plot import critical_com
from plot import fatal_com

def fit(self):
    """
    This method use the given data to find values of our model who minimise square error
    between predictions and original data.
    """
    # =========================================================================== #
    # PART 1: We fit the parameters beta, sigma and a temp version of gamma by computing the
    # sum of errors between the daily cumulative of positive tests and the
    # product of the I, H and R curves. All others parameters are set on zero.
    # So, in this first part, we are equivalent to a basic SEIR model
    # =========================================================================== #
    # Generate initial state:
    initial_state = self.get_initial_state()
    # Time vector:
    time = self.dataset[:, 0]
    # Bounds: Given ranges for beta, sigma and gamma
    bounds = [(0, 1), (1 / 5, 1), (1 / 10, 1 / 4)]
    # Start values
    start_values = [self.beta, self.sigma, self.gamma]
    # Use Scipy.optimize.minimize with L-BFGS_B method
    res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, 'first_part'),
                   method='L-BFGS-B', bounds=bounds)

    print(res)
    self.beta = res.x[0]
    self.sigma = res.x[1]
    self.gamma = res.x[2]

    # Compare data with predictions on cumulative positive
    predictions = self.predict(self.dataset.shape[0])
    # Store I + R
    cumul_positive = predictions[:, 3] + predictions[:, 5]

    cumul_positif_comp(self)


    # =========================================================================== #
    # PART 2: In our model, infectious people can go from I to H and not
    # only to R. So, we have to split a part of gamma parameter to hp parameter:
    # =========================================================================== #
    self.gamma_hp_slide()

    # Compare data with predictions on cumulative hospit
    predictions = self.predict(self.dataset.shape[0])
    # Store H
    cumul_hospit = predictions[:, 4]

    cumul_hospit_comp(self)

    # =========================================================================== #
    # PART 3: compute the probability to out of H.
    # WARNING: this probability contain the probability to be cure and the
    # probability to fall in ICU
    # =========================================================================== #
    self.fit_hcr()

    # Compare data with hospit data and non cumulative h curve
    predictions = self.predict(self.dataset.shape[0])
    # Store H
    hospit = predictions[:, 4]
    non_cum_hospit_comp(self)


    # =========================================================================== #
    # PART 4: People in H state can not only being cured. so we will distribute
    # the actual value of hcr in hcr (probability to be cured) and pc (probability
    # to fall in Critical cases). Because we don't have cumulative informations
    # about critical cases, we have to fit in the same time a probability pcr who
    # represent the probability to leave the critical state. So, we are optimizing
    # 1. the ratio of actual hcr who begin pc
    # 2. the value of pcr
    # =========================================================================== #
    # Generate initial state:
    initial_state = self.get_initial_state()
    # Time vector:
    time = self.dataset[:, 0]
    # Bounds: hcr/pc ratio, pcr
    bounds = [(0, 1), (0, 1)]
    # Start values
    start_values = [0.7, 0.1]
    # Use Scipy.optimize.minimize with L-BFGS_B method
    res = minimize(self.SSE, np.asarray(start_values), args=(initial_state, time, 'part_four'),
                   method='L-BFGS-B', bounds=bounds)
    print(res)
    initial_hcr = self.hcr
    self.hcr = res.x[0] * initial_hcr
    self.pc = (1 - res.x[0]) * initial_hcr
    self.pcr = res.x[1]

    # Compare data with critical
    predictions = self.predict(self.dataset.shape[0])
    # Store C
    critical = predictions[:, 7]

    critical_com(self)


    # =========================================================================== #
    # PART 5: We can now slide the actual value of pcr to spare the probability
    # to be cure from C (definitive pcr) and to die form C (pd)
    # =========================================================================== #
    self.pcr_pd_slide()

    # Compare data with critical
    predictions = self.predict(self.dataset.shape[0])
    # Store C
    fatalities = predictions[:, 8]

    fatal_com(self)
