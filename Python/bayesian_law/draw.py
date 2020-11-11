import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import random


nb_sympt_tested = random.uniform(0.5, 1.0)
nb_not_sympt_tested = 1-nb_sympt_tested

sensitivity = random.uniform(0.7, 0.85)
not_sensitivity = 1 - sensitivity



test_y = np.array([0]*int(nb_sympt_tested*100) + [1]*int(nb_not_sympt_tested*100))

predicted_y_probs = np.concatenate((np.random.beta(not_sensitivity*10,sensitivity*10,int(nb_sympt_tested*100)), np.random.beta(sensitivity*10,not_sensitivity*10,int(nb_not_sympt_tested*100))))



def estimate_beta(X):
    xbar = np.mean(X)
    vbar = np.var(X,ddof=1)
    alphahat = xbar*(xbar*(1-xbar)/vbar - 1)
    betahat = (1-xbar)*(xbar*(1-xbar)/vbar - 1)
    return alphahat, betahat

positive_beta_estimates = estimate_beta(predicted_y_probs[test_y == 1])
negative_beta_estimates = estimate_beta(predicted_y_probs[test_y == 0])

unit_interval = np.linspace(0,1,100)
plt.plot(unit_interval, scipy.stats.beta.pdf(unit_interval, *positive_beta_estimates), c='g', label="positive")
plt.plot(unit_interval, scipy.stats.beta.pdf(unit_interval, *negative_beta_estimates), c='r', label="negative")

# Show the threshold.
plt.axvline(sensitivity, c='black', ls='dashed')
plt.xlim(0,1)

# Add labels
plt.legend()

plt.savefig('p_tested_Sym.png')
