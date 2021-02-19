import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import binom


binom_obj = binom(n=100, p=0.5)

k = np.arange(0, 150)

probs = binom_obj.pmf(k)

plt.scatter(k, probs, c='blue')
plt.show()

for i in range(0, len(k)):
    print('{} - {}'.format(k[i], probs[i]))