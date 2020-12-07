import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def normal_density(sigma_sq, dx):

    return (np.exp(((dx ** 2) / sigma_sq) * (-0.5)) / np.sqrt(sigma_sq * 2 * np.pi))

x = np.linspace(-5, 20, 35)
y = []
ev = 12
for item in x:

    sigma_sq = np.fabs(ev)
    dx = np.fabs(item - ev)
    if sigma_sq == 0:
        sigma_sq = 1
    y.append(norm.ppf(normal_density(sigma_sq, dx)))

plt.scatter(x, y)
plt.show()

for i in range(0, len(x)):
    print('{} - {}'.format(x[i], y[i]))

print(np.log(0.00000000000000000001))

for i in range(0, 30):
    print(np.random.uniform(0.3, 0.8))