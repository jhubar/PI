import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.stats as stats


num_sims = 5000

num_periods = 5

final_returns = []
for sim_num in range(num_sims):
    time = [0]
    returns_ = [0]

    for period in range(1,num_periods+1):
        time.append(period)
        returns_.append(returns_[period -1]+ stats.laplace.rvs(loc = 0.05, scale = 0.07, size = 1))

    final_returns.append(float(returns_[num_periods-1]))
    plt.plot(time,returns_)
plt.savefig('Plot/randomwalk.png')
