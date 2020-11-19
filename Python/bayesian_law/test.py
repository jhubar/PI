import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy import signal


uniform_dist = stats.uniform(loc=2, scale=3)
std = 0.25
normal_dist = stats.norm(loc=0, scale=std)

delta = 1e-4
big_grid = np.arange(-10,10,delta)

pmf1 = uniform_dist.pdf(big_grid)*delta
print("Sum of uniform pmf: "+str(sum(pmf1)))

pmf2 = normal_dist.pdf(big_grid)*delta
print("Sum of normal pmf: "+str(sum(pmf2)))


conv_pmf = signal.fftconvolve(pmf1,pmf2,'same')
print("Sum of convoluted pmf: "+str(sum(conv_pmf)))

pdf1 = pmf1/delta
pdf2 = pmf2/delta
conv_pdf = conv_pmf/delta
print("Integration of convoluted pdf: " + str(np.trapz(conv_pdf, big_grid)))


plt.plot(big_grid,pdf1, label='Uniform')
plt.plot(big_grid,pdf2, label='Gaussian')
plt.plot(big_grid,conv_pdf, label='Sum')
plt.legend(loc='best'), plt.suptitle('PDFs')
plt.savefig('Plot/test.png')
