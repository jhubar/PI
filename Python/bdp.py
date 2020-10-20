import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import array as ar
import time


from scipy.integrate import solve_ivp

url = 'https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv'
df = pd.read_csv(url, error_bad_lines=False)

# we evaluate the EDO for those value of t
t_eval_ = ar.array('i', range(0, df.shape[0]))
'''
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.scatter(df['Day'], df['num_positive'])  # Plot some data on the axes.
'''
# FAIRE UN JACOBIEN
def EDOs(t, z, beta, lamb, mu, v):
    I, S, R, C = z
    return [(beta * I * S) - (I / lamb) - (mu * I), -(beta * I * S), I/lamb, (beta * I * S) - (C / v)]

'''	
sol = solve_ivp(EDOs, [0, df.shape[0]], [1, 999999, 0, 0], args=(0.1, , 0.1, 0.1),
                dense_output=True, t_eval=t_eval_, method='RK45')


t = np.linspace(0, 30, 300)
z = sol.sol(t)
plt.plot(t, z.T)
plt.xlabel('t')
plt.legend(['I', 'S', 'R', 'C'])
plt.title('model IS')
plt.show()
'''
def determine_best_fitting_attributes():
    
    # attention div by 0
    beta = np.linspace(0.5, 1.5, num=10)
    lamb = np.linspace(0.01, 0.5, num=10)
    mu = np.linspace(0.01, 10, num=2)
    v = np.linspace(0.01, 0.25, num=2)
    
    beta_opt = 0
    lamb_opt = 0
    mu_opt = 0
    v_opt = 0
    
    ms_min = float('inf')
    
    for b in beta:
        print("b" + str(b))
        for l in lamb:
            for m in mu:
                for v_ in v:
                    
                    ms = 0
                    
                    sol = solve_ivp(EDOs, [0, df.shape[0]], [1, 999999, 0, 0],
                                    args=(b, l, m, v_),
                                    dense_output=True, t_eval=t_eval_)
                    
                    # fit data with total number of infected
                    for i in range(sol.y.shape[1]):
                        ms += ((sol.y[0][i] - df['num_positive'][i]) ** 2)
                        
                    if(ms < ms_min):
                        ms_min = ms
                        beta_opt = b
                        lamb_opt = l
                        mu_opt = m
                        v_opt = v_
    
    return beta_opt, lamb_opt, mu_opt, v_opt, ms_min

def add_total_positive_atm(df, spreading_period):
    
    new_c = []
    
    for i in range(len(df['num_positive'])):
        
        tot = 0
        
        for j in range(0, spreading_period + 1):
            
            if(i - j >= 0):
                tot += df['num_positive'][i-j]
                
        new_c.append(tot)

    df['Total_positive_atm'] = new_c
        

def add_susceptible_atm(df, initial_population):
    
    new_c = []
    
    for i in range(len(df['num_positive'])):
        
        tot = initial_population
        
        for j in range(0, i):
            tot -= df['num_positive'][j]
            
        new_c.append(tot)
        
    df['Total_susceptible_atm'] = new_c
    
add_total_positive_atm(df, 5)
add_susceptible_atm(df, 1000000)

 
start = time.time()
beta_opt, lamb_opt, mu_opt, v_opt, ms_min = determine_best_fitting_attributes()
print("Algorithm took : " + str((time.time() - start)/60) +
      " minutes, " + str((time.time() - start) % 60) + " secondes")

print(beta_opt)
print("\n")
print(lamb_opt)
print("\n")
print(mu_opt)
print("\n")
print(v_opt)
print("\n")
print(ms_min)
print("\n")













