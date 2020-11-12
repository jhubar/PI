import uncertainpy as un
import chaospy as cp
import numpy as np
from scipy.integrate import odeint

# Create the coffee cup model function
def coffee_cup(kappa, T_env):
    # Initial temperature and time array
    time = np.linspace(0, 200, 150)            # Minutes
    T_0 = 95                                   # Celsius

    # The equation describing the model
    def f(T, time, kappa, T_env):
        return -kappa*(T - T_env)

    # Solving the equation by integration.
    temperature = odeint(f, T_0, time, args=(kappa, T_env))[:, 0]

    # Return time and model output
    return time, temperature

# Create a model from the coffee_cup function and add labels
model = un.Model(run=coffee_cup, labels=["Time (min)", "Temperature (C)"])

# Create the distributions
kappa_dist = cp.Uniform(0.025, 0.075)
T_env_dist = cp.Uniform(15, 25)

# Define the parameter dictionary
parameters = {"kappa": kappa_dist, "T_env": T_env_dist}

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters)

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
data = UQ.quantify()
