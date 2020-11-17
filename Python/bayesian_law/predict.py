import numpy as np
from scipy.integrate import odeint   # To integrate our equation


def __predict__(self, duration):
    """
    Predict epidemic curves from t_0 for the given duration
    """
    # Initialisation vector:
    initial_state = self.get_initial_state()
    # Time vector:
    time = np.arange(duration)
    # Parameters vector
    parameters = self.get_parameters()
    # Solve differential equations:
    predict = odeint(func=self.differential,
                     y0=initial_state,
                     t=time,
                     args=parameters)

    return np.vstack((time, predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3], predict[:, 4],
                      predict[:, 5], predict[:, 6], predict[:, 7])).T
