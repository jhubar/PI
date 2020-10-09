import numpy as np
import sklearn
import pandas as pd
from numpy import matrix
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    print("{}...".format(label))
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))
def load_from_url(url, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the url csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(url,delimiter=',')

def tune(model, X, y, tune_input = 10):
    """Apply the cross validation algorithm to a model.

    Parameters
    ----------
    model :
        Supervised learning model to be evaluated.
    X : array [n_cases, ..]
        Training input.
    y: array [n_cases, 1]
        Training output.
    tune_input : int (default=10)
        Number of divisions in the cross validation algorithm.

    Return
    ------=
    score : float
        The fraction of correctly classified samples.
    """
    cross_val = cross_val_score(model, X, y, cv = tune_input,
                                scoring = 'roc_auc', n_jobs=-1)
    print(cross_val)
    print(np.mean(cross_val))

def test_accuracy(model, X_LS, y_LS, X_TS, y_TS):
    model.fit(X_LS, y_LS)

    y_pred = model.predict(X_TS)

    print(roc_auc_score(y_pred, y_TS))

def load_current_casses(X_data):

    X = (np.array(([i[0] for i in X_data]))).reshape(-1,1)
    y = np.array(([i[1] for i in X_data])).reshape(-1,1)

    return X,y




if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv'
    data = load_from_url(url)
    data_array = np.array(data)
    X = np.delete(data_array, [2,3,4,5], axis=1)

    # Load the current casses dataset

    current_casses_X, current_casses_y = load_current_casses(X)

    porportion_ls_ts = current_casses_X.size-int(2*(current_casses_X.size/3))
    # Use only one feature
    # current_casses_X = current_casses_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    current_casses_X_train = current_casses_X[:-porportion_ls_ts]
    current_casses_X_test = current_casses_X[-porportion_ls_ts:]

    # Split the targets into training/testing sets
    current_casses_y_train = current_casses_y[:-porportion_ls_ts]
    current_casses_y_test = current_casses_y[-porportion_ls_ts:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(current_casses_X_train, current_casses_y_train)

    # Make predictions using the testing set
    current_casses_y_pred = regr.predict(current_casses_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(current_casses_y_test, current_casses_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(current_casses_y_test, current_casses_y_pred))

    # Plot outputs
    plt.scatter(current_casses_X_test, current_casses_y_test,  color='black')
    plt.plot(current_casses_X_test, current_casses_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
