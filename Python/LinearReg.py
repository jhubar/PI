import numpy as np
import sklearn
import pandas as pd

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
    return pd.read_csv(url)
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




if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv'
    data = load_from_url(url)
    X = np.array(data)
    print(X)

    # Load training data
    # LS = load_from_csv(args.ls)
    # Load test data
    # TS = load_from_csv(args.ts)


    # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = np.dot(X, np.array([1, 2])) + 3
    # reg = LinearRegression().fit(X, y)
    # print(reg.score(X, y))
    # print(reg.coef_)
    # print(reg.predict(np.array([[3, 5]])))
