import numpy as np
import sklearn
import pandas as pd
import shapefile as shp
import seaborn as sns
from numpy import matrix
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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




if __name__ == '__main__':


    sns.set(style=”whitegrid”, palette=”pastel”, color_codes=True) sns.mpl.rc(“figure”, figsize=(10,6))
    #opening the vector map
    shp_path = “\\District_Boundary.shp”
    #reading the shape file by using reader function of the shape lib
    sf = shp.Reader(shp_path)
    len(sf.shapes())
    sf.records()

    # population = np.random.rand(1000000,2)
    #
    # plt.scatter(population[:,0], population[:,1])
    #
    #
    # plt.grid()
    #
    #
    # plt.xlabel('Day')
    # plt.ylabel('Active cases ')
    #
    #
    # # plt.show()
    #
    # plt.savefig('img/Population.png')
