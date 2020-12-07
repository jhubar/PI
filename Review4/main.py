from stats import *
from smooth import *
from plot import *



def data_preprocessing():
    _stats = stats()
    _stats.import_stats()
    print(_stats.age_df.age)

    plot_age(_stats.age_df.age,_stats.age_df.number)
    plot_households(_stats.households_df.category,_stats.households_df.number)
    # plot_schools(import_stats)



if __name__ == "__main__":
    data_preprocessing()
