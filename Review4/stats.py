import pandas as pd
import numpy as np
from plot import *
from smooth import *

class stats():

    def __init__(self):
        self.age_path = "stats/age.csv"
        self.households_path = "stats/households.csv"
        self.schools_path = "stats/schools.csv"
        self.workplaces_path = "stats/workplaces.csv"

        self.age_df = None
        self.households_df = None
        self.schools_df = None
        self.workplaces_df = None

    def import_stats(self):
        self.age_df = pd.read_csv(self.age_path,sep = ';')
        self.households_df = pd.read_csv(self.households_path,sep = ';')
        self.schools_df = pd.read_csv(self.schools_path,sep = ';')
        self.workplaces_df = pd.read_csv(self.workplaces_path,sep = ';')

    def plot_stats(self):
        self.import_stats()
        plot_age(self.age_df.age,self.age_df.number)
        plot_households(self.households_df.category,self.households_df.number)
        plot_schools(self.schools_df.size_of_schools,self.schools_df.number)
        plot_workplaces(self.workplaces_df.size_of_workplaces,self.workplaces_df.number)

    def data_preprocessing(self):
        self.import_stats()
        _age = self.age_df.to_numpy()

        plot_age(self.age_df.age,self.age_df.number)
        plot_age2(_age[:,0],own_NRMAS_age(_age[:,1],19))
