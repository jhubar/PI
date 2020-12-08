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

        self.small_schools_df = None
        self.medium_schools_df = None
        self.large_schools_df = None

        self.mean_medium_schools = 0
        self.mean_large_schools = 0

        self.mean_small_schools_people_meet = 0
        self.mean_medium_schools_people_meet = 0
        self.mean_large_schools_people_meet = 0

        self.small_schools_df = None
        self.medium_schools_df = None
        self.large_schools_df = None

        self.mean_medium_schools = 0
        self.mean_large_schools = 0

        self.mean_small_schools_people_meet = 0
        self.mean_medium_schools_people_meet = 0
        self.mean_large_schools_people_meet = 0



    def import_stats(self):
        self.age_df = pd.read_csv(self.age_path,sep = ';')
        self.households_df = pd.read_csv(self.households_path,sep = ';')
        self.schools_df = pd.read_csv(self.schools_path,sep = ';')
        self.workplaces_df = pd.read_csv(self.workplaces_path,sep = ';')

    def plot_stats(self):
        plot_age(self.age_df.age,self.age_df.number)
        plot_households(self.households_df.category
                                    ,self.households_df.number)
        plot_schools(self.schools_df.size_of_schools
                                    ,self.schools_df.number)
        plot_workplaces(self.workplaces_df.size_of_workplaces
                                    ,self.workplaces_df.number)

        plot_medium_workplaces(self.medium_workplaces_df.size_of_workplaces
                                    ,self.medium_workplaces_df.number
                                    ,self.mean_medium_workplaces)

        plot_large_workplaces(self.large_workplaces_df.size_of_workplaces
                                ,self.large_workplaces_df.number
                                ,self.mean_large_workplaces)


        plot_schools(self.schools_df.size_of_schools
                                    ,self.schools_df.number)

        plot_small_schools(self.small_schools_df.size_of_schools
                                    ,self.small_schools_df.number
                                    ,self.mean_small_schools)
        plot_medium_schools(self.medium_schools_df.size_of_schools
                                    ,self.medium_schools_df.number
                                    ,self.mean_medium_schools)
        plot_large_schools(self.large_schools_df.size_of_schools
                                    ,self.large_schools_df.number
                                    ,self.mean_large_schools)

    def data_preprocessing_age(self):
        _age = self.age_df.to_numpy()
        _age[:,1] = own_NRMAS_age(_age[:,1],19)
        self.age_df = pd.DataFrame({'age': _age[:, 0], 'number': _age[:, 1]})

    def data_preprocessing_households(self):
        _houseHolds = self.households_df.to_numpy()
        _houseHolds[:,1] = own_NRMAS_houseHolds(_houseHolds[:,1],3)
        self.houseHolds_df = pd.DataFrame({'category': _houseHolds[:, 0],
                                                   'number': _houseHolds[:, 1]})


    def data_preprocessing_workplaces(self):
        _workplaces = self.workplaces_df.to_numpy()
        small_workplaces = _workplaces[1,1]
        medium_workplaces_x = []
        medium_workplaces_y = []
        large_workplaces_x = []
        large_workplaces_y = []

        for i in range(1,10):
            medium_workplaces_x.append(5+10*i)
            medium_workplaces_y.append(_workplaces[i,1])

        for i in range(10,_workplaces[:,1].size):
            large_workplaces_x.append(_workplaces[i,2])
            large_workplaces_y.append(_workplaces[i,1])


        self.mean_medium_workplaces = np.sum(
                                medium_workplaces_y)/len(medium_workplaces_x)
        self.mean_large_workplaces = np.sum(
                                large_workplaces_y)/len(large_workplaces_x)

        """
        compute the mean of people meet in the SMALL workplaces
        """
        self.mean_small_workplaces_people_meet = small_workplaces*5
        """
        compute the mean of people meet in the MEDIUM workplaces
        """
        self.mean_medium_workplaces_people_meet = np.sum(
                                [(medium_workplaces_x[i]*medium_workplaces_y[i])
                                /len(medium_workplaces_x)
                                for i in range(0, len(medium_workplaces_x))])
        """
        compute the mean of people meet in the LARGE workplaces
        """
        self.mean_large_workplaces_people_meet = np.sum(
                                [(large_workplaces_x[i]*large_workplaces_y[i])
                                /len(large_workplaces_x)
                                for i in range(0, len(large_workplaces_x))])



        _workplacesUpdated_number = np.array([small_workplaces,
                                    np.sum(medium_workplaces_y),
                                    np.sum(large_workplaces_y)])
        _workplacesUpdated_category = np.array(["small","medium","large"])

        self.workplaces_df = pd.DataFrame(
                    {'size_of_workplaces': _workplacesUpdated_category,
                    'number': _workplacesUpdated_number})

        self.medium_workplaces_df = pd.DataFrame(
                    {'size_of_workplaces': medium_workplaces_x,
                    'number': medium_workplaces_y})

        self.large_workplaces_df = pd.DataFrame(
                    {'size_of_workplaces': large_workplaces_x,
                    'number': large_workplaces_y})

    def data_preprocessing_schools(self):
        _schools = self.schools_df.to_numpy()
        small_schools= _schools[1,1]
        small_schools_x = []
        small_schools_y = []
        medium_schools_x = []
        medium_schools_y = []
        large_schools_x = []
        large_schools_y = []

        for i in range(1,10):
            small_schools_x.append(_schools[i,2])
            small_schools_y.append(_schools[i,1])

        for i in range(10,50):
            medium_schools_x.append(_schools[i,2])
            medium_schools_y.append(_schools[i,1])

        for i in range(50,_schools[:,1].size):
            large_schools_x.append(_schools[i,2])
            large_schools_y.append(_schools[i,1])

        self.mean_small_schools = np.sum(
                                small_schools_y)/len(small_schools_x)
        self.mean_medium_schools = np.sum(
                                medium_schools_y)/len(medium_schools_x)
        self.mean_large_schools = np.sum(
                                large_schools_y)/len(large_schools_x)

        """
        compute the mean of people meet in the SMALL schools
        """
        self.mean_small_schools_people_meet = np.sum(
                                [(small_schools_x[i]*small_schools_y[i])
                                /len(small_schools_x)
                                for i in range(0, len(small_schools_x))])
        """
        compute the mean of people meet in the MEDIUM schools
        """
        self.mean_medium_schools_people_meet = np.sum(
                                [(medium_schools_x[i]*medium_schools_y[i])
                                /len(medium_schools_x)
                                for i in range(0, len(medium_schools_x))])
        """
        compute the mean of people meet in the LARGE schools
        """
        self.mean_large_schools_people_meet = np.sum(
                                [(large_schools_x[i]*large_schools_y[i])
                                /len(large_schools_x)
                                for i in range(0, len(large_schools_x))])



        _schoolsUpdated_number = np.array([np.sum(small_schools_y),
                                    np.sum(medium_schools_y),
                                    np.sum(large_schools_y)])
        _schoolsUpdated_category = np.array(["small:[0-100]","medium:[100-500]","large:[500+]"])

        self.schools_df = pd.DataFrame(
                    {'size_of_schools': _schoolsUpdated_category,
                    'number': _schoolsUpdated_number})
        self.small_schools_df = pd.DataFrame(
                    {'size_of_schools': small_schools_x,
                    'number': small_schools_y})

        self.medium_schools_df = pd.DataFrame(
                    {'size_of_schools': medium_schools_x,
                    'number': medium_schools_y})

        self.large_schools_df = pd.DataFrame(
                    {'size_of_schools': large_schools_x,
                    'number': large_schools_y})

    def data_preprocessing(self):
        self.import_stats()
        self.data_preprocessing_age()
        self.data_preprocessing_households()
        self.data_preprocessing_workplaces()
        self.data_preprocessing_schools()
        self.plot_stats()
