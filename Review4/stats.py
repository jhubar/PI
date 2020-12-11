import pandas as pd
import numpy as np
from plot import *
from smooth import *
import json

class stats():

    def __init__(self):
        self.age_path = "stats/age.csv"
        self.households_path = "stats/households.csv"
        self.schools_path = "stats/schools.csv"
        self.workplaces_path = "stats/workplaces.csv"
        self.communities_path = "stats/communities.csv"

        self.age_df = None
        self.households_df = None
        self.schools_df = None
        self.workplaces_df = None
        self.communities_df = None


        self.nbWorker = 0
        self.nbStudent = 0
        self.nbOther = 0
        """
        workplaces
        """
        self._workplaces_df = None
        self.small_workplaces_df = None
        self.medium_workplaces_df = None
        self.large_workplaces_df = None

        self.mean_workplaces = 0
        self.mean_small_workplaces = 0
        self.mean_medium_workplaces = 0
        self.mean_large_workplaces = 0

        self.mean_workplaces_people_meet = 0
        self.mean_small_workplaces_people_meet = 0
        self.mean_medium_workplaces_people_meet = 0
        self.mean_large_workplaces_people_meet = 0
        """
        schools
        """
        self._schools_df = None
        self.small_schools_df = None
        self.medium_schools_df = None
        self.large_schools_df = None

        self.mean_schools =0
        self.mean_small_schools = 0
        self.mean_medium_schools = 0
        self.mean_large_schools = 0

        self._small_schools_people_meet = 0
        self.mean_small_schools_people_meet = 0
        self.mean_medium_schools_people_meet = 0
        self.mean_large_schools_people_meet = 0





    def import_stats(self):
        self.age_df = pd.read_csv(self.age_path,sep = ';')
        self.households_df = pd.read_csv(self.households_path,sep = ';')
        self.schools_df = pd.read_csv(self.schools_path,sep = ';')
        self.workplaces_df = pd.read_csv(self.workplaces_path,sep = ';')
        self.communities_df = pd.read_csv(self.communities_path,sep = ';')

    def plot_stats(self):
        plot_pir()
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

        # _age[:,1] = own_NRMAS_age(_age[:,1],19)
        young_people_x = []
        young_people_y = []

        junior_people_x =[]
        junior_people_y =[]

        medior_people_x =[]
        medior_people_y =[]

        seignior_people_x =[]
        seignior_people_y =[]

        for i in range(0,6):
            young_people_x.append(_age[i,0])
            young_people_y.append(_age[i,1])

        for i in range(6,23):
            junior_people_x.append(_age[i,0])
            junior_people_y.append(_age[i,1])

        for i in range(23,65):
            medior_people_x.append(_age[i,0])
            medior_people_y.append(_age[i,1])

        for i in range(65,_age[:,1].size):
            seignior_people_x.append(_age[i,0])
            seignior_people_y.append(_age[i,1])

        # print("###############################################################")
        # print("young: " + str(np.sum(young_people_y)))
        # print("Junior: " + str(np.sum(junior_people_y)))
        # print("Medior: " + str(np.sum(medior_people_y)))
        # print("Seignior: " + str(np.sum(seignior_people_y)))
        # print("tot: " + str( np.sum(_age[:,1])  ))
        # print("###############################################################")
        #
        #


        self.age_df = pd.DataFrame({'age': _age[:, 0], 'number': _age[:, 1]})

    def data_preprocessing_households(self):
        _houseHolds = self.households_df.to_numpy()
        p_1 = 0.32
        p_2 = 0.277
        p_3 = 0.16
        p_4 = 0.121
        p_5 = 0.063
        p_6 = 0.03
        p_7 = 0.029

        somme_y = p_3+p_4+p_5+p_6+p_7
        somme_j = p_2+p_3+p_4+p_5+p_6+p_7
        somme_m = p_1+p_2+p_3+p_4+p_5+p_6+p_7
        somme_s = p_1+p_2

        p_y = 0.075
        p_j = 0.217
        p_m = 0.587
        p_s = 0.122

        print("###############################################################")




        household_mean_y = (((p_3/somme_y)*3)+
                            ((p_4/somme_y)*4)+
                            ((p_5/somme_y)*5)+
                            ((p_6/somme_y)*6)+
                            ((p_7/somme_y)*7))

        print("mean for young households: "+ str(household_mean_y))
        print("------------------------------------")


        household_mean_j = (((p_2/somme_j)*2)+
                            ((p_3/somme_j)*3)+
                            ((p_4/somme_j)*4)+
                            ((p_5/somme_j)*5)+
                            ((p_6/somme_j)*6)+
                            ((p_7/somme_j)*7))

        print("mean for junior households: "+ str(household_mean_j))
        print("------------------------------------")
        household_mean_m = (((p_1/somme_m)*1)+
                            ((p_2/somme_m)*2)+
                            ((p_3/somme_m)*3)+
                            ((p_4/somme_m)*4)+
                            ((p_5/somme_m)*5)+
                            ((p_6/somme_m)*6)+
                            ((p_7/somme_m)*7))

        print("mean for medior households: "+ str(household_mean_m))

        print("------------------------------------")


        household_mean_s = (((p_1/somme_s)*1)
                           +((p_2/somme_s)*2))

        print("mean for seignior households: "+ str(household_mean_s))


        print("###############################################################")



        plot_pie_households()

        self.houseHolds_df = pd.DataFrame({'category': _houseHolds[:, 0],
                                                   'number': _houseHolds[:, 1]})



    def data_preprocessing_workplaces(self):
        _workplaces = self.workplaces_df.to_numpy()
        # print("nb worker " +str(np.sum(_workplaces[:,1])))
        small_workplaces = _workplaces[1,1]
        _workplaces_x = []
        _workplaces_y = []
        small_workplaces_x = []
        small_workplaces_y = []
        medium_workplaces_x = []
        medium_workplaces_y = []
        large_workplaces_x = []
        large_workplaces_y = []

        small_workplaces_x =_workplaces[0,2]
        small_workplaces_y =_workplaces[0,1]

        for i in range(0,_workplaces[:,1].size):
            _workplaces_x.append(_workplaces[i,2])
            _workplaces_y.append(_workplaces[i,1])

        for i in range(1,10):
            medium_workplaces_x.append(_workplaces[i,2])
            medium_workplaces_y.append(_workplaces[i,1])

        for i in range(10,_workplaces[:,1].size):
            large_workplaces_x.append(_workplaces[i,2])
            large_workplaces_y.append(_workplaces[i,1])


        plot_pie_workplaces()

        self.mean_workplaces = np.sum(
                                _workplaces_y)/len(_workplaces_x)
        self.mean_small_workplaces = small_workplaces_y/small_workplaces_x
        self.mean_medium_workplaces = np.sum(
                                medium_workplaces_y)/len(medium_workplaces_x)
        self.mean_large_workplaces = np.sum(
                                large_workplaces_y)/len(large_workplaces_x)

        """
        compute the mean of people meet in the SMALL workplaces
        """
        self.mean_small_workplaces_people_meet = small_workplaces_x
        """
        compute the mean of people meet in the MEDIUM workplaces

        """
        # print(medium_workplaces_x)
        print((np.sum(
        [(medium_workplaces_x[i]*medium_workplaces_y[i])
        for i in range(0, len(medium_workplaces_x))]))
        /np.sum(medium_workplaces_y))

        self.mean_workplaces_people_meet = ((np.sum(
        [(_workplaces_x[i]*_workplaces_y[i])
        for i in range(0, len(_workplaces_x))]))
        /np.sum(_workplaces_y))

        self.mean_medium_workplaces_people_meet = ((np.sum(
        [(medium_workplaces_x[i]*medium_workplaces_y[i])
        for i in range(0, len(medium_workplaces_x))]))
        /np.sum(medium_workplaces_y))

        """
        compute the mean of people meet in the LARGE workplaces
        """
        self.mean_large_workplaces_people_meet = ((np.sum(
        [(large_workplaces_x[i]*large_workplaces_y[i])
        for i in range(0, len(large_workplaces_x))])
        )/np.sum(large_workplaces_y))



        _workplacesUpdated_number = np.array([small_workplaces,
                                    np.sum(medium_workplaces_y),
                                    np.sum(large_workplaces_y)])
        _workplacesUpdated_category = np.array(
                        ["small: [0-10]","medium: [10-100]","large: [100+]"])

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
        _schools_x = []
        _schools_y = []
        small_schools= _schools[1,1]
        small_schools_x = []
        small_schools_y = []
        medium_schools_x = []
        medium_schools_y = []
        large_schools_x = []
        large_schools_y = []

        # print("nb student " +str(np.sum(_schools[:,1])))

        for i in range(0,_schools[:,1].size):
            _schools_x.append(_schools[i,2])
            _schools_y.append(_schools[i,1])

        for i in range(0,10):
            small_schools_x.append(_schools[i,2])
            small_schools_y.append(_schools[i,1])

        for i in range(10,50):
            medium_schools_x.append(_schools[i,2])
            medium_schools_y.append(_schools[i,1])

        for i in range(50,_schools[:,1].size):
            large_schools_x.append(_schools[i,2])
            large_schools_y.append(_schools[i,1])

        # print("###############################################################")
        # print("small: " + str(np.sum(small_schools_y)))
        # print("medium: " + str(np.sum(medium_schools_y)))
        # print("large: " + str(np.sum(large_schools_y)))
        # print("tot: " + str( np.sum(_schools[:,1])  ))
        # print("###############################################################")
        plot_pie_schools()
        self.mean_schools = np.sum(
                                _schools_y)/len(_schools_x)
        self.mean_small_schools = np.sum(
                                small_schools_y)/len(small_schools_x)
        self.mean_medium_schools = np.sum(
                                medium_schools_y)/len(medium_schools_x)
        self.mean_large_schools = np.sum(
                                large_schools_y)/len(large_schools_x)

        """
        compute the mean of people meet in the SMALL schools
        """
        self.mean_schools_people_meet = ((np.sum(
        [(_schools_x[i]*_schools_y[i])
        for i in range(0, len(_schools_x))]))
        /np.sum(_schools_y))

        """
        compute the mean of people meet in the SMALL schools
        """
        self.mean_small_schools_people_meet = ((np.sum(
        [(small_schools_x[i]*small_schools_y[i])
        for i in range(0, len(small_schools_x))]))
        /np.sum(small_schools_y))
        """
        compute the mean of people meet in the MEDIUM schools
        """
        self.mean_medium_schools_people_meet = ((np.sum(
        [(medium_schools_x[i]*medium_schools_y[i])
        for i in range(0, len(medium_schools_x))]))
        /np.sum(medium_schools_y))
        """
        compute the mean of people meet in the LARGE schools
        """
        self.mean_large_schools_people_meet = ((np.sum(
        [(large_schools_x[i]*large_schools_y[i])
        for i in range(0, len(large_schools_x))]))
        /np.sum(large_schools_y))



        _schoolsUpdated_number = np.array([np.sum(small_schools_y),
                                    np.sum(medium_schools_y),
                                    np.sum(large_schools_y)])
        _schoolsUpdated_category = np.array(
                        ["small: [0-100]","medium: [100-500]","large: [500+]"])

        self._schools_df = pd.DataFrame(
                    {'size_of_schools': _schools_x,
                    'number': _schools_y})
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
        self.data_preprocessing_communities()
        self.plot_stats()


    def data_preprocessing_communities(self):
        _communities = self.communities_df.to_numpy()
        _communities_x = []
        _communities_y = []

        for i in range(0,_communities[:,1].size):
            _communities_x.append(_communities[i,0])
            _communities_y.append(_communities[i,2])



        # """
        # compute the mean of people meet in the SMALL communities
        # """
        mean_communities_people_meet = ((np.sum(
        [(_communities_x[i]*_communities_y[i])
        for i in range(0, len(_communities_x))]))
        /np.sum(_communities_y))

        print("mean_communities_people_meet: "+str(mean_communities_people_meet))



    def data_information(self):
        print("###############################################################")
        print("a student in a school meets: "
                + str(round(self.mean_schools_people_meet,2))
                +" persons")
        print("a worker in a workplaces meets: "
                + str(round(self.mean_workplaces_people_meet,2))
                +" persons")
        # print("###############################################################")
        print("###############################################################")
        # print("On average,\na person working in a SMALL sized company meets: "
        #         + str(round(self.mean_small_workplaces_people_meet,2))
        #         +" persons")
        # print("a person working in a MEDIUM sized company meets: "
        #         + str(round(self.mean_medium_workplaces_people_meet,2))
        #         +" persons")
        # print("a person working in a LARGE sized company meets: "
        #         + str(round(self.mean_large_workplaces_people_meet,2))
        #         +" persons")
        # print("###############################################################")
        # print("On average,\na student in a SMALL school meets: "
        #         + str(round(self.mean_small_schools_people_meet,2))
        #         +" persons")
        # print("a student in a MEDIUM school meets: "
        #         + str(round(self.mean_medium_schools_people_meet,2))
        #         +" persons")
        # print("a student in a LARGE school meets: "
        #         + str(round(self.mean_large_schools_people_meet,2))
        #         +" persons")
        # print("###############################################################")
