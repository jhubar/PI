
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools

def scenario_ld():
    # Create the model:
    model = SEIR()
    # Load the dataset
    model.import_dataset()
    i = 0
    time = np.arange(300)

    # --------------------------- ALL scenar --------------------------- #
    size = 5

    scenario_matrix = []
    for wm in range(0,size):
        scenario_wm_sd = []
        for sd in range(0,size):
            scenario_wm_sd_cs = []
            # for cs in range(0,size):
            for ld in range(0,size):

                scenario_wm_sd_cs.append({
                    'duration': 300,
                    'wearing_mask': [73, 73+(wm*30)],
                    'social_dist': [73, 73+(sd*30), 6],
                    'lock_down': [73, 73+(ld*30)]
                    # 'close_schools': [73, 73+(cs*30)],
                    # 'home_quarantine': [73, 119],
                    # 'case_isolation': [73, 193]



                })
            scenario_wm_sd.append(scenario_wm_sd_cs)
        scenario_matrix.append(scenario_wm_sd)
    # sys.exit()
    print(len(scenario_wm_sd))
    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                print(scenario_matrix[i][j][k])
            print("-----------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------")
        print("#############################################################################")
        print("#############################################################################")

    # --------------------------- Create models --------------------------- #
    mean_scenar = []

    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                model.set_scenario(scenario_matrix[i][j][k])
                mean_scenar.append(np.mean(model.stochastic_predic(duration=300, parameters=None,nb_simul=200, scenar=True),axis = 2))

    for i in range(0,size*size*size):
            data_to_export = pd.DataFrame(dict(Date = time,
                                        S = mean_scenar[i][:,0],
                                        E = mean_scenar[i][:,1],
                                        I = mean_scenar[i][:,2],
                                        R = mean_scenar[i][:,3],
                                        H = mean_scenar[i][:,4],
                                        C = mean_scenar[i][:,5],
                                        D = mean_scenar[i][:,6],
                                        index = i
                                        )
                                )
            # data_to_export.to_csv(r'Data_Scenario/scenario_'+str(i)+'.csv', header=True, index=False)
            data_to_export.to_csv(r'data/scenario_ld_'+str(i)+'.csv', header=True, index=False)
def scenario_ci():
    # Create the model:
    model = SEIR()
    # Load the dataset
    model.import_dataset()
    i = 0
    time = np.arange(300)

    # --------------------------- ALL scenar --------------------------- #
    size = 5

    scenario_matrix = []
    for wm in range(0,size):
        scenario_wm_sd = []
        for sd in range(0,size):
            scenario_wm_sd_cs = []
            for cs in range(0,size):
            # for ld in range(0,size):

                scenario_wm_sd_cs.append({
                    'duration': 300,
                    'wearing_mask': [73, 73+(wm*30)],
                    'social_dist': [73, 73+(sd*30), 6],
                    'close_schools': [73, 73+(cs*30)],
                    # 'home_quarantine': [73, 119],
                    'case_isolation': [73, 193]



                })
            scenario_wm_sd.append(scenario_wm_sd_cs)
        scenario_matrix.append(scenario_wm_sd)
    # sys.exit()
    print(len(scenario_wm_sd))
    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                print(scenario_matrix[i][j][k])
            print("-----------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------")
        print("#############################################################################")
        print("#############################################################################")

    # --------------------------- Create models --------------------------- #
    mean_scenar = []

    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                model.set_scenario(scenario_matrix[i][j][k])
                mean_scenar.append(np.mean(model.stochastic_predic(duration=300, parameters=None,nb_simul=200, scenar=True),axis = 2))

    for i in range(0,size*size*size):
            data_to_export = pd.DataFrame(dict(Date = time,
                                        S = mean_scenar[i][:,0],
                                        E = mean_scenar[i][:,1],
                                        I = mean_scenar[i][:,2],
                                        R = mean_scenar[i][:,3],
                                        H = mean_scenar[i][:,4],
                                        C = mean_scenar[i][:,5],
                                        D = mean_scenar[i][:,6],
                                        index = i
                                        )
                                )
            # data_to_export.to_csv(r'Data_Scenario/scenario_'+str(i)+'.csv', header=True, index=False)
            data_to_export.to_csv(r'data/scenario_ci_'+str(i)+'.csv', header=True, index=False)
def scenario_hm():
    # Create the model:
    model = SEIR()
    # Load the dataset
    model.import_dataset()
    i = 0
    time = np.arange(300)

    # --------------------------- ALL scenar --------------------------- #
    size = 5

    scenario_matrix = []
    for wm in range(0,size):
        scenario_wm_sd = []
        for sd in range(0,size):
            scenario_wm_sd_cs = []
            for cs in range(0,size):
            # for ld in range(0,size):

                scenario_wm_sd_cs.append({
                    'duration': 300,
                    'wearing_mask': [73, 73+(wm*30)],
                    'social_dist': [73, 73+(sd*30), 6],
                    'close_schools': [73, 73+(cs*30)],
                    'home_quarantine': [73, 193]




                })
            scenario_wm_sd.append(scenario_wm_sd_cs)
        scenario_matrix.append(scenario_wm_sd)
    # sys.exit()
    print(len(scenario_wm_sd))
    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                print(scenario_matrix[i][j][k])
            print("-----------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------")
        print("#############################################################################")
        print("#############################################################################")

    # --------------------------- Create models --------------------------- #
    mean_scenar = []

    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                model.set_scenario(scenario_matrix[i][j][k])
                mean_scenar.append(np.mean(model.stochastic_predic(duration=300, parameters=None,nb_simul=200, scenar=True),axis = 2))

    for i in range(0,size*size*size):
            data_to_export = pd.DataFrame(dict(Date = time,
                                        S = mean_scenar[i][:,0],
                                        E = mean_scenar[i][:,1],
                                        I = mean_scenar[i][:,2],
                                        R = mean_scenar[i][:,3],
                                        H = mean_scenar[i][:,4],
                                        C = mean_scenar[i][:,5],
                                        D = mean_scenar[i][:,6],
                                        index = i
                                        )
                                )
            # data_to_export.to_csv(r'Data_Scenario/scenario_'+str(i)+'.csv', header=True, index=False)
            data_to_export.to_csv(r'data/scenario_hm_'+str(i)+'.csv', header=True, index=False)
def scenario_hm_ci():
    # Create the model:
    model = SEIR()
    # Load the dataset
    model.import_dataset()
    i = 0
    time = np.arange(300)

    # --------------------------- ALL scenar --------------------------- #
    size = 5

    scenario_matrix = []
    for wm in range(0,size):
        scenario_wm_sd = []
        for sd in range(0,size):
            scenario_wm_sd_cs = []
            for cs in range(0,size):
            # for ld in range(0,size):

                scenario_wm_sd_cs.append({
                    'duration': 300,
                    'wearing_mask': [73, 73+(wm*30)],
                    'social_dist': [73, 73+(sd*30), 6],
                    'close_schools': [73, 73+(cs*30)],
                    'home_quarantine': [73, 193],
                    'case_isolation': [73, 193]



                })
            scenario_wm_sd.append(scenario_wm_sd_cs)
        scenario_matrix.append(scenario_wm_sd)
    # sys.exit()
    print(len(scenario_wm_sd))
    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                print(scenario_matrix[i][j][k])
            print("-----------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------")
        print("#############################################################################")
        print("#############################################################################")

    # --------------------------- Create models --------------------------- #
    mean_scenar = []

    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                model.set_scenario(scenario_matrix[i][j][k])
                mean_scenar.append(np.mean(model.stochastic_predic(duration=300, parameters=None,nb_simul=200, scenar=True),axis = 2))

    for i in range(0,size*size*size):
            data_to_export = pd.DataFrame(dict(Date = time,
                                        S = mean_scenar[i][:,0],
                                        E = mean_scenar[i][:,1],
                                        I = mean_scenar[i][:,2],
                                        R = mean_scenar[i][:,3],
                                        H = mean_scenar[i][:,4],
                                        C = mean_scenar[i][:,5],
                                        D = mean_scenar[i][:,6],
                                        index = i
                                        )
                                )
            # data_to_export.to_csv(r'Data_Scenario/scenario_'+str(i)+'.csv', header=True, index=False)
            data_to_export.to_csv(r'data/scenario_hm_ci_'+str(i)+'.csv', header=True, index=False)
def scenario():
    # Create the model:
    model = SEIR()
    # Load the dataset
    model.import_dataset()
    i = 0
    time = np.arange(300)

    # --------------------------- ALL scenar --------------------------- #
    size = 5

    scenario_matrix = []
    for wm in range(0,size):
        scenario_wm_sd = []
        for sd in range(0,size):
            scenario_wm_sd_cs = []
            for cs in range(0,size):
            # for ld in range(0,size):

                scenario_wm_sd_cs.append({
                    'duration': 300,
                    'wearing_mask': [73, 73+(wm*30)],
                    'social_dist': [73, 73+(sd*30), 6],
                    'close_schools': [73, 73+(cs*30)]
                    # 'home_quarantine': [73, 119],
                    # 'case_isolation': [73, 193]



                })
            scenario_wm_sd.append(scenario_wm_sd_cs)
        scenario_matrix.append(scenario_wm_sd)
    # sys.exit()
    print(len(scenario_wm_sd))
    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                print(scenario_matrix[i][j][k])
            print("-----------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------")
        print("#############################################################################")
        print("#############################################################################")

    # --------------------------- Create models --------------------------- #
    mean_scenar = []

    for i in range(0,size):
        for j in range(0,size):
            for k in range(0,size):
                model.set_scenario(scenario_matrix[i][j][k])
                mean_scenar.append(np.mean(model.stochastic_predic(duration=300, parameters=None,nb_simul=200, scenar=True),axis = 2))

    for i in range(0,size*size*size):
            data_to_export = pd.DataFrame(dict(Date = time,
                                        S = mean_scenar[i][:,0],
                                        E = mean_scenar[i][:,1],
                                        I = mean_scenar[i][:,2],
                                        R = mean_scenar[i][:,3],
                                        H = mean_scenar[i][:,4],
                                        C = mean_scenar[i][:,5],
                                        D = mean_scenar[i][:,6],
                                        index = i
                                        )
                                )
            # data_to_export.to_csv(r'Data_Scenario/scenario_'+str(i)+'.csv', header=True, index=False)
            data_to_export.to_csv(r'data/scenario_'+str(i)+'.csv', header=True, index=False)



if __name__ == "__main__":

    scenario()
