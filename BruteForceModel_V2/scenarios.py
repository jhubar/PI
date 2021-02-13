
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from SEIR import SEIR
import tools




def scenario():
    # Create the model:
    model = SEIR()

    # Load the dataset
    model.import_dataset()

    # Load best models file:
    result = pd.read_csv('FINAL_MODEL.csv', header=0, sep=';', index_col=0)
    # Numpy version
    npr = result.to_numpy()
    # Chose index 0:
    i = 0
    # Print the selected row
    # row = result.loc[i, :]
    # print(row)
    # Load parameters
    model.beta = npr[i][0]
    model.sigma = npr[i][1]
    model.gamma = npr[i][2]
    model.hp = npr[i][3]
    model.hcr = npr[i][4]
    model.pc = npr[i][5]
    model.pd = npr[i][6]
    model.pcr = npr[i][7]
    model.s = npr[i][8]
    model.t = npr[i][9]
    model.I_0 = npr[i][23]

    # Load parameters
    model.beta = npr[i][0]
    model.sigma = npr[i][1]
    model.gamma = npr[i][2]
    model.hp = npr[i][3]
    model.hcr = npr[i][4]
    model.pc = npr[i][5]
    model.pd = npr[i][6]
    model.pcr = npr[i][7]
    model.s = npr[i][8]
    model.t = npr[i][9]
    model.I_0 = npr[i][23]
    # Get time vector:
    time = np.arange(300)





    # --------------------------- Build scenarios --------------------------- #

    # Choose scenario


    scenario_1 = {
        'duration': 300,
        'social_dist': [73, 119, 6],
        'wearing_mask': [73, 119]
    }
    title_1 = 'Social distancing + wearing mask'

    scenario_2 = {
        'duration': 300,
        'social_dist': [73, 119, 6],
        'wearing_mask': [73, 119],
        'close_schools': [73, 119]
    }
    title_2 = 'Social + Mask + School'
    scenario_3 = {
        'duration': 300,
        'social_dist': [73, 119, 6],
        'home_quarantine': [73, 119],
        'wearing_mask': [73, 119]
    }
    title_3 = 'Mask + Social + quarantine'
    scenario_4 = {
        'duration': 300,
        'social_dist': [73, 119, 6],
        'case_isolation': [73, 119],
        'wearing_mask': [73, 119]
    }
    title_4 = 'Mask + Social + Case isolation'
    scenario_5 = {
        'duration': 300,
        'social_dist': [73, 119, 6],
        'home_quarantine': [73, 119],
        'wearing_mask': [73, 119],
        'close_schools': [73, 119]
    }
    title_5 = 'Mask + Quarantine + School'
    scenario_6 = {
        'duration': 300,
        'home_quarantine': [73, 119],
        'wearing_mask': [73, 119],
        'social_dist': [73, 119, 6],
        "lock_down": [73, 119]
    }
    title_6 = 'Lock down + measures'
    scenario_7 = {
        'duration': 300,
        'social_dist': [73, 119, 6],
        'wearing_mask': [73, 119],
        'home_quarantine': [73, 119]
    }
    title_7 = 'Mask + social + quarantine'
    # --------------------------- ALL scenar --------------------------- #
    size = 10

    scenario_matrix = []
    for wm in range(0,size):
        scenario_wm_sd = []
        for sd in range(0,size):


            scenario_wm_sd.append({
                'duration': 300,
                'wearing_mask': [73, 73+(wm*10)],
                'social_dist': [73, 73+(sd*10), 6],

            })
        scenario_matrix.append(scenario_wm_sd)

    # --------------------------- Create models --------------------------- #
    mean_scenar = []

    for i in range(0,size):
        for j in range(0,size):
            model.set_scenario(scenario_matrix[i][j])
            mean_scenar.append(np.mean(model.stochastic_predic(duration=300, parameters=None,nb_simul=200, scenar=True),axis = 2))

    for i in range(0,size):
            data_to_export = pd.DataFrame(dict(Date = time,
                                        S = mean_scenar[i][:,0],
                                        E = mean_scenar[i][:,1],
                                        I = mean_scenar[i][:,2],
                                        R = mean_scenar[i][:,3],
                                        H = mean_scenar[i][:,4],
                                        C = mean_scenar[i][:,5],
                                        D = mean_scenar[i][:,6]
                                        )
                                )
            data_to_export.to_csv(r'Data_Scenario/scenario_'+str(i)+'.csv', header=True, index=False)



    # model.set_scenario(scenario_1)
    # pred_scenar_1 = model.stochastic_predic(duration=300, parameters=None,
    #                                       nb_simul=200, scenar=True)
    # pred_normal_1 = model.stochastic_predic(duration=300, parameters=None,
    #                                       nb_simul=200, scenar=False)
    #
    # model.set_scenario(scenario_2)
    # pred_scenar_2 = model.stochastic_predic(duration=300, parameters=None,
    #                                         nb_simul=200, scenar=True)
    #
    # model.set_scenario(scenario_3)
    # pred_scenar_3 = model.stochastic_predic(duration=300, parameters=None,
    #                                         nb_simul=200, scenar=True)
    #
    # model.set_scenario(scenario_4)
    # pred_scenar_4 = model.stochastic_predic(duration=300, parameters=None,
    #                                         nb_simul=200, scenar=True)
    #
    # model.set_scenario(scenario_5)
    # pred_scenar_5 = model.stochastic_predic(duration=300, parameters=None,
    #                                         nb_simul=200, scenar=True)
    #
    # model.set_scenario(scenario_6)
    # pred_scenar_6 = model.stochastic_predic(duration=300, parameters=None,
    #                                         nb_simul=200, scenar=True)
    #
    # model.set_scenario(scenario_7)
    # pred_scenar_7 = model.stochastic_predic(duration=300, parameters=None,
    #                                         nb_simul=200, scenar=True)
    #
    # # Compute mean predictions
    # mean_scenar_1 = np.mean(pred_scenar_1, axis=2)
    # mean_normal_1 = np.mean(pred_normal_1, axis=2)
    # model.nb_simul = 200
    #
    # mean_scenar_2 = np.mean(pred_scenar_2, axis=2)
    #
    # mean_scenar_3 = np.mean(pred_scenar_3, axis=2)
    #
    # mean_scenar_4 = np.mean(pred_scenar_4, axis=2)
    #
    # mean_scenar_5 = np.mean(pred_scenar_5, axis=2)
    #
    # mean_scenar_6 = np.mean(pred_scenar_6, axis=2)
    #
    # mean_scenar_7 = np.mean(pred_scenar_7, axis=2)
    #
    # # --------------------------- Plot scenarios --------------------------- #
    # # Data storing:
    #
    # data_to_export = pd.DataFrame(dict(Date = time,
    #                                     S = mean_scenar_1[:,0],
    #                                     E = mean_scenar_1[:,1],
    #                                     I = mean_scenar_1[:,2],
    #                                     R = mean_scenar_1[:,3],
    #                                     H = mean_scenar_1[:,4],
    #                                     C = mean_scenar_1[:,5],
    #                                     D = mean_scenar_1[:,6]
    #                                     )
    #                             )
    # data_to_export.to_csv(r'scenario_1.csv', header=True, index=False)
    # # Data storing:
    #
    # data_to_export = pd.DataFrame(dict(Date = time,
    #                                     S = mean_scenar_2[:,0],
    #                                     E = mean_scenar_2[:,1],
    #                                     I = mean_scenar_2[:,2],
    #                                     R = mean_scenar_2[:,3],
    #                                     H = mean_scenar_2[:,4],
    #                                     C = mean_scenar_2[:,5],
    #                                     D = mean_scenar_2[:,6]
    #                                     )
    #                             )
    # data_to_export.to_csv(r'scenario_2.csv', header=True, index=False)
    #
    # data_to_export = pd.DataFrame(dict(Date = time,
    #                                     S = mean_scenar_3[:,0],
    #                                     E = mean_scenar_3[:,1],
    #                                     I = mean_scenar_3[:,2],
    #                                     R = mean_scenar_3[:,3],
    #                                     H = mean_scenar_3[:,4],
    #                                     C = mean_scenar_3[:,5],
    #                                     D = mean_scenar_3[:,6]
    #                                     )
    #                             )
    # data_to_export.to_csv(r'scenario_3.csv', header=True, index=False)
    # # Data storing:
    #
    # data_to_export = pd.DataFrame(dict(Date = time,
    #                                     S = mean_scenar_4[:,0],
    #                                     E = mean_scenar_4[:,1],
    #                                     I = mean_scenar_4[:,2],
    #                                     R = mean_scenar_4[:,3],
    #                                     H = mean_scenar_4[:,4],
    #                                     C = mean_scenar_4[:,5],
    #                                     D = mean_scenar_4[:,6]
    #                                     )
    #                             )
    # data_to_export.to_csv(r'scenario_4.csv', header=True, index=False)
    #
    # data_to_export = pd.DataFrame(dict(Date = time,
    #                                     S = mean_scenar_5[:,0],
    #                                     E = mean_scenar_5[:,1],
    #                                     I = mean_scenar_5[:,2],
    #                                     R = mean_scenar_5[:,3],
    #                                     H = mean_scenar_5[:,4],
    #                                     C = mean_scenar_5[:,5],
    #                                     D = mean_scenar_5[:,6]
    #                                     )
    #                             )
    # data_to_export.to_csv(r'scenario_5.csv', header=True, index=False)
    # # Data storing:
    #
    # data_to_export = pd.DataFrame(dict(Date = time,
    #                                     S = mean_scenar_6[:,0],
    #                                     E = mean_scenar_6[:,1],
    #                                     I = mean_scenar_6[:,2],
    #                                     R = mean_scenar_6[:,3],
    #                                     H = mean_scenar_6[:,4],
    #                                     C = mean_scenar_6[:,5],
    #                                     D = mean_scenar_6[:,6]
    #                                     )
    #                             )
    # data_to_export.to_csv(r'scenario_6.csv', header=True, index=False)
    #
    # data_to_export = pd.DataFrame(dict(Date = time,
    #                                     S = mean_scenar_7[:,0],
    #                                     E = mean_scenar_7[:,1],
    #                                     I = mean_scenar_7[:,2],
    #                                     R = mean_scenar_7[:,3],
    #                                     H = mean_scenar_7[:,4],
    #                                     C = mean_scenar_7[:,5],
    #                                     D = mean_scenar_7[:,6]
    #                                     )
    #                             )
    # data_to_export.to_csv(r'scenario_7.csv', header=True, index=False)

    # # Hospit
    # fig_H = plt.figure()
    # ax_H = plt.subplot()
    # ax_H.plot(time, mean_normal_1[:, 4], c='black', label='initial')
    # ax_H.plot(time, mean_scenar_1[:, 4], label=title_1)
    # ax_H.plot(time, mean_scenar_2[:, 4], label=title_2)
    # ax_H.plot(time, mean_scenar_3[:, 4], label=title_3)
    # ax_H.plot(time, mean_scenar_4[:, 4], label=title_4)
    # ax_H.plot(time, mean_scenar_5[:, 4], label=title_5)
    # ax_H.plot(time, mean_scenar_6[:, 4], label=title_6)
    # ax_H.plot(time, mean_scenar_7[:, 4], label=title_7)
    # plt.axhline(1500, label="max hospit.")
    # plt.legend()
    # plt.title("Hospitalization")
    # fig_H.savefig("simul_francois/Hospitalization_scenario_2.pdf")
    #
    # #Criticals
    # fig_C = plt.figure()
    # ax_C = plt.subplot()
    # ax_C.plot(time, mean_normal_1[:, 5], c='black', label='initial')
    # ax_C.plot(time, mean_scenar_1[:, 5], label=title_1)
    # ax_C.plot(time, mean_scenar_2[:, 5], label=title_2)
    # ax_C.plot(time, mean_scenar_3[:, 5], label=title_3)
    # ax_C.plot(time, mean_scenar_4[:, 5], label=title_4)
    # ax_C.plot(time, mean_scenar_5[:, 5], label=title_5)
    # ax_C.plot(time, mean_scenar_6[:, 5], label=title_6)
    # ax_C.plot(time, mean_scenar_7[:, 5], label=title_7)
    # plt.axhline(300, label='max criticals')
    # plt.legend()
    # plt.title("Criticals")
    # fig_C.savefig('simul_francois/Criticals_scenario_2.pdf')


if __name__ == "__main__":

    scenario()
