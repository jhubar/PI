import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SEIR import SEIR
import os



def sequential_analysis():
    # List of file in the result folder
    files_lst = os.listdir('result/')
    # Import header:
    hd = 'score;beta_final;sigma_final;gamma_final;hp_final;hcr_final;pc_final;pd_final;pcr_final;s_final;' \
             't_final;beta_init;sigma_init;gamma_init;hp_init;hcr_init;pc_init;pd_init;pcr_init;s_init;t_init;' \
             'w1;w2;w3;w4;w5;vw1;vw2;vw3;vw4;vw5;smoothing;optimizer;step_sizee;I_0'
    hd_lst = hd.split(";")
    # Store in a list of dataframe:
    df_lst = []
    # Import each file in the dataframe:
    for filename in files_lst:
        df = pd.read_csv('result/{}'.format(filename), header=0, names=hd_lst, sep=';')
        df_lst.append(df)
    # Concat all dataframes:
    result = pd.concat(df_lst, axis=0, ignore_index=True)

    # Sort:
    result.sort_values(by=['score'], inplace=True, ignore_index=True, ascending=True)
    print(result)

    # Save All the table in a csv in order to analyze them in a tabar
    result.to_csv('Total_result.csv', sep=';', header=True, index=True)

    npr = result.to_numpy()

    for i in range(0, npr.shape[0]):

        # Create a model:
        model = SEIR()
        model.fit_display = True
        model.basis_obj_display = False
        model.full_obj_display = False
        # Load parameters:
        model.beta = npr[i][1]
        model.sigma = npr[i][2]
        model.gamma = npr[i][3]
        model.hp = npr[i][4]
        model.hcr = npr[i][5]
        model.pc = npr[i][6]
        model.pd = npr[i][7]
        model.pcr = npr[i][8]
        model.s = npr[i][9]
        model.t = npr[i][10]

        # Import dataset:
        model.import_dataset()

        # Fit the model:
        model.fit()

        # Make predictions:
        predictions = model.predict(duration=model.dataset.shape[0])

        # Uncumul contaminations data
        uncumul = []
        uncumul.append(predictions[0][7])
        for j in range(1, predictions.shape[0]):
            uncumul.append(predictions[j][7] - predictions[j - 1][7])

        # Plot:
        time = model.dataset[:, 0]
        # Adapt test + with sensit and testing rate
        for j in range(0, len(time)):
            uncumul[j] = uncumul[j] * model.s * model.t

        # Plot cumul positive
        plt.scatter(time, model.dataset[:, 1], c='blue', label='test+')
        plt.plot(time, uncumul, c='blue', label='test+')
        # Plot hospit
        plt.scatter(time, model.dataset[:, 3], c='red', label='hospit data')
        plt.plot(time, predictions[:, 4], c='red', label='hospit pred')
        plt.legend()
        plt.title('Index {}'.format(i))
        plt.show()

        plt.scatter(time, model.dataset[:, 5], c='green', label='Critical data')
        plt.plot(time, predictions[:, 5], c='green', label='Critical predictions')
        plt.scatter(time, model.dataset[:, 6], c='black', label='Fatalities data')
        plt.plot(time, predictions[:, 6], c='black', label='Fatalities predictions')
        plt.legend()
        plt.title('index {}'.format(i))
        plt.show()

        print('---------------------------------------------------------')

        row = result.loc[i, :]

        print(row)

        print("<Press enter/return to continue>")
        input()

def result_concatener(options=''):
    """
    This method read each bruteforce result file, concatenate each in one dataframe, sort result by ascending score and
    export everything in global_result.csv
    Options:
        - 'boundaries' : delete each row who converge against boundaries
    """
    # List of file in the result folder
    files_lst = os.listdir('result/')
    # Import header:
    hd = 'score;beta_final;sigma_final;gamma_final;hp_final;hcr_final;pc_final;pd_final;pcr_final;s_final;' \
             't_final;beta_init;sigma_init;gamma_init;hp_init;hcr_init;pc_init;pd_init;pcr_init;s_init;t_init;' \
             'w1;w2;w3;w4;w5;vw1;vw2;vw3;vw4;vw5;smoothing;optimizer;step_sizee;I_0'
    hd_lst = hd.split(";")
    # Store in a list of dataframe:
    df_lst = []
    # Import each file in the dataframe:
    for filename in files_lst:
        df = pd.read_csv('result/{}'.format(filename), header=0, names=hd_lst, sep=';')
        df_lst.append(df)
    # Concat all dataframes:
    result = pd.concat(df_lst, axis=0, ignore_index=True)

    # Sort:
    result.sort_values(by=['score'], inplace=True, ignore_index=True, ascending=True)

    # Save All the table in a csv in order to analyze them in a tabar
    result.to_csv('result_analysis\global_result.csv', sep=';', header=True, index=True)

    npr = result.to_numpy()

    print('Number of total entries = {}'.format(npr.shape[0]))

    if 'boundaries' in options:

        # open the file to write:
        file = open('result_analysis/global_no_boundaries.csv', 'w')
        # Write headers
        file.write(';')
        file.write(hd)
        file.write('\n')

        # Boundaries:
        sigma_min = 1 / 5
        sigma_max = 1
        gamma_min = 1 / 10
        gamma_max = 1 / 4
        s_min = 0.7
        s_max = 0.85
        t_min = 0.5
        t_max = 1

        idx = 0
        for i in range(0, npr.shape[0]):

            select = True
            # Boundaries checking
            if np.fabs(npr[i][2] - sigma_min) < 0.01:
                select = False
            if np.fabs(npr[i][2] - sigma_max) < 0.01:
                select = False
            if np.fabs(npr[i][3] - gamma_min) < 0.01:
                select = False
            if np.fabs(npr[i][3] - gamma_max) < 0.01:
                select = False
            if np.fabs(npr[i][9] - s_min) < 0.01:
                select = False
            if np.fabs(npr[i][9] - s_max) < 0.01:
                select = False
            if np.fabs(npr[i][10] - t_min) < 0.01:
                select = False
            if np.fabs(npr[i][10] - t_max) < 0.01:
                select = False

            if select:
                string = [str(idx)]
                for j in range(0, npr.shape[1]):
                    string.append(str(npr[i][j]))
                # Write in the file
                file.write(';'.join(string))
                file.write('\n')

                idx += 1
        file.close()
        print('number of no_boundaries entries = {}'.format(idx))

def result_reader(file_name='result_analysis/global_no_boundaries.csv'):

    # Import file in a dataframe:
    result = pd.read_csv(file_name, header=0, sep=';', index_col=0)
    # Numpy version
    npr = result.to_numpy()

    # Create a model:
    model = SEIR()
    # Import the dataset
    model.import_dataset()

    while True:

        print("===========================Enter the index of the row to analyse=================================")
        i = int(input())

        # Print the selected row
        row = result.loc[i, :]
        print(row)

        # Load parameters:
        model.beta = npr[i][1]
        model.sigma = npr[i][2]
        model.gamma = npr[i][3]
        model.hp = npr[i][4]
        model.hcr = npr[i][5]
        model.pc = npr[i][6]
        model.pd = npr[i][7]
        model.pcr = npr[i][8]
        model.s = npr[i][9]
        model.t = npr[i][10]

        # Make predictions
        predictions = model.predict(duration=model.dataset.shape[0])

        # Uncumul contaminations data
        uncumul = []
        uncumul.append(predictions[0][7])
        for j in range(1, predictions.shape[0]):
            uncumul.append(predictions[j][7] - predictions[j - 1][7])

        # Plot:
        time = model.dataset[:, 0]
        # Adapt test + with sensit and testing rate
        for j in range(0, len(time)):
            uncumul[j] = uncumul[j] * model.s * model.t

        # Plot cumul positive
        plt.scatter(time, model.dataset[:, 1], c='blue', label='test+')
        plt.plot(time, uncumul, c='blue', label='test+')
        # Plot hospit
        plt.scatter(time, model.dataset[:, 3], c='red', label='hospit data')
        plt.plot(time, predictions[:, 4], c='red', label='hospit pred')
        plt.legend()
        plt.title('Index {}'.format(i))
        plt.show()

        plt.scatter(time, model.dataset[:, 5], c='green', label='Critical data')
        plt.plot(time, predictions[:, 5], c='green', label='Critical predictions')
        plt.scatter(time, model.dataset[:, 6], c='black', label='Fatalities data')
        plt.plot(time, predictions[:, 6], c='black', label='Fatalities predictions')
        plt.legend()
        plt.title('index {}'.format(i))
        plt.show()

def rel_var_weights_score():

    """    # Import file in a dataframe:
    file_name = 'result_analysis/global_no_boundaries.csv'
    result = pd.read_csv(file_name, header=0, sep=';', index_col=0)
    # Numpy version
    npr = result.to_numpy()
    marker_size = 2
    plt.scatter(result['score'], result['vw1'], c='blue', label='vw1', s=marker_size)
    plt.scatter(result['score'], result['vw2'], c='red', label='vw2', s=marker_size)
    plt.scatter(result['score'], result['vw3'], c='black', label='vw3', s=marker_size)
    plt.scatter(result['score'], result['vw4'], c='yellow', label='vw4', s=marker_size)
    plt.scatter(result['score'], result['vw5'], c='green', label='vw5', s=marker_size)
    plt.legend()
    plt.title('relation between score and variance weights in no_boudaries'.format(file_name))
    plt.savefig('result_analysis/figures/var_weigts_no_bound.pdf')
    plt.show()
    """
    file_name = 'result_analysis/global_result.csv'
    result = pd.read_csv(file_name, header=0, sep=';', index_col=0)
    # Numpy version
    npr = result.to_numpy()

    # Analyse of the probability to reach boundary and var weights:
    bound_reach = []
    # Boundaries:
    sigma_min = 1 / 5
    sigma_max = 1
    gamma_min = 1 / 10
    gamma_max = 1 / 4
    s_min = 0.7
    s_max = 0.85
    t_min = 0.5
    t_max = 1
    for i in range(0, npr.shape[0]):

        select = True
        # Boundaries checking
        if np.fabs(npr[i][2] - sigma_min) < 0.01:
            select = False
        if np.fabs(npr[i][2] - sigma_max) < 0.01:
            select = False
        if np.fabs(npr[i][3] - gamma_min) < 0.01:
            select = False
        if np.fabs(npr[i][3] - gamma_max) < 0.01:
            select = False
        if np.fabs(npr[i][9] - s_min) < 0.01:
            select = False
        if np.fabs(npr[i][9] - s_max) < 0.01:
            select = False
        if np.fabs(npr[i][10] - t_min) < 0.01:
            select = False
        if np.fabs(npr[i][10] - t_max) < 0.01:
            select = False

        bound_reach.append(select)

    # Add the new column in the dataset
    result.insert(npr.shape[1], 'bd_reach', bound_reach)
    # Convert boolean to color:
    bnd_colors = []
    for i in range(0, len(bound_reach)):
        if bound_reach[i]:
            bnd_colors.append('blue')
        else:
            bnd_colors.append('red')

    # Plot correlations plot:
    size = 10
    corr = result.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()






if __name__ == "__main__":

    # Concatène tous les dataset en un fichier Global_result
    # Et un fichier global_no_boundaries ne contenant que les vombinaisons ne touchant pas de bornes
    result_concatener(options='boundaries')

    # Lit un fichier au choix, puis permet de choisir l'index de la ligue que l'on veut ploter
    result_reader(file_name='result_analysis/global_no_boundaries.csv')

    #rel_var_weights_score()