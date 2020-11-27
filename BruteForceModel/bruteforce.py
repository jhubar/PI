from SEIR import SEIR
import numpy as np

def brute_force_fitting(name='Francois', id='1'):
    """

    :param name:
    :param id:
    :return:
    Header:

    """
    # Create a model:
    mdl = SEIR()
    # Import the dataset
    mdl.import_dataset()

    # Shut down displays
    mdl.fit_display = False
    mdl.basis_obj_display = False
    mdl.full_obj_display = False

    # Infinite loop:
    iter = 0
    while True:
        iter += 1
        # Set model parameters:
        mdl.beta = np.random.uniform(mdl.beta_min, mdl.beta_max)
        mdl.sigma = np.random.uniform(mdl.sigma_min, mdl.sigma_max)
        mdl.gamma = np.random.uniform(mdl.gamma_min, mdl.gamma_max)
        mdl.hp = np.random.uniform(mdl.hp_min, mdl.hp_max)
        mdl.hcr = np.random.uniform(mdl.hcr_min, mdl.hcr_max)
        mdl.pc = np.random.uniform(mdl.pc_min, mdl.pc_max)
        mdl.pd = np.random.uniform(mdl.pd_min, mdl.pd_max)
        mdl.pcr = np.random.uniform(mdl.pcr_min, mdl.pcr_max)
        mdl.s = np.random.uniform(mdl.s_min, mdl.s_max)
        mdl.t = np.random.uniform(mdl.t_min, mdl.t_max)

        # Store initial parameters value:
        init_params = mdl.get_parameters()
        # Store hyperparameters:
        init_hparams = mdl.get_hyperparameters()

        print('================================================')
        print('Iteration {}'.format(iter))
        print('Epidemic starting parameters: ')
        print(init_params)

        # Fit the model
        score = mdl.fit(method='bruteforce')
        print('Score = {}'.format(score))
        final_param = mdl.get_parameters()
        print('Final parameters: ')
        print(final_param)

        # Make the string and store in file
        pre_string = [str(score)]
        for item in final_param:
            pre_string.append(str(item))
        for item in init_params:
            pre_string.append(str(item))
        for item in init_hparams:
            pre_string.append(str(item))

        string = ';'.join(pre_string)

        # Open and write in the file
        file = open('result/{}-{}.csv'.format(name, id), 'a')
        file.write(string)
        file.write('\n')
        file.close()

