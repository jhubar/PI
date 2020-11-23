import matplotlib.pyplot as plt

def plot_dataset(self, filename, type):
    '''
    Parameters
    ----------
    filename: String
        Name of the pdf file to save the plot in

    type : String
        Type of dataset to plot.
        Format: "--typeofdataset1-curve1 --typeofdataset2-curve2"

                Type of datasets: - ds = datasets

                Type of curves: = - num_pos = num_positive
                                  - num_tested = num_tested
                                  - num_hospit = num_hospitalization
                                  - num_crit = num_critical
                                  - num_cum_hospit =
                                    num_cumulative_hospitalization
                                  - num_fatal = num_fatalities
                                  - cum-num-pos = cumul_num_positive

    Exemple
    -------
    plot_dataset("--sds-num_pos --df-num_pos"), to see the smoothing on
    num_positive

    Returns
    -------
    None.
    '''

    if duration == "dataset":
        time = np.arange(1, self.dataset.shape[0]+1, 1)
    else:
        time = np.arange(1, duration+1, 1)

    fig_plot_dataset = plt.figure("plt_dataset", figsize=(25,20))
    ax_plot_dataset = plt.subplot()

    if "--ds-num_pos" in type:
        ax_plot_dataset.plot(time,
                             self.dataset[:,1],
                             label='ds-num_pos')
    if "--ds-num_tested" in type:
        ax_plot_dataset.plot(time,
                             self.dataset[:,2],
                             label='ds-num_tested')
    if "--ds-num_hospit" in type:
        ax_plot_dataset.plot(time,
                             self.dataset[:,3],
                             label='ds-num_hospit')
    if "--ds-num_crit" in type:
        ax_plot_dataset.plot(time,
                             self.dataset[:,4],
                             label='ds-num_crit')
    if "--ds-num_cum_hospit" in type:
        ax_plot_dataset.plot(time,
                             self.dataset[:,5],
                             label='ds-num_cum_hospit')
    if "--ds-num_fatal" in type:
        ax_plot_dataset.plot(time,
                             self.dataset[:,6],
                             label='ds-num_fatal')
    if "--ds-cum_num_pos" in type:
        ax_plot_dataset.plot(time,
                             self.dataset[:,7],
                             label='ds-cum_num_pos')

    plt.figure("plt_dataset")
    plt.xlabel('Days', fontsize=30)
    plt.legend(fontsize=30)
    plt.title('Plot of the dataset for smoothed={}'.format(self.smoothing), fontsize=30)
    for tick in ax_plot_dataset.xaxis.get_major_ticks():
        	tick.label.set_fontsize(30)
    for tick in ax_plot_dataset.yaxis.get_major_ticks():
        	tick.label.set_fontsize(30)
    fig_plot_dataset.savefig(filename)

    plt.close()
