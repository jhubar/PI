import matplotlib.pyplot as plt
import numpy as np

def plot_dataset(self, filename, type, duration_=0):
    '''
    Parameters
    ----------
    filename: String
        Name of the pdf file to save the plot in

    type : String
        Type of dataset to plot.
        Format: "--typeofdataset1-curve1 --typeofdataset2-curve2"

                Type of datasets: - ds = datasets
                                  - det = deterministic prediction

                Type of curves: = (ds)
                                  - num_pos = num_positive
                                  - num_tested = num_tested
                                  - num_hospit = num_hospitalization
                                  - num_crit = num_critical
                                  - num_cum_hospit =
                                    num_cumulative_hospitalization
                                  - num_fatal = num_fatalities
                                  - cum_num_pos = cumul_num_positive
                                  (det, sto)
                                  - S = Succeptible
                                  - E = Exposed
                                  - I = Infected
                                  - R = Recovered
                                  - H = Hospitalized
                                  - C = Criticals
                                  - D = Deaths
                                  - +CC = Cumulative contaminations
                                  - +CH = Cumulative hospitalization

    Exemple
    -------
    plot_dataset("--det-I --df-num_pos"), to see the smoothing on
    num_positive

    Returns
    -------
    None.
    '''


    fig_plot_dataset = plt.figure("plt_dataset", figsize=(25,20))
    ax_plot_dataset = plt.subplot()

    if "--ds-num_pos" in type:
        ax_plot_dataset.plot(self.dataset[:,0],
                             self.dataset[:,1],
                             label='ds-num_pos')
    if "--ds-num_tested" in type:
        ax_plot_dataset.plot(self.dataset[:,0],
                             self.dataset[:,2],
                             label='ds-num_tested')
    if "--ds-num_hospit" in type:
        ax_plot_dataset.plot(self.dataset[:,0],
                             self.dataset[:,3],
                             label='ds-num_hospit')
    if "--ds-num_crit" in type:
        ax_plot_dataset.plot(self.dataset[:,0],
                             self.dataset[:,4],
                             label='ds-num_crit')
    if "--ds-num_cum_hospit" in type:
        ax_plot_dataset.plot(self.dataset[:,0],
                             self.dataset[:,5],
                             label='ds-num_cum_hospit')
    if "--ds-num_fatal" in type:
        ax_plot_dataset.plot(self.dataset[:,0],
                             self.dataset[:,6],
                             label='ds-num_fatal')
    if "--ds-cum_num_pos" in type:
        ax_plot_dataset.plot(self.dataset[:,0],
                             self.dataset[:,7],
                             label='ds-cum_num_pos')

    duration=self.dataset.shape[0]
    if duration_ != 0:
        duration = duration_


    time = np.arange(1, duration+1, 1)
    det = self.predict(duration)
    sto, _ = self.stochastic_mean(time, 1000)

    if "--det-S" in type:
        ax_plot_dataset.plot(time,
                             det[:,0],
                             label='deterministic - S')
    if "--det-E" in type:
        ax_plot_dataset.plot(time,
                             det[:,1],
                             label='deterministic - E')
    if "--det-I" in type:
        ax_plot_dataset.plot(time,
                             det[:,2],
                             label='deterministic - I')
    if "--det-R" in type:
        ax_plot_dataset.plot(time,
                            det[:,3],
                            label='deterministic - R')
    if "--det-H" in type:
        ax_plot_dataset.plot(time,
                             det[:,4],
                             label='deterministic - H')
    if "--det-C" in type:
        ax_plot_dataset.plot(time,
                             det[:,5],
                             label='deterministic - C')
    if "--det-D" in type:
        ax_plot_dataset.plot(time,
                             det[:,6],
                             label='deterministic - D')
    if "--det-+CC" in type:
        ax_plot_dataset.plot(time,
                             det[:,7],
                             label='deterministic - Cumul. Contam.')
    if "--det-+CH" in type:
        ax_plot_dataset.plot(time,
                             det[:,8],
                             label='deterministic - Cumul. Hospit')

    if "--sto-S" in type:
        ax_plot_dataset.plot(time,
                             sto[:,0],
                             label='stochastic - S')
    if "--sto-E" in type:
        ax_plot_dataset.plot(time,
                             sto[:,1],
                             label='stochastic - E')
    if "--sto-I" in type:
        ax_plot_dataset.plot(time,
                             sto[:,2],
                             label='stochastic - I')
    if "--sto-R" in type:
        ax_plot_dataset.plot(time,
                             sto[:,3],
                             label='stochastic - R')
    if "--sto-H" in type:
        ax_plot_dataset.plot(time,
                             sto[:,4],
                             label='stochastic - H')
    if "--sto-C" in type:
        ax_plot_dataset.plot(time,
                             sto[:,5],
                             label='stochastic - C')
    if "--sto-D" in type:
        ax_plot_dataset.plot(time,
                             sto[:,6],
                             label='stochastic - D')
    if "--sto-+CC" in type:
        ax_plot_dataset.plot(time,
                             sto[:,7],
                             label='stochastic - Cumul. Contam.')
    if "--sto-+CH" in type:
        ax_plot_dataset.plot(time,
                             sto[:,8],
                             label='stochastic - Cumul. Hospit')

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
