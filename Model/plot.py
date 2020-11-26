import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(self, filename, type, duration_=0,
                 plot_conf_inter=False, global_view=False,plot_param=False):
    """
	Parameters
	----------

	@filename: String
		Name of the pdf file to save the plot in

	@type: String
		Type of dataset to plot.
		Format: "--typeofdataset1-curve1 --typeofdataset2-curve2"

				Type of datasets: - ds = datasets
								  - det = deterministic prediction
								  - sto = mean stochastic prediction

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
								  - +CH = Cumulative hospitalization (only deterministic)
	@plot_conf_inter: Bool
		Print 67% confidence interval  for each  stochastic curve plotted if set to TRUE

	Exemple
	-------
	plot_dataset("--det-I --ds-num_pos")

	Returns
	-------
	None.
	"""

    fig_plot_dataset = plt.figure("plt_dataset", figsize=(25, 20))
    ax_plot_dataset = plt.subplot()

    # --------------------------------------------#
    #                Plot dataset
    # --------------------------------------------#

    if "--ds-num_pos" in type:
        ax_plot_dataset.plot(self.dataset[:, 0],
                             self.dataset[:, 1],
                             label='ds-num_pos')
    if "--ds-num_tested" in type:
        ax_plot_dataset.plot(self.dataset[:, 0],
                             self.dataset[:, 2],
                             label='ds-num_tested')
    if "--ds-num_hospit" in type:
        ax_plot_dataset.plot(self.dataset[:, 0],
                             self.dataset[:, 3],
                             label='ds-num_hospit')
    if "--ds-num_crit" in type:
        ax_plot_dataset.plot(self.dataset[:, 0],
                             self.dataset[:, 5],
                             label='ds-num_crit')
    if "--ds-num_cum_hospit" in type:
        ax_plot_dataset.plot(self.dataset[:, 0],
                             self.dataset[:, 4],
                             label='ds-num_cum_hospit')
    if "--ds-num_fatal" in type:
        ax_plot_dataset.plot(self.dataset[:, 0],
                             self.dataset[:, 6],
                             label='ds-num_fatal')
    if "--ds-cum_num_pos" in type:
        ax_plot_dataset.plot(self.dataset[:, 0],
                             self.dataset[:, 7],
                             label='ds-cum_num_pos')

    duration = self.dataset.shape[0]
    if duration_ != 0:
        duration = duration_

    time = np.arange(1, duration + 1, 1)
    det = self.predict(duration)
    sto_m, sto_hq, sto_lq, _,\
        res_S, res_E, res_I, res_R, res_H, res_C, res_F, res_Conta = \
            self.stochastic_mean(time, 1000)

    # --------------------------------------------#
    #              Plot deterministic
    # --------------------------------------------#

    if "--det-S" in type:
        ax_plot_dataset.plot(time,
                             det[:, 0],
                             label='deterministic - S')
    if "--det-E" in type:
        ax_plot_dataset.plot(time,
                             det[:, 1],
                             label='deterministic - E')
    if "--det-I" in type:
        ax_plot_dataset.plot(time,
                             det[:, 2],
                             label='deterministic - I')
    if "--det-R" in type:
        ax_plot_dataset.plot(time,
                             det[:, 3],
                             label='deterministic - R')
    if "--det-H" in type:
        ax_plot_dataset.plot(time,
                             det[:, 4],
                             label='deterministic - H')
    if "--det-C" in type:
        ax_plot_dataset.plot(time,
                             det[:, 5],
                             label='deterministic - C')
    if "--det-D" in type:
        ax_plot_dataset.plot(time,
                             det[:, 6],
                             label='deterministic - D')
    if "--det-+CC" in type:
        ax_plot_dataset.plot(time,
                             det[:, 7],
                             label='deterministic - Cumul. Contam.')
    if "--det-+CH" in type:
        ax_plot_dataset.plot(time,
                             det[:, 8],
                             label='deterministic - Cumul. Hospit')

    # --------------------------------------------#
    #               Plot all curves
    # --------------------------------------------#
    line_width = 0.3
    qt_of_plot = 10
    if global_view:
        if "--sto-S" in type:
            for i in range(res_S.shape[1]):
                if i%qt_of_plot == 0:
                    ax_plot_dataset.plot(time,
                                         res_S[:, i],
                                         linewidth=line_width,
                                         color='green')

        if "--sto-E" in type:
            for i in range(res_E.shape[1]):
                if i%qt_of_plot == 0:
                    ax_plot_dataset.plot(time,
                                         res_E[:, i],
                                         linewidth=line_width,
                                         color='orange')

        if "--sto-I" in type:
            for i in range(res_I.shape[1]):
                if i%qt_of_plot == 0:
                    ax_plot_dataset.plot(time,
                                         res_I[:, i],
                                         linewidth=line_width,
                                         color='indianred')

        if "--sto-R" in type:
            for i in range(res_R.shape[1]):
                if i%qt_of_plot == 0:
                    ax_plot_dataset.plot(time,
                                         res_R[:, i],
                                         linewidth=line_width,
                                         color='mediumpurple')

        if "--sto-H" in type:
            for i in range(res_H.shape[1]):
                if i%qt_of_plot == 0:
                    ax_plot_dataset.plot(time,
                                         res_H[:, i],
                                         linewidth=line_width,
                                         color='steelblue')

        if "--sto-C" in type:
            for i in range(res_C.shape[1]):
                if i%qt_of_plot == 0:
                    ax_plot_dataset.plot(time,
                                         res_C[:, i],
                                         linewidth=line_width,
                                         color='plum')

        if "--sto-D" in type:
            for i in range(res_F.shape[1]):
                if i%qt_of_plot == 0:
                    ax_plot_dataset.plot(time,
                                         res_F[:, i],
                                         linewidth=line_width,
                                         color='peru')

        if "--sto-+CC" in type:
            for i in range(res_Conta.shape[1]):
                if i%qt_of_plot == 0:
                    ax_plot_dataset.plot(time,
                                         res_Conta[:, i],
                                         linewidth=line_width,
                                         color='cornflowerblue')
    # --------------------------------------------#
    #            Plot stochastic hq & lq
    # --------------------------------------------#

    if plot_conf_inter:
        if "--sto-S" in type:
            ax_plot_dataset.plot(time,
                                 sto_hq[:, 0],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.plot(time,
                                 sto_lq[:, 0],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.fill_between(time,
                                         sto_hq[:, 0],
                                         sto_lq[:, 0],
                                         color='lavender',
                                         alpha=0.7)
        if "--sto-E" in type:
            ax_plot_dataset.plot(time,
                                 sto_hq[:, 1],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.plot(time,
                                 sto_lq[:, 1],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.fill_between(time,
                                         sto_hq[:, 1],
                                         sto_lq[:, 1],
                                         color='thistle',
                                         alpha=0.7)
        if "--sto-I" in type:
            ax_plot_dataset.plot(time,
                                 sto_hq[:, 2],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.plot(time,
                                 sto_lq[:, 2],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.fill_between(time,
                                         sto_hq[:, 2],
                                         sto_lq[:, 2],
                                         color='mistyrose',
                                         alpha=0.7)
        if "--sto-R" in type:
            ax_plot_dataset.plot(time,
                                 sto_hq[:, 3],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.plot(time,
                                 sto_lq[:, 3],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.fill_between(time,
                                         sto_hq[:, 3],
                                         sto_lq[:, 3],
                                         color='lavenderblush',
                                         alpha=0.7)
        if "--sto-H" in type:
            ax_plot_dataset.plot(time,
                                 sto_hq[:, 4],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.plot(time,
                                 sto_lq[:, 4],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.fill_between(time,
                                         sto_hq[:, 4],
                                         sto_lq[:, 4],
                                         color='palegreen',
                                         alpha=0.7)
        if "--sto-C" in type:
            ax_plot_dataset.plot(time,
                                 sto_hq[:, 5],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.plot(time,
                                 sto_lq[:, 5],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.fill_between(time,
                                         sto_hq[:, 5],
                                         sto_lq[:, 5],
                                         color='peachpuff',
                                         alpha=0.7)
        if "--sto-D" in type:
            ax_plot_dataset.plot(time,
                                 sto_hq[:, 6],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.plot(time,
                                 sto_lq[:, 6],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.fill_between(time,
                                         sto_hq[:, 6],
                                         sto_lq[:, 6],
                                         color='lightcyan',
                                         alpha=0.7)
        if "--sto-+CC" in type:
            ax_plot_dataset.plot(time,
                                 sto_hq[:, 7],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.plot(time,
                                 sto_lq[:, 7],
                                 color='grey',
                                 linewidth=0.3)
            ax_plot_dataset.fill_between(time,
                                         sto_hq[:, 7],
                                         sto_lq[:, 7],
                                         color='bisque',
                                         alpha=0.7)

    # --------------------------------------------#
    #            Plot stochastic mean
    # --------------------------------------------#
    if global_view:
        line_width_sto_m = 4
    else:
        line_width_sto_m = 2

    if "--sto-S" in type:
        ax_plot_dataset.plot(time,
                             sto_m[:, 0],
                             label='stochastic - S',
                             linewidth=line_width_sto_m)
    if "--sto-E" in type:
        ax_plot_dataset.plot(time,
                             sto_m[:, 1],
                             label='stochastic - E',
                             linewidth=line_width_sto_m)
    if "--sto-I" in type:
        ax_plot_dataset.plot(time,
                             sto_m[:, 2],
                             label='stochastic - I',
                             linewidth=line_width_sto_m)
    if "--sto-R" in type:
        ax_plot_dataset.plot(time,
                             sto_m[:, 3],
                             label='stochastic - R',
                             linewidth=line_width_sto_m)
    if "--sto-H" in type:
        ax_plot_dataset.plot(time,
                             sto_m[:, 4],
                             label='stochastic - H',
                             linewidth=line_width_sto_m)
    if "--sto-C" in type:
        ax_plot_dataset.plot(time,
                             sto_m[:, 5],
                             label='stochastic - C',
                             linewidth=line_width_sto_m)
    if "--sto-D" in type:
        ax_plot_dataset.plot(time,
                             sto_m[:, 6],
                             label='stochastic - D',
                             linewidth=line_width_sto_m)
    if "--sto-+CC" in type:
        ax_plot_dataset.plot(time,
                             sto_m[:, 7],
                             label='stochastic - Cumul. Contam.',
                             linewidth=line_width_sto_m)

    plt.figure("plt_dataset")
    plt.xlabel('Days', fontsize=30)
    plt.legend(fontsize=30)

    if plot_param == True:
        plt.suptitle(filename + ".pdf" + '(smoothed={})'.format(self.smoothing), fontsize=30)
        plt.title("beta={},sigma={},gamma={},hp={},".format(self.beta,
                                                               self.sigma,
                                                               self.gamma,
                                                               self.hp) +
                  "hcr={},pc={},pd={},pcr={},s={},t={}".format(self.hcr,
                                                               self.pc,
                                                               self.pd,
                                                               self.pcr,
                                                               self.s,
                                                               self.t) +
                  "\n \n" +
                  "S_0={},E_0={},I_0={},R_0={},H_0={},C_0={},D_0={}".format(self.S_0,
                                                                           self.E_0,
                                                                           self.I_0,
                                                                           self.R_0,
                                                                           self.H_0,
                                                                           self.C_0,
                                                                           self.D_0),
                  fontsize=20)

    else:
        plt.title(filename + '(smoothed={})'.format(self.smoothing),
                     fontsize=30)

    for tick in ax_plot_dataset.xaxis.get_major_ticks():
        tick.label.set_fontsize(30)
    for tick in ax_plot_dataset.yaxis.get_major_ticks():
        tick.label.set_fontsize(30)
    fig_plot_dataset.savefig('img/'+filename+".pdf")

    plt.close()
