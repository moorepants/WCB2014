#/usr/bin/env python

from os.path import join, split

import numpy as np
import matplotlib.pyplot as plt
import pandas
from dtk import process
from gaitanalysis import gait, controlid
from gaitanalysis.utils import _percent_formatter

directory = split(__file__)[0]

perturbation_data_frame = pandas.read_hdf(join(directory,
                                               '../data/perturbation.h5'),
                                          'table')

perturbation_data = gait.WalkingData(perturbation_data_frame)

# The following identifies the steps based on vertical ground reaction
# forces.
perturbation_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                filter_frequency=15.0,
                                num_steps_to_plot=None, do_plot=False,
                                threshold=30.0, min_time=290.0)
perturbation_data_right_steps = \
    perturbation_data.split_at('right', num_samples=20,
                               belt_speed_column='RightBeltSpeed')

# Controller identification.
sensors = ['Right.Ankle.Flexion.Angle',
           'Right.Ankle.Flexion.Rate',
           'Right.Knee.Flexion.Angle',
           'Right.Knee.Flexion.Rate',
           'Right.Hip.Flexion.Angle',
           'Right.Hip.Flexion.Rate',
           'Left.Ankle.Flexion.Angle',
           'Left.Ankle.Flexion.Rate',
           'Left.Knee.Flexion.Angle',
           'Left.Knee.Flexion.Rate',
           'Left.Hip.Flexion.Angle',
           'Left.Hip.Flexion.Rate']

controls = ['Right.Ankle.PlantarFlexion.Moment',
            'Right.Knee.PlantarFlexion.Moment',
            'Right.Hip.PlantarFlexion.Moment',
            'Left.Ankle.PlantarFlexion.Moment',
            'Left.Knee.PlantarFlexion.Moment',
            'Left.Hip.PlantarFlexion.Moment']

perturbation_data_solver = \
    controlid.SimpleControlSolver(perturbation_data_right_steps, sensors,
                                  controls)

gain_omission_matrix = np.zeros((len(controls), len(sensors))).astype(bool)
for i, row in enumerate(gain_omission_matrix):
    row[2 * i:2 * i + 2] = True

gains, controls_star, variance, gain_var, control_var, estimated_controls = \
    perturbation_data_solver.solve(gain_omission_matrix=gain_omission_matrix)

# Gain plot

fig_width_pt = 234.8775  # column width in abstract
inches_per_pt = 1.0 / 72.27

params = {'backend': 'ps',
          'font.family': 'serif',
          'font.serif': 'times',
          'axes.labelsize': 6,
          'text.fontsize': 6,
          'legend.fontsize': 6,
          'xtick.labelsize': 4,
          'ytick.labelsize': 4,
          'axes.titlesize': 6,
          'text.usetex': True,
          'figure.figsize': (fig_width_pt * inches_per_pt,
                             fig_width_pt * inches_per_pt * 0.80)}

plt.rcParams.update(params)

fig, axes = plt.subplots(3, 2, sharex=True)

for i, row in enumerate(['Ankle', 'Knee', 'Hip']):
    for j, (col, unit) in enumerate(zip(['Angle', 'Rate'],
                                        ['Nm/rad', r'Nm $\cdot$ s/rad'])):
        for side, marker, color in zip(['Right', 'Left'],
                                       ['o', 'o'],
                                       ['Blue', 'Red']):

            row_label = '.'.join([side, row, 'PlantarFlexion.Moment'])
            col_label = '.'.join([side, row, 'Flexion', col])

            gain_row_idx = controls.index(row_label)
            gain_col_idx = sensors.index(col_label)

            gains_per = gains[:, gain_row_idx, gain_col_idx]
            sigma = np.sqrt(gain_var[:, gain_row_idx, gain_col_idx])

            percent_of_gait_cycle = \
                perturbation_data_solver.identification_data.iloc[0].index.values.astype(float)

            xlim = (percent_of_gait_cycle[0], percent_of_gait_cycle[-1])

            if side == 'Left':
                # Shift that diggidty-dogg signal 50%
                # This only works for an even number of samples.
                if len(percent_of_gait_cycle) % 2 != 0:
                    raise StandardError("Doesn't work with odd samples.")

                first = percent_of_gait_cycle[percent_of_gait_cycle < 0.5] + 0.5
                second = percent_of_gait_cycle[percent_of_gait_cycle > 0.5] - 0.5
                percent_of_gait_cycle = np.hstack((first, second))

                # sort and sort gains/sigma same way
                sort_idx = np.argsort(percent_of_gait_cycle)
                percent_of_gait_cycle = percent_of_gait_cycle[sort_idx]
                gains_per = gains_per[sort_idx]
                sigma = sigma[sort_idx]

            axes[i, j].fill_between(percent_of_gait_cycle,
                                    gains_per - sigma,
                                    gains_per + sigma,
                                    alpha=0.5,
                                    color=color)

            axes[i, j].plot(percent_of_gait_cycle, gains_per,
                            marker='o',
                            ms=2,
                            color=color,
                            label=side)

            #axes[i, j].set_title(' '.join(col_label.split('.')[1:]))
            axes[i, j].set_title(r"{}: {} $\rightarrow$ Moment".format(row, col))

            axes[i, j].set_ylabel(unit)

            if i == 2:
                axes[i, j].set_xlabel(r'\% of Gait Cycle')
                axes[i, j].xaxis.set_major_formatter(_percent_formatter)
                axes[i, j].set_xlim(xlim)

plt.tight_layout()

fig.savefig(join(directory, '../fig/gains.pdf'))

# Fit plot.
estimated_walking = \
    pandas.concat([df for k, df in estimated_controls.iteritems()],
                  ignore_index=True)

actual_walking = \
    pandas.concat([df for k, df in
                   perturbation_data_solver.validation_data.iteritems()],
                  ignore_index=True)

params = {
          'axes.labelsize': 8,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          }

plt.rcParams.update(params)

fig, ax = plt.subplots(1)

sample_number = actual_walking.index.values
measured = actual_walking['Right.Ankle.PlantarFlexion.Moment'].values
predicted = estimated_walking['Right.Ankle.PlantarFlexion.Moment'].values
std_of_predicted = np.sqrt(variance) * np.ones_like(predicted)
error = measured - predicted
rms = np.sqrt(np.linalg.norm(error).mean())
r_squared = process.coefficient_of_determination(measured, predicted)

ax.plot(sample_number, measured, color='black')
ax.plot(sample_number, predicted, color='blue', ms=4)
#ax.errorbar(sample_number, predicted, yerr=std_of_predicted, fmt='.', ms=4)
ax.set_ylabel('Right Ankle Torque')
ax.set_xlabel('Sample Number')
# TODO : Figure out how to get matplotlib + tex to print the percent sign.
ax.legend(('Measured', r'Estimated [VAF={:1.0f}\%]'.format(r_squared * 100.0)))
ax.set_xlim((100, 200))

plt.tight_layout()

fig.savefig(join(directory, '../fig/fit.pdf'))
