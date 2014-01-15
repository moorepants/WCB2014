#/usr/bin/env python

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas
from dtk import process
from gaitanalysis import motek, gait, controlid
from gaitanalysis.utils import _percent_formatter

# TODO : Setup some kind of dependency chain so the big computations can be
# avoided if the previous data is fine.

# TODO : Load this path from a configuration file, our download from
# Figshare once I post the data.
root_data_directory = "/home/moorepants/Data/human-gait/gait-control-identification"

mocap_file_path = join(root_data_directory, 'T006', 'mocap-006.txt')
record_file_path = join(root_data_directory, 'T006', 'record-006.txt')
meta_file_path = join(root_data_directory, 'T006', 'meta-006.yml')

dflow_data = motek.DFlowData(mocap_file_path, record_file_path,
                             meta_file_path)
dflow_data.clean_data(interpolate_markers=True)

# 'TreadmillPerturbation' is the current name of the longitudinal
# perturbation trials. This returns a data frame of processed data.
perturbation_data_frame = \
    dflow_data.extract_processed_data(event='TreadmillPerturbation',
                                      index_col='TimeStamp')

# Here I compute the joint angles, rates, and torques.
inv_dyn_low_pass_cutoff = 6.0  # Hz
inv_dyn_labels = motek.markers_for_2D_inverse_dynamics()


def add_negative_columns(data):
    """Creates new columns in the DataFrame for any D-Flow measurements in
    the Z axis."""
    new_inv_dyn_labels = []
    for label_set in inv_dyn_labels:
        new_label_set = []
        for label in label_set:
            if 'Z' in label:
                new_label = 'Negative' + label
                data[new_label] = -data[label]
            else:
                new_label = label
            new_label_set.append(new_label)
        new_inv_dyn_labels.append(new_label_set)
    return new_inv_dyn_labels


new_inv_dyn_labels = add_negative_columns(perturbation_data_frame)

perturbation_data = gait.WalkingData(perturbation_data_frame)

args = new_inv_dyn_labels + [dflow_data.meta['subject']['mass'],
                             inv_dyn_low_pass_cutoff]

perturbation_data.inverse_dynamics_2d(*args)

# The following identifies the steps based on vertical ground reaction
# forces.
perturbation_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                filter_frequency=15.0,
                                num_steps_to_plot=None, do_plot=True,
                                threshold=30.0, min_time=290.0)
perturbation_data_right_steps = perturbation_data.split_at('right',
                                                           num_samples=20)

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

# Fit plot.
estimated_walking = \
    pandas.concat([df for k, df in estimated_controls.iteritems()],
                  ignore_index=True)

actual_walking = \
    pandas.concat([df for k, df in
                   perturbation_data_solver.validation_data.iteritems()],
                  ignore_index=True)

params = {'figure.figsize': (fig_width_pt * inches_per_pt,
                             fig_width_pt * inches_per_pt * 0.5)}

plt.rcParams.update(params)

fig, ax = plt.subplots(1)

sample_number = actual_walking.index.values
measured = actual_walking['Right.Ankle.PlantarFlexion.Moment'].values
predicted = estimated_walking['Right.Ankle.PlantarFlexion.Moment'].values
std_of_predicted = np.sqrt(variance) * np.ones_like(predicted)
error = measured - predicted
rms = np.sqrt(np.linalg.norm(error).mean())
r_squared = process.coefficient_of_determination(measured, predicted)

ax.plot(sample_number, measured, color='black', marker='.', ms=6)
ax.errorbar(sample_number, predicted, yerr=std_of_predicted, fmt='.', ms=4)
ax.set_ylabel('Right Ankle Torque')
ax.set_xlabel('Sample Number')
# TODO : Figure out how to get matplotlib + tex to print the percent sign.
ax.legend(('Measured', 'Estimated [VAF={:1.1%}%]'.format(r_squared)))
ax.set_xlim((100, 200))

plt.tight_layout()

fig.savefig('../fig/fit.pdf')
