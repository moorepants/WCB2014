#/usr/bin/env python

from os.path import join, split
from gaitanalysis import motek, gait

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

#args = new_inv_dyn_labels + [dflow_data.meta['subject']['mass'],
                             #inv_dyn_low_pass_cutoff]
args = new_inv_dyn_labels + [101.0, inv_dyn_low_pass_cutoff]

perturbation_data.inverse_dynamics_2d(*args)

perturbation_data.raw_data.to_hdf(join(split(__file__)[0], '../data/perturbation.h5'), 'table')
