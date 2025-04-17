# Built-in imports
import glob
import os

# Third party imports
import numpy as np

# My imports
from .datasets import SubjectEEGDataset


PRE_STIMULI_PERIOD_IN_POINTS = 500


def select_dataset_subjects(dataset_name):

    fpath = glob.glob(os.path.join(
        "Datasets",
        dataset_name,
        "*.npz",

    ))

    subjects = {os.path.split(x)[-1].split('.')[0]: x for x in fpath}
    return subjects


def _load_subject_data(fpath, target):

    all_data = np.load(fpath)
    data = all_data['data']
    labels = all_data['labels'][:,target]

    return data, labels


def _build_dataset_for_pytorch(subjects=None, subjects_files=None, data=None, labels=None, normalize_pre_signal=False):

    if subjects and subjects_files:
        pass

    pre_stimuli = data[:, :, :PRE_STIMULI_PERIOD_IN_POINTS]
    stimuli_data = data[:, :, PRE_STIMULI_PERIOD_IN_POINTS:]

    if normalize_pre_signal:
        stimuli_data = __normalized_for_pre_stimuli_mean(stimuli_data, pre_stimuli)

    standardized_stimuli_data = __standardize_temporal_dimension(stimuli_data)

    dataset_data, dataset_labels = __data_to_pytorch_format(standardized_stimuli_data, labels)
    return SubjectEEGDataset(dataset_data, dataset_labels)


def __normalized_for_pre_stimuli_mean(data, pre_stimuli_data):
    channels_mean = np.mean(pre_stimuli_data, axis=2)

    channels_mean = channels_mean.reshape(
        channels_mean.shape[0],
        channels_mean.shape[1],
        1,
    )

    normalized_data = data - channels_mean
    return normalized_data


def __standardize_temporal_dimension(data):

    data_mean = np.mean(data, axis=2)
    data_std = np.std(data, axis=2)

    data_mean = data_mean.reshape(
        data_mean.shape[0],
        data_mean.shape[1],
        1,
    )

    data_std = data_std.reshape(
        data_std.shape[0],
        data_std.shape[1],
        1,
    )

    demeaned_data = (data - data_mean)
    standardized_data = demeaned_data / data_std
    return standardized_data


def __data_to_pytorch_format(data, labels):

    data = __reshape_pytorch(data)
    labels = __convert_labels(labels)
    return data, labels


def __convert_labels(labels):
    n_classes = int(labels.max() + 1)
    new_labels = np.zeros((labels.shape[0], n_classes))

    for i in range(labels.shape[0]):
        new_labels[i, int(labels[i])] = 1

    return new_labels


def __reshape_pytorch(data):
    new_x = data.reshape((
        data.shape[0],
        1,
        data.shape[1],
        data.shape[2]
    ))

    return new_x


def _make_dir(fpath):

    if not os.path.isdir(fpath):
        os.makedirs(fpath)


def freeze_layers(model):
    all_layers = model.children()
    for i in range(model.number_of_layers - 1):
        layer2freeze = next(all_layers)
        for parameter in layer2freeze.parameters():
            parameter.requires_grad = False


def unfreeze_layers(model):
    all_layers = model.children()
    for i in range(model.number_of_layers - 1):
        layer2freeze = next(all_layers)
        for parameter in layer2freeze.parameters():
            parameter.requires_grad = True


if __name__ == '__main__':
    pass
