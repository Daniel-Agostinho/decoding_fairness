# Build-in imports
import glob
import os

# Third party imports
import numpy as np

LABELS = {
    'faircongruent': 0,
    'fairincongruent': 1,
    'unfaircongruent': 2,
    'unfairincongruent': 3,
}

FAIRNESS = {
    'faircongruent': 1,
    'fairincongruent': 1,
    'unfaircongruent': 0,
    'unfairincongruent': 0,
}

CONGRUENCE = {
    'faircongruent': 1,
    'fairincongruent': 0,
    'unfaircongruent': 1,
    'unfairincongruent': 0,
}

EPOCH_LENGTH = 1500


def get_all_files(folder):
    complete_files = glob.glob(os.path.join(folder, "Lisboa", "*.txt"))
    incomplete_files = glob.glob(os.path.join(folder, "Coimbra", "*.txt"))
    return complete_files, incomplete_files


def _build_subject_key(subject):

    if len(subject) != 2:
        return subject.capitalize()

    subject_id = subject[-1]
    return f"P0{subject_id}"


def _get_subjects_files(all_files):
    subjects = np.unique(list(map(lambda x: os.path.split(x)[-1].split("_")[0], all_files)))

    out_subjects = {}

    for subject in subjects:
        subject_key = _build_subject_key(subject)
        subject_files = [x for x in all_files if subject in x]
        out_subjects[subject_key] = subject_files

    return out_subjects


def _get_file_condition_label(fpath):
    file_id = os.path.split(fpath)[-1]
    file_condition = file_id.split("_")[1]
    return LABELS[file_condition], FAIRNESS[file_condition], CONGRUENCE[file_condition]


def _get_file_data(fpath):

    data_label, fair_label, congruence_label = _get_file_condition_label(fpath)

    all_data = np.loadtxt(fpath, delimiter="\t")
    all_data = np.transpose(all_data)
    number_of_epochs = int(all_data.shape[1] / EPOCH_LENGTH)
    epoch_data = np.split(all_data, number_of_epochs, axis=1)
    epoch_data = np.asarray(epoch_data)

    labels = np.ones((number_of_epochs, 3))
    labels[:, 0] = data_label
    labels[:, 1] = fair_label
    labels[:, 2] = congruence_label
    return epoch_data, labels


def construct_subjects_data(all_files, dataset_name):

    subjects = _get_subjects_files(all_files)

    for subject in subjects:
        print(f"Building {subject} subjects data...")
        subject_data = None
        subject_labels = None

        subject_files = subjects[subject]

        for subject_file in subject_files:
            file_data, file_labels = _get_file_data(subject_file)

            if subject_data is None:
                subject_data = file_data
                subject_labels = file_labels
            else:
                subject_data = np.concatenate((subject_data, file_data), axis=0)
                subject_labels = np.concatenate((subject_labels, file_labels), axis=0)

        save_dataset(subject_data, subject_labels, subject, dataset_name)


def save_dataset(data, labels, subject_id, dataset_name):

    print(f"Saving dataset {subject_id} data ...")

    fpath = os.path.join(
        "Datasets",
        dataset_name,
    )

    _make_folders(fpath)

    file_path = os.path.join(
        fpath,
        f"{subject_id}.npz",
    )

    np.savez_compressed(file_path, data=data, labels=labels)


def _make_folders(fpath):

    if not os.path.exists(fpath):
        os.makedirs(fpath)


if __name__ == '__main__':
    pass
