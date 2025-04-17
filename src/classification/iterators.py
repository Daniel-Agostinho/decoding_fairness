# Third party imports
from sklearn.model_selection import StratifiedKFold

# My imports
from .tools import select_dataset_subjects, _load_subject_data, _build_dataset_for_pytorch


def select_iterator(iterator, parameters):

    if iterator == "Within Subject":
        return within_iterator(**parameters)

    return loso_iterator(**parameters)


def within_iterator(dataset='Complete', outer_k_folds=10, target=0, normalize_pre_signal=False, inner_k_folds=10):
    subjects_files = select_dataset_subjects(dataset)
    subjects = list(subjects_files.keys())

    for subject in subjects:
        subject_fpath = subjects_files[subject]
        subject_data, subject_labels = _load_subject_data(subject_fpath, target)

        skf = StratifiedKFold(n_splits=outer_k_folds, shuffle=True)

        for outer_n_fold, test_fold_indexes in enumerate(skf.split(subject_data, subject_labels)):
            train_index, test_index = test_fold_indexes

            test_data, test_labels = subject_data[test_index, ...], subject_labels[test_index]
            train_data, train_labels = subject_data[train_index, ...], subject_labels[train_index]

            test_dataset = _build_dataset_for_pytorch(data=test_data, labels=test_labels,
                                                      normalize_pre_signal=normalize_pre_signal)

            train_generator = _within_iterator_inner_fold(train_data, train_labels, inner_k_folds, normalize_pre_signal)

            yield outer_n_fold, subject, test_dataset, train_generator


def _within_iterator_inner_fold(data, labels, k_folds, normalize_pre_signal):
    skf_train = StratifiedKFold(n_splits=k_folds, shuffle=True)

    for inner_n_fold, train_fold_indexes in enumerate(skf_train.split(data, labels)):
        train_index, valid_index = train_fold_indexes

        train_data, train_labels = data[train_index], labels[train_index]
        valid_data, valid_labels = data[valid_index], labels[valid_index]

        train_dataset = _build_dataset_for_pytorch(data=train_data, labels=train_labels,
                                                   normalize_pre_signal=normalize_pre_signal)

        valid_dataset = _build_dataset_for_pytorch(data=valid_data, labels=valid_labels,
                                                   normalize_pre_signal=normalize_pre_signal)

        yield inner_n_fold, train_dataset, valid_dataset


def loso_iterator(dataset='Complete', k_folds=10):
    pass


if __name__ == '__main__':
    pass
