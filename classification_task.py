# Third party imports
import torch

# My imports
from src.classification.iterators import select_iterator
from src.classification.loggers import WithinLogger
from src.classification.models import EEGNet, evaluate_models_performance
from src.classification.trainners import ModelTrainner


CLASSES_FOR_TARGET = {
    0: 4,
    1: 2,
    2: 2,
}

LABELS = {
    0: [0, 1, 2, 3],
    1: [0, 1],
    2: [0, 1],
}

TARGET_NAMES = {
    0: ['Fair Congruent', 'Fair Incongruent', 'Unfair Congruent', 'Unfair Incongruent'],
    1: ['Unfair', 'Fair'],
    2: ['Incongruent', 'Congruent'],
}


def main():

    experiment_parameters = {
        'approach': "Within Subject",
        'dataset': "Complete",
        'outer_k_folds': 10,
        'inner_k_folds': 10,
        'target': 0,
        'normalize_pre_signal': False,
        'identifier': "WS_4C_NN",
    }

    network_parameters = {
        "f1": 8,
        "eeg_channels": 25,
        "f2": 16,
        "d": 2,
        "p": 0.5,
        "n_classes": CLASSES_FOR_TARGET[experiment_parameters['target']],
    }

    trainner_parameters = {
        "max_epochs": 600,
        "early_stop_limit": 150,
        "learning_rate": 1e-3,
        "batch_size": 246,
        "device": 0,
        "target_labels": LABELS[experiment_parameters['target']],
        "target_labels_names": TARGET_NAMES[experiment_parameters['target']],
    }

    evaluation_parameters = {
        "target_labels": LABELS[experiment_parameters['target']],
        "target_labels_names": TARGET_NAMES[experiment_parameters['target']],
    }

    run_experiment(experiment_parameters, network_parameters, trainner_parameters, evaluation_parameters)


def run_experiment(experiment_parameters, network_parameters, trainner_parameters, evaluation_parameters):

    type_of_experiment = experiment_parameters['approach']
    experiment_identifier = experiment_parameters['identifier']
    experiment_parameters.pop('approach')
    experiment_parameters.pop('identifier')

    data_iterator = select_iterator(type_of_experiment, experiment_parameters)

    if type_of_experiment == "Within Subject":
        run_within_subject_experiment(experiment_identifier, data_iterator, network_parameters, trainner_parameters,
                                      evaluation_parameters)


def run_within_subject_experiment(experiment_identifier, data_iterator, network_parameters, trainner_parameters,
                                  evaluation_parameters):

    for outer_fold, subject, test_dataset, train_generator in data_iterator:

        logger = WithinLogger(experiment_identifier, subject, outer_fold)
        best_cross_fold = -1
        best_cross_loss = 1000

        # Train
        for inner_fold, train, valid in train_generator:

            # Create new model
            model = EEGNet(**network_parameters)

            logger.set_inner_fold_folder(inner_fold)
            trainner = ModelTrainner(model=model, train_dataset=train, valid_dataset=valid, **trainner_parameters)

            inner_fold_loss = trainner.train(logger)

            if inner_fold_loss < best_cross_loss:
                best_cross_loss = inner_fold_loss
                best_cross_fold = inner_fold

        # Test
        best_model_fpath = logger.get_best_model(best_cross_fold)

        test_model = EEGNet(**network_parameters)
        test_model.load_state_dict(torch.load(best_model_fpath))

        evaluate_models_performance(test_model, test_dataset, logger, **evaluation_parameters)




if __name__ == '__main__':
    main()
