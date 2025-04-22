# Built-in imports
import os
import csv
import pickle

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import torch

# My imports
from .tools import _make_dir


class WithinLogger:

    def __init__(self, experiment_identifier, subject_id, outer_fold):
        self.subject = subject_id
        self.outer_fold = outer_fold

        self.subject_folder = os.path.join(
            "Results",
            "Within_Subject",
            experiment_identifier,
            f"{self.subject}",
        )

        self.outer_fold_folder = os.path.join(
            self.subject_folder,
            f"Fold_{self.outer_fold}",
        )

        self.inner_fold_folder =None
        _make_dir(self.outer_fold_folder)

    def set_inner_fold_folder(self, inner_fold_folder):

        self.inner_fold_folder = os.path.join(
            self.outer_fold_folder,
            f"Fold_{inner_fold_folder}",
        )

        _make_dir(self.inner_fold_folder)

    def log_epoch_train(self, epoch, epoch_info):

        train_loss = epoch_info['train_loss']
        valid_loss = epoch_info['valid_loss']
        train_acc = epoch_info['train_info'][2]['accuracy']
        valid_acc = epoch_info['valid_info'][2]['accuracy']

        epoch_log_entry = [epoch, train_loss, train_acc, valid_loss, valid_acc]

        file_name = os.path.join(
            self.inner_fold_folder,
            "training_log.csv"
        )

        file_exists = os.path.isfile(file_name)

        with open(file_name, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Valid Accuracy"])
            writer.writerow(epoch_log_entry)


    def save_improved_model(self, model_data, epoch_info):
        train_info = epoch_info['train_info']
        valid_info = epoch_info['valid_info']

        model_path = os.path.join(
            self.inner_fold_folder,
            "best_model",
        )

        torch.save(model_data, model_path)
        self._save_info(train_info, target='train')
        self._save_info(valid_info, target='valid')

        print("Model Saved!")


    def _save_info(self, info, target='train'):

        main_out_put_folder = self.inner_fold_folder

        if target == 'test':
            main_out_put_folder = self.outer_fold_folder

        cm_file = os.path.join(
            main_out_put_folder,
            f"{target}_cm_data.npz"
        )

        np.savez_compressed(cm_file, cm=info[0])

        summary_file = os.path.join(
            main_out_put_folder,
            f"{target}_summary.txt"
        )

        with open(summary_file, "w") as file:
            file.write(info[1])

        summary_data_file = os.path.join(
            main_out_put_folder,
            f"{target}_summary_data.pkl"
        )

        with open(summary_data_file, "wb") as file:
            pickle.dump(info[2], file)

        confusion_matrix_file = os.path.join(
            main_out_put_folder,
            f"{target}_confusion_matrix",
        )

        plt.figure(figsize=(19.20, 10.80))
        info[3].plot(xticks_rotation='vertical')
        plt.tight_layout()
        plt.savefig(confusion_matrix_file)

    def get_best_model(self, best_cross_fold):

        model_path = os.path.join(
            self.outer_fold_folder,
            f"Fold_{best_cross_fold}",
            "best_model",
        )

        return model_path

    def log_test(self, model, info, data, labels):

        # Save test info
        self._save_info(info, target='test')

        # Save fold test data

        data_file = os.path.join(
            self.outer_fold_folder,
            "test_data.npz",
        )

        np.savez_compressed(data_file, data=data, labels=labels)

        # Save used model
        model_path = os.path.join(
            self.outer_fold_folder,
            "test_model",
        )

        torch.save(model.state_dict(), model_path)

        # Save Summary of outer folds
        outer_fold_summary = self._compile_result_summary(info)

        summary_file = os.path.join(
            self.subject_folder,
            "test_summary.csv",
        )

        file_exists = os.path.isfile(summary_file)

        with open(summary_file, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Accuracy", "Precision", "Recall", "F1-Score"])
            writer.writerow(outer_fold_summary)

    @staticmethod
    def _compile_result_summary(data):
        accuracy = data["accuracy"]
        precision = data['macro avg']["precision"]
        recall = data['macro avg']['recall']
        f1 = data['macro avg']['f1-score']
        return accuracy, precision, recall, f1


if __name__ == '__main__':
    pass
