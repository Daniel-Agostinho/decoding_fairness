# Third party imports
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import torch
import torch.nn as nn

# My imports
from .tools import freeze_layers, unfreeze_layers


class EEGNet(nn.Module):
    def __init__(self, f1=8, f2=16, d=2, eeg_channels=25, p=0.25, n_classes=4):
        """
        :param f1: number of temporal filters
        :param f2: number of point-wise filters f2 = f1 * D
        :param d: number of spatial filters to learn within each temporal convolution
        :param eeg_channels: number of EEG channels
        :param p: dropout rate
        :param n_classes: number of output classes
        """
        super(EEGNet, self).__init__()
        self.num_classes = n_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, 64)),
            nn.BatchNorm2d(num_features=f1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(in_channels=f1, out_channels=f1 * d, groups=f1, kernel_size=(eeg_channels, 1)),
            nn.BatchNorm2d(num_features=f1 * d, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout2d(p=p),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=f1 * d, out_channels=f1 * d, groups=f1 * d, kernel_size=(1, 16), bias=False),
            nn.Conv2d(in_channels=f1 * d, out_channels=f2, kernel_size=(1, 1), stride=(1, 1), bias=False,
                      padding=(0, 0)),
            nn.BatchNorm2d(num_features=f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout2d(p=p),
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=self.num_classes),
            nn.Softmax(dim=1),
        )

        self.number_of_layers = 5

    def forward(self, x):
        x= self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def freeze_feature_layers(self):
        freeze_layers(self)

    def unfreeze_feature_layers(self):
        unfreeze_layers(self)


def evaluate_models_performance(model, test_dataset, logger, target_labels=None, target_labels_names=None):

    data, labels = test_dataset.data, test_dataset.labels
    model.eval()

    with torch.no_grad():
        prediction = model(data)
        prediction = torch.argmax(prediction, dim=1).detach().cpu().numpy()
        true_labels = torch.argmax(labels, dim=1).detach().cpu().numpy()

    test_info = _evaluation(prediction, true_labels, target_labels, target_labels_names)
    logger.log_test(model, test_info)


def _evaluation(prediction, true_labels, data_labels, target_labels_names):

    cm = confusion_matrix(true_labels, prediction)
    summary = classification_report(true_labels, prediction, labels=data_labels, target_names=target_labels_names)
    summary_data = classification_report(true_labels, prediction, labels=data_labels, target_names=target_labels_names,
                                         output_dict=True)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_labels_names)
    return cm, summary, summary_data, disp


if __name__ == '__main__':
    pass
