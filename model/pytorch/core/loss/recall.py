"""
Author: Vincent Polfliet
Institute: SKENE
Year: 2022
"""

# External imports
import torch
import torch.nn as nn

# Internal imports


class Recall(nn.Module):

    def __init__(self, ignore_index=None):
        """
            Module to calculate the recall.

            Parameters:
                ignore_index: (int) ignore specific indices when calculating
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, y_ground_truth, y_prediction):
        """
            recall - expected format - C x N1 x N2 x ...

            Parameters:
            y_ground_truth:
               Ground truth mask
            y_prediction:
               Predicted mask
        """
        channel_indices = torch.arange(y_prediction.shape[0])
        y_ground_truth = y_ground_truth[channel_indices[channel_indices != self.ignore_index]]
        y_prediction = y_prediction[channel_indices[channel_indices != self.ignore_index]]
        smooth = 1
        y_prediction_positive = torch.clip(y_prediction, 0, 1)
        y_prediction_negative = 1 - y_prediction_positive
        y_positive = torch.clip(y_ground_truth, 0, 1)
        true_positive = torch.sum(y_positive * y_prediction_positive, dim=1)
        false_negative = torch.sum(y_positive * y_prediction_negative, dim=1)
        recall = (true_positive + smooth) / (true_positive + false_negative + smooth)
        return torch.mean(recall)
