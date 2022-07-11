import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    """
        Pytorch criterion loss implementation of the dice score
    """

    def init(self):
        """
            Initialise this criterion
        """
        super(DiceLoss, self).init()

    def forward(self, predictions, target):
        """
            Runs the dice score formula on a prediction and the targets

            Args:
                predictions: a tensor containing values between 0 and 1, predicted values
                target: a tensor containing values between 0 and 1, true values
            Returns:
                dice score, pytorch tensor
        """
        smooth = 1.
        iflat = predictions.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
