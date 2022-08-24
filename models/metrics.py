import torch
from torch import nn 

EPSILON = 1e-15


class DiceLoss(nn.Module):
    """
    Defining Loss function using Dice Loss, widely used to calculate the
    similarity between two images.

    Formula: 2(A intersect B)/(A + B)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Calculate dice loss of inputs and targets.

        Args:
            input (tensor): Output from the forward pass.
            targets (tensor): Original value
            smooth (float): Value to avoid division by zero

        Usage example:
            diceloss = DiceLoss()
            loss = diceloss(inputs, target)
            >> (tensor): Dice loss.
        """

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() +
                smooth)

        return 1 - dice


def binary_mean_iou(inputs, targets):
    """
    Calculate binary mean intersection over union.

    IoU formula = area of intersection/area of union

    Args:
        inputs (tensor): Output from the forward pass
        targets (tensor): Labels

    Returns (tensor): Intersection over union value

    Usage example:
        binary_mean_iou(inputs, target)
        >> (tensor)
    """
    output = (inputs > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result
