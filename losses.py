import torch
import torch.nn.functional as F


def class_loss(pred, target, reduction="mean"):
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    loss_per_image = loss.mean(dim=(1, 2, 3))
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


def seg_loss(pred, target, gamma=0.04, reduction="mean"):
    gamma = torch.tensor(gamma, device=pred.device)
    pred_sigmoid = torch.sigmoid(pred)

    greater_gamma = target * torch.where(
        pred_sigmoid >= gamma, torch.tensor(1.0), torch.tensor(0.0)
    ) + (1 - target) * torch.where(
        1.0 - gamma >= pred_sigmoid, torch.tensor(1.0), torch.tensor(0.0)
    )

    loss_lower = target * (
        -torch.log(gamma) + 0.5 * (1 - pred_sigmoid ** 2 / gamma ** 2)
    ) + (1 - target) * (
        -torch.log(gamma) + 0.5 * (1 - (1 - pred_sigmoid) ** 2 / gamma ** 2)
    )
    loss_greater = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

    loss = loss_greater * greater_gamma + loss_lower * (1 - greater_gamma)

    loss_per_image = loss.mean(dim=(1, 2, 3))
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image


def synergistic_loss(pred, target, lambda_=0.5, reduction="mean"):
    loss_cla = class_loss(pred, target, reduction="none")
    loss_seg = seg_loss(pred, target, reduction="none")

    loss_per_image = loss_cla + lambda_ * loss_seg
    if reduction == "mean":
        return loss_per_image.mean()
    if reduction == "sum":
        return loss_per_image.sum()
    return loss_per_image
