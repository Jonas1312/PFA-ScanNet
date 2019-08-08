import torch

from losses import seg_loss

EPS = 1e-6


def test_seg_loss_pos_class_greater_gamma():
    target = torch.ones((1, 1, 1, 1))
    pred_sigmoid = torch.tensor(0.1).view(target.size())
    pred_logit = -(-1 + 1.0 / pred_sigmoid).log()
    assert (
        abs(
            seg_loss(pred_logit, target, reduction="none").item()
            - (-torch.log(pred_sigmoid)).item()
        )
        < EPS
    )

    target = torch.ones((1, 1, 1, 1))
    pred_sigmoid = torch.tensor(0.9).view(target.size())
    pred_logit = -(-1 + 1.0 / pred_sigmoid).log()
    assert (
        abs(
            seg_loss(pred_logit, target, reduction="none").item()
            - (-torch.log(pred_sigmoid)).item()
        )
        < EPS
    )

    target = torch.ones((1, 1, 1, 1))
    pred_sigmoid = torch.tensor(0.98).view(target.size())
    pred_logit = -(-1 + 1.0 / pred_sigmoid).log()
    assert (
        abs(
            seg_loss(pred_logit, target, reduction="none").item()
            - (-torch.log(pred_sigmoid)).item()
        )
        < EPS
    )


def test_seg_loss_pos_class_lower_gamma():
    target = torch.ones((1, 1, 1, 1))
    pred_sigmoid = torch.tensor(0.03).view(target.size())
    pred_logit = -(-1 + 1.0 / pred_sigmoid).log()

    gamma = torch.tensor(0.04)

    assert (
        abs(
            seg_loss(pred_logit, target, reduction="none").item()
            - (-torch.log(gamma) + 0.5 * (1 - pred_sigmoid ** 2 / gamma ** 2)).item()
        )
        < EPS
    )


def test_seg_loss_neg_class_greater_gamma():
    target = torch.zeros((1, 1, 1, 1))
    pred_sigmoid = torch.tensor(0.1).view(target.size())
    pred_logit = -(-1 + 1.0 / pred_sigmoid).log()
    assert (
        abs(
            seg_loss(pred_logit, target, reduction="none").item()
            - (-torch.log(1 - pred_sigmoid)).item()
        )
        < EPS
    )

    target = torch.zeros((1, 1, 1, 1))
    pred_sigmoid = torch.tensor(0.9).view(target.size())
    pred_logit = -(-1 + 1.0 / pred_sigmoid).log()
    assert (
        abs(
            seg_loss(pred_logit, target, reduction="none").item()
            - (-torch.log(1 - pred_sigmoid)).item()
        )
        < EPS
    )

    target = torch.zeros((1, 1, 1, 1))
    pred_sigmoid = torch.tensor(0.03).view(target.size())
    pred_logit = -(-1 + 1.0 / pred_sigmoid).log()
    assert (
        abs(
            seg_loss(pred_logit, target, reduction="none").item()
            - (-torch.log(1 - pred_sigmoid)).item()
        )
        < EPS
    )


def test_seg_loss_neg_class_lower_gamma():
    target = torch.zeros((1, 1, 1, 1))
    pred_sigmoid = torch.tensor(0.97).view(target.size())
    pred_logit = -(-1 + 1.0 / pred_sigmoid).log()

    gamma = torch.tensor(0.04)

    assert (
        abs(
            seg_loss(pred_logit, target, reduction="none").item()
            - (
                -torch.log(gamma) + 0.5 * (1 - (1 - pred_sigmoid) ** 2 / gamma ** 2)
            ).item()
        )
        < EPS
    )
