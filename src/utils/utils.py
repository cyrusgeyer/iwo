import torch


class DiscreteLoss:
    def __init__(self, out_size):
        self.out_size = out_size
        self.loss = torch.nn.CrossEntropyLoss()
        self.baseline_score = (out_size - 1) / out_size
        self.zero_loss = -torch.log(
            torch.ones(1).exp() / (out_size - 1 + torch.ones(1).exp())
        )

    def __call__(self, prediction, targets):
        target_index = ((targets * (self.out_size - 1) + 1.0e-5)).type(torch.int64)
        loss = self.loss(prediction, target_index)
        correct_pred = prediction.detach().argmax(dim=1) == target_index
        score = 1 - torch.mean(correct_pred.type(torch.float))
        return loss, score


class ContLoss:
    def __init__(self, variance):
        self.variance = variance
        self.loss = torch.nn.MSELoss()
        self.baseline_score = 1.0
        self.zero_loss = torch.zeros(1)

    def __call__(self, prediction, targets):
        loss = self.loss(prediction.squeeze(), targets)
        score = loss.detach() / self.variance
        return loss, score


def get_variance(targets):
    return ((targets - targets.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)


def batch_eye(batch_size, dim1, dim2):
    return torch.eye(dim1, dim2).unsqueeze(0).repeat(batch_size, 1, 1)
