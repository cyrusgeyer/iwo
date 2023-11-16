import torch
import lightning as L
from .utils.utils import (
    get_variance,
    DiscreteLoss,
    ContLoss,
)
from .utils.callbacks import IWOCallback


class LitIWO(L.LightningModule):
    def __init__(self, train_dataset, model, cfg):
        super().__init__()

        # Register Model:
        self.model = model

        factor_sizes = train_dataset._factor_sizes
        self.num_factors = len(factor_sizes)
        factor_discrete = train_dataset._factor_discrete
        self.cfg = cfg

        # Score for model_selection
        self.mean_scores = {}
        # Info for IWO-calculations. Populated by IWO callback
        self.scores = {}
        for mode in ["train", "val", "test"]:
            self.mean_scores[mode] = []
            self.scores[mode] = [[] for _ in range(self.num_factors)]

        self.Qs = [[] for _ in range(self.num_factors)]
        self.baselines = []

        # Make Losses:
        self.loss_functions = []
        for k in range(self.num_factors):
            if factor_discrete[k]:
                loss = DiscreteLoss(factor_sizes[k])
            else:
                variance = get_variance(train_dataset.normalized_targets[:, k])
                loss = ContLoss(variance)
            self.loss_functions.append(loss)
            self.baselines.append(loss.baseline_score)

        self.iwo_test_out = {}

    def training_step(self, batch, batch_idx):
        inputs, all_targets = batch

        all_losses = []
        all_scores = []

        pred_per_factors = self.model(inputs)

        for k, pred in enumerate(pred_per_factors):
            if pred is None:
                continue
            losses = []
            scores = []
            targets = all_targets[:, k]
            for dim_wise_prediction in pred:
                loss, score = self.loss_functions[k](dim_wise_prediction, targets)
                losses.append(loss)
                scores.append(score.detach())
            all_losses.append(torch.stack(losses))
            all_scores.append(torch.stack(scores))

        all_losses = torch.stack(all_losses)
        all_scores = torch.stack(all_scores)

        loss = all_losses.sum()
        return {"loss": loss, "scores": all_scores}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        if self.cfg.training.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        elif self.cfg.training.optimizer == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr=self.cfg.training.lr)
        else:
            raise NotImplementedError(
                f"Optimizer {self.cfg.training.optimizer} not implemented. but can be added."
            )

    def configure_callbacks(self):
        baselines = [
            self.loss_functions[k].baseline_score for k in range(self.num_factors)
        ]
        iwo = IWOCallback(self.num_factors, baselines, self.cfg)
        return [iwo]
