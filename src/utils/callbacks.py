import lightning as L
import torch
from .iwo import get_basis, calculate_iwo


class IWOCallback(L.Callback):
    def __init__(self, num_factors, baselines, cfg):
        super().__init__()
        self.num_factors = num_factors
        self.score_buffer = {}
        for mode in ["train", "val", "test"]:
            self.score_buffer[mode] = []
        self.mean_scores_all_epochs = []
        self.baselines = baselines
        self.cfg = cfg
        self.indenpendent_models = cfg["model"]["mode"] == "independent"
        if self.indenpendent_models:
            self.early_stoppings = [
                EarlyStopping(cfg["training"]["es_patience"])
                for _ in range(num_factors)
            ]
            self.frozen_scores = [
                torch.tensor(
                    [float("inf") for _ in range(cfg["model"]["default"]["first_dim"])]
                )
                for _ in range(num_factors)
            ]

    def iwo(self, trainer, pl_module, all_scores, mode):
        B_per_factor = []
        w_list = pl_module.model.get_w()

        for k, scores in enumerate(all_scores):
            B = get_basis(w_list[k])
            B_per_factor.append(B)

            # Log values in Module
            if mode == "train":
                pl_module.Bs[k].append(B)
                pl_module.Ws[k].append(w_list[k])

            pl_module.scores[mode][k].append(scores)

        iwo_list, iwr_list, mw_iwo, mw_iwr, importance, _ = calculate_iwo(
            B_per_factor, all_scores, self.baselines
        )
        for k in range(self.num_factors):
            pl_module.log(f"Factor_{k}/{mode}/iwo", iwo_list[k])
            pl_module.log(f"Factor_{k}/{mode}/iwr", iwr_list[k])

        pl_module.log(f"Mean_iwo/{mode}", mw_iwo)
        pl_module.log(f"Mean_iwr/{mode}", mw_iwr)
        if mode == "test":
            pl_module.iwo_test_out["iwo_list"] = iwo_list
            pl_module.iwo_test_out["iwr_list"] = iwr_list
            pl_module.iwo_test_out["mw_iwo"] = mw_iwo
            pl_module.iwo_test_out["mw_iwr"] = mw_iwr
            pl_module.iwo_test_out["importance"] = importance

    def on_train_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, "test")

    def on_epoch_end(self, trainer, pl_module, mode):
        all_scores = self.log_scores(trainer, pl_module, mode)
        if self.cfg["training"]["log_iwo"]:
            self.iwo(trainer, pl_module, all_scores, mode)
        self.score_buffer[mode] = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.on_batch_end(outputs, mode="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.on_batch_end(outputs, mode="val")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.on_batch_end(outputs, mode="test")

    def on_batch_end(self, outputs, mode):
        self.score_buffer[mode].append(outputs["scores"])
        return

    def log_scores(self, trainer, pl_module, mode):
        _all_scores = torch.stack(self.score_buffer[mode]).mean(0)
        mean_score = _all_scores.mean().item()
        pl_module.log(f"Mean_score/{mode}", mean_score)
        pl_module.mean_scores[mode].append(_all_scores.mean().item())
        if not self.indenpendent_models:
            # log all scores
            for k, scores in enumerate(_all_scores):
                for d in range(scores.shape[0]):
                    pl_module.log(f"Factor_{k}/{mode}/score_{d}", scores[d].item())
            return _all_scores

        all_scores = []
        _k = 0
        for k in range(self.num_factors):
            if (
                pl_module.model.per_factor_training != "all"
                and pl_module.model.per_factor_training != k
            ):
                all_scores.append(self.frozen_scores[k].to(pl_module.device))
            else:
                scores = _all_scores[_k]
                _k += 1
                all_scores.append(scores)
                stop = self.early_stoppings[k].should_stop(scores)
                if stop:
                    self.frozen_scores[k] = scores
                    pl_module.model.train_factor(k, False)
                # log the factor score
                for d in range(scores.shape[0]):
                    pl_module.log(f"Factor_{k}/{mode}/score_{d}", scores[d].item())
                mean_per_factor_score = scores.mean().item()
                pl_module.log(f"Factor_{k}/{mode}/mean_score", mean_per_factor_score)
        if _k == 0:
            trainer.should_stop = True
        return torch.stack(all_scores)


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_losses = None
        self.epochs_without_improvement = None

    def should_stop(self, epoch_losses):
        if self.best_losses is None or self.epochs_without_improvement is None:
            self.best_losses = [torch.tensor(float("inf")) for _ in epoch_losses]
            self.epochs_without_improvement = [0 for _ in epoch_losses]

        all_not_improved = True
        for i, loss in enumerate(epoch_losses):
            if loss < self.best_losses[i]:
                self.best_losses[i] = loss
                self.epochs_without_improvement[i] = 0
                all_not_improved = False
            else:
                self.epochs_without_improvement[i] += 1

        if all_not_improved and all(
            [e >= self.patience for e in self.epochs_without_improvement]
        ):
            return False  # Never stop early for now. –> TODO: Modify this to actually implement early stopping.
        return False
