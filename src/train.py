from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from .pl_module import LitIWO


def train_iwo_pipeline(cfg, train_dataset, val_dataset, test_dataset):
    shuffle = [True, False, False]

    dataloaders = [
        DataLoader(
            dataset=dataset,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle[i],
            num_workers=cfg.representation.num_workers,
        )
        for i, dataset in enumerate([train_dataset, val_dataset, test_dataset])
    ]

    _, (batch_inputs, _) = next(enumerate(dataloaders[0]))

    input_dim = batch_inputs.shape[1]

    factor_sizes = train_dataset._factor_sizes
    factor_discrete = train_dataset._factor_discrete

    if cfg.model.mode == "joint":
        from .models.joint import IWOModel
    elif cfg.model.mode == "independent":
        from .models.independent import IWOModel
    else:
        raise NotImplementedError

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    model = IWOModel(
        cfg_dict["model"],
        input_dim,
        factor_sizes,
        factor_discrete,
    )

    wandb_logger = WandbLogger(
        project=f"IWO_{cfg.training.wandb_project_name}", config=cfg_dict
    )
    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.training.accelerator,
    )

    litiwo = LitIWO(train_dataset, model, cfg)

    trainer.fit(litiwo, dataloaders[0], dataloaders[1])

    trainer.test(litiwo, dataloaders[2])
    iwo_test_out = litiwo.iwo_test_out
    # Construct out as such for backwards compatibility
    out = [
        {
            "scores_train": [score.cpu() for score in litiwo.scores["train"][k]],
            "scores_val": [score.cpu() for score in litiwo.scores["val"][k]],
            "scores": [score.cpu() for score in litiwo.scores["test"][k]],
            "Qs": [Q.cpu() for Q in litiwo.Qs[k]],
            "baseline": litiwo.baselines[k],
        }
        for k in range(len(factor_sizes))
    ]
    mean_scores = litiwo.mean_scores
    return out, mean_scores, iwo_test_out
