import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
from src.train import train_iwo_pipeline

import numpy as np
import json

# custom list merge resolver
OmegaConf.register_new_resolver("merge", lambda *args: ",".join(map(str, args)))


@hydra.main(version_base=None, config_path="conf", config_name="default_synthetic")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    if cfg.representation.name == "synthetic":
        from src.utils.representation_synthetic import make_datasets
    elif cfg.representation.name in ["dsprites", "cars3d", "smallnorb"]:
        from src.utils.representation_learned import make_datasets
    else:
        raise NotImplementedError

    train_dataset, val_dataset, test_dataset = make_datasets(cfg.representation)

    all_outs, mean_scores, iwo_test_out = train_iwo_pipeline(
        cfg, train_dataset, val_dataset, test_dataset
    )

    mean_iwo = np.mean(iwo_test_out["mw_iwo"])
    mean_rnk = np.mean(iwo_test_out["mw_iwr"])

    if cfg.training.debug or cfg.representation.name == "synthetic":
        print(f"Mean IWO: {mean_iwo}")
        print(f"Mean IWR: {mean_rnk}")
        # We return min of validation mean_scores for hyperparameter optimization.
        return min(mean_scores["val"])

    # Write results into the file structure provided by disentanglement_lib
    metric_dir = os.path.join(cfg.representation.exp_path, "metrics/mean/iwo")
    json_dir = os.path.join(metric_dir, "results/json")
    out_dir = os.path.join(metric_dir, "out")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(json_dir, "evaluation_results.json")

    with open(json_path, "w") as f:
        json.dump({"iwo": mean_iwo, "mean_rank": mean_rnk}, f, indent=4)

    with open(os.path.join(out_dir, "out.pkl"), "wb") as file:
        pickle.dump(all_outs, file)

    with open(os.path.join(out_dir, "iwo_dict.pkl"), "wb") as file:
        pickle.dump(iwo_test_out, file)


if __name__ == "__main__":
    main()
