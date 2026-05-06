"""Main experiment entry point."""

import hydra
from omegaconf import DictConfig

from adjoint_matching.algorithm import AdjointMatching
from adjoint_matching.utils import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    if cfg.logging.wandb:
        import wandb
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            config=dict(cfg),
        )

    algo = AdjointMatching(cfg)
    algo.run()


if __name__ == "__main__":
    main()
