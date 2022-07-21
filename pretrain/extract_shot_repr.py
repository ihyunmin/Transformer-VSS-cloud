from utils.main_utils import (
    init_config,
    init_data_loader,
    init_model,
    init_trainer,
)


def main():
    # init configurations
    config_path = "configs/pretrain.yaml"
    cfg = init_config(config_path)
    cfg.LOAD_FROM = "swin"

    # init dataloader
    cfg, test_loader = init_data_loader(cfg, mode="extract_shot", is_train=False)

    # init model
    cfg, model = init_model(cfg)

    # init trainer
    cfg, trainer = init_trainer(cfg)

    # train
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()