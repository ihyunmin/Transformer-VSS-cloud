from utils.main_utils import (
    init_config,
    init_data_loader,
    init_model,
    init_trainer,
)

def main():
    # init configurations
    config_path = "configs/finetune.yaml"
    cfg = init_config(config_path)

    # init dataloader
    loaders = []
    cfg, train_loader = init_data_loader(cfg, mode="finetune", is_train=True)
    loaders.append(train_loader)
    cfg, val_loader = init_data_loader(cfg, mode="finetune", is_train=False)
    loaders.append(val_loader)

    # init model
    cfg, model = init_model(cfg)

    # init trainer
    cfg, trainer = init_trainer(cfg)

    # train
    trainer.fit(model, *loaders)


if __name__ == "__main__":
    main()