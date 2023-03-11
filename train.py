import platform
from multiprocessing import Manager

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

from src import *

assert torch.__version__.split(".")[0] == "1"


def to_list(number: int):
    gpu_list = []
    for g in str(number):
        gpu_list.append(int(g))
    return gpu_list


def get_args(args=None):
    """
    Helper function for the default settings under Windows and Linux
    """
    if platform.system() == "Windows":
        return get_args_windows(args)
    else:
        return get_args_server(args)


def get_dataloader(args, cf, epoch_manager):
    """
    Wrapper function that creates the three datasets (train, val, and test) and
    creates a PytorchLightningDataloaderModule for managing the datasets.
    """

    number_of_images = {"train": 300, "val": 50, "test": 50}
    datasets = [
        TestDataset(
            transformer=transforms.Compose(
                [
                    Augmenter(cf=cf, predict_mode=False if mode == "train" else True),
                    Normalizer(),
                ]
            ),
            number_of_images=number_of_images[mode],
            config=cf,
            pos=args.positive,
            neg=args.negative,
            adaptive_epochs=args.adaptive_epochs,
            epoch_dict=epoch_manager,
        )
        for mode in ["train", "val", "test"]
    ]

    dataset_train, dataset_val, dataset_test = tuple(datasets)

    dataloader = PytorchLightningDataLoader(
        batch_size=args.batch_size,
        cpu=args.num_workers,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        dataset_val=dataset_val,
    )

    return dataloader, dataset_train.num_classes()


def main(args=None):
    # Create a manager object to share data between processes
    manager = Manager()
    epoch_manager = manager.dict()
    epoch_manager["epoch"] = 0

    # Clear GPU memory and enable autograd anomaly detection
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    # Load configuration and command-line arguments
    cf = Config()
    args = get_args(args)

    # Create dataloader and retrieve number of classes
    dataloader, n_cls = get_dataloader(args, cf, epoch_manager)

    # Initialize RetinaNet model
    retinanet = RetinaNet(n_cls=n_cls, epoch_dict=epoch_manager, cf=cf)

    # Define metrics for ModelCheckpoint callback
    metrics = [
        ("Val_Accuracy", "max", 1),
        ("Erosion_Accuracy", "max", 1),
        ("mAP_0", "max", 1),
        ("mAP_1", "max", 1),
    ]
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor=metric[0],
            dirpath="./images_bbox/",
            filename=f"sample-mnist-{{epoch:02d}}-{metric[0]}={{metric:.3f}}",
            save_top_k=metric[2],
            mode=metric[1],
        )
        for metric in metrics
    ]

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        gpus=0 if not torch.cuda.is_available() else to_list(args.num_gpus),
        callbacks=checkpoint_callbacks,
        max_epochs=args.max_epochs,
        auto_lr_find=True,
        auto_scale_batch_size=False,
        fast_dev_run=False,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        benchmark=True,
    )

    # Train RetinaNet model
    # trainer.tune(retinanet, dataloader)
    trainer.fit(retinanet, dataloader)
    trainer.test(retinanet, dataloader)


if __name__ == "__main__":
    main()
