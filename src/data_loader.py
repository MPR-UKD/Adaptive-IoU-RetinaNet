import pytorch_lightning as pl
from torch.utils.data import DataLoader


class PytorchLightningDataLoader(pl.LightningDataModule):
    def __init__(
        self, batch_size, cpu, dataset_train, dataset_test=None, dataset_val=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        self.cpu_count = cpu  # if batch_size > cpu else batch_size

    def train_dataloader(self):

        dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.cpu_count,
            persistent_workers=True if self.cpu_count != 0 else False,
        )
        return dataloader_train

    def val_dataloader(self):

        dataloader_val = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.cpu_count,
            persistent_workers=True if self.cpu_count != 0 else False,
        )
        return dataloader_val

    def test_dataloader(self):

        dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.cpu_count,
            persistent_workers=True if self.cpu_count != 0 else False,
        )
        return dataloader_test
