from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import RandomSampler

from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def get_dataloaders(config):
    """
    Create dataloaders for each of the dataset partitions.

    Args:
        config (DictConfig): hydra experiment config.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
    """

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        # dataset partition init
        dataset = instantiate(
            config.datasets[dataset_partition],
        )

        sampler = None
        if dataset_partition == "train" and config.train_samples_per_epoch is not None:
            sampler = RandomSampler(
                dataset,
                replacement=True,
                num_samples=config.train_samples_per_epoch,
            )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            sampler=sampler,
            drop_last=(dataset_partition == "train"),
            shuffle=False,
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders
