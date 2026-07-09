from hydra.utils import instantiate
from torch.utils.data import DataLoader


def build_loader(cfg, split: str, batch_size: int, num_workers: int = 4):
    dataset = instantiate(cfg, split=split)
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=shuffle,
    )
