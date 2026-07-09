from collections import defaultdict

from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader, Sampler


class _ModeBatchSampler(Sampler[list[int]]):
    """Groups active dataset indices by mode so target tensors stay collatable."""

    def __init__(self, dataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        batches = []
        positions_by_mode = defaultdict(list)
        for position, sample_idx in enumerate(self.dataset.active_indices):
            mode = self.dataset.samples[sample_idx].mode
            positions_by_mode[mode].append(position)

        for positions in positions_by_mode.values():
            ordered = list(positions)
            if self.shuffle and len(ordered) > 1:
                perm = torch.randperm(len(ordered)).tolist()
                ordered = [ordered[idx] for idx in perm]
            for start in range(0, len(ordered), self.batch_size):
                batches.append(ordered[start : start + self.batch_size])

        if self.shuffle and len(batches) > 1:
            perm = torch.randperm(len(batches)).tolist()
            batches = [batches[idx] for idx in perm]

        yield from batches

    def __len__(self) -> int:
        total = 0
        positions_by_mode = defaultdict(int)
        for sample_idx in self.dataset.active_indices:
            mode = self.dataset.samples[sample_idx].mode
            positions_by_mode[mode] += 1
        for count in positions_by_mode.values():
            total += (count + self.batch_size - 1) // self.batch_size
        return total


def build_loader(cfg, split: str, batch_size: int, num_workers: int = 4):
    dataset = instantiate(cfg, split=split)
    shuffle = split == "train"
    batch_sampler = _ModeBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
