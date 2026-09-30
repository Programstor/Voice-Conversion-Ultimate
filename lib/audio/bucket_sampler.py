"""
bucket_sampler.py - length bucketing for the RVC training DataLoader.

Why: clips have different lengths and the collate function zero-pads every clip in a batch to
the longest one. A random batch of 1.5 s and 6 s clips wastes most of the compute on padding.
RVC's DistributedBucketSampler groups clips of similar length so each batch is nearly
rectangular. This is the single-GPU version: no distributed logic, nothing dropped.

Usage (batch_size / shuffle / drop_last must NOT be passed to DataLoader together with it):

    lengths = clip_lengths_in_frames(dataset.gt_dir, dataset.files, hop)   # 10 ms frames, dataset order
    sampler = BucketBatchSampler(lengths, batch_size=4, seed=1234)
    loader  = DataLoader(dataset, batch_sampler=sampler, num_workers=4,
                         pin_memory=True, collate_fn=rvc_collate,
                         persistent_workers=True)

Each epoch reshuffles both the order inside buckets and the order of batches (deterministic
for a given seed), so runs are reproducible.
"""
from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

from lib.audio.audio_io import audio_info


class BucketBatchSampler:
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        boundaries: Sequence[int] = (100, 200, 300, 400, 500, 600, 700, 800, 900),
        seed: int = 0,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        self.boundaries = tuple(sorted(boundaries))
        self.seed = seed
        self.epoch = 0
        # Clips outside the boundaries go into the first/last bucket (RVC drops them; we keep them).
        buckets = defaultdict(list)
        for idx, length in enumerate(lengths):
            buckets[bisect.bisect_right(self.boundaries, length)].append(idx)
        self.buckets = [b for _, b in sorted(buckets.items())]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)

    def __iter__(self) -> Iterator[list]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1                       # next call to iter() gives a new order
        batches = []
        for bucket in self.buckets:
            idx = bucket[:]
            rng.shuffle(idx)
            remainder = (-len(idx)) % self.batch_size
            if remainder:                     # top up the last batch with repeats from the same bucket
                idx += rng.choices(idx, k=remainder)
            batches.extend(idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size))
        rng.shuffle(batches)
        yield from batches


def clip_lengths_in_frames(gt_dir, names: Sequence[str], hop: int) -> list:
    """Length in model frames (sr // 100 samples each) of gt_dir/<name>.wav for every name.

    `names` MUST be in the same order as the Dataset (pass dataset.files), because the sampler
    returns dataset indices.
    """
    gt_dir = Path(gt_dir)
    return [audio_info(gt_dir / f"{name}.wav").frames // hop for name in names]