import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.helpers import comm


class DistributedSampler(_DistributedSampler):
    """
    Drop-in DistributedSampler that lets us disable shuffling (useful for eval).
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=4, batch_size=None, dist=False, test_dist=False):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    bs = batch_size if batch_size is not None else cfg['batch_size']

    train_sampler = None
    test_sampler = None
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    if test_dist:
        rank, world_size = comm.get_rank(), comm.get_world_size()
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=bs,
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=(train_sampler is None),
                              pin_memory=False,
                              drop_last=False,
                              sampler=train_sampler)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=bs,
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False if test_sampler is None else False,
                             pin_memory=False,
                             drop_last=False,
                             sampler=test_sampler)

    return train_loader, test_loader, train_sampler, test_sampler
