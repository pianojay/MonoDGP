import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import torch.distributed as dist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from tensorboardX import SummaryWriter
from lib.helpers.launch import launch
from lib.helpers import comm


def parse_config():
    parser = argparse.ArgumentParser(description='Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors')
    parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--batch_size', type=int, default=None, help='total batch size (will be split across GPUs for DDP)')
    parser.add_argument('--model_name', type=str, default=None, help='override model name used for outputs')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_machines', type=int, default=1, help='number of machines')
    parser.add_argument('--dist_url', type=str, default='auto', help='url used to set up distributed training')
    parser.add_argument('--max_epoch', type=int, default=None, help='override max epochs')
    return parser.parse_args()


def main(args):
    assert os.path.exists(args.config)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    distributed = comm.get_world_size() > 1
    cfg_local_rank = comm.get_local_rank() if distributed else args.local_rank
    cfg['local_rank'] = cfg_local_rank

    if args.model_name is not None:
        cfg['model_name'] = args.model_name
    model_name = cfg['model_name']

    if args.max_epoch is not None:
        cfg['trainer']['max_epoch'] = args.max_epoch

    total_gpus = comm.get_world_size() if distributed else 1
    if args.batch_size is None:
        batch_size = cfg['dataset']['batch_size']
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of GPUs'
        batch_size = args.batch_size // total_gpus
    cfg['dataset']['batch_size'] = batch_size

    set_random_seed(cfg.get('random_seed', 444) + cfg_local_rank)

    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file, rank=cfg_local_rank)

    # build dataloader (train uses distributed sampler when enabled; test stays single-process)
    train_loader, test_loader, train_sampler, _ = build_dataloader(cfg['dataset'],
                                                                   batch_size=batch_size,
                                                                   dist=distributed,
                                                                   test_dist=False)

    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda", cfg_local_rank) if torch.cuda.is_available() else torch.device("cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg_local_rank)

    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg_local_rank], find_unused_parameters=True)
    elif len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if args.evaluate_only:
        if distributed and not comm.is_main_process():
            comm.synchronize()
            dist.destroy_process_group()
            return
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        if distributed:
            dist.destroy_process_group()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    tb_log = SummaryWriter(log_dir=str(output_path + '/tensorboard')) if comm.is_main_process() else None

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name,
                      tb_log=tb_log,
                      train_sampler=train_sampler,
                      rank=cfg_local_rank,
                      distributed=distributed,
                      is_main_process=comm.is_main_process())

    tester = None
    if cfg['dataset']['test_split'] != 'test' and (not distributed or comm.is_main_process()):
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
    if tester is not None:
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size (per GPU): %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_config()

    if args.launcher == 'none':
        main(args)
    else:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.local_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
