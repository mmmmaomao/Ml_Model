import _init_path
import argparse
import datetime
import os

from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter

from src.dataset import collate_fn, meta_collate_fn, SubsetSequentialSampler
from src.dataset.cifar10 import MetaCifar10, Cifar10
from src.model import Learner
from tools.common_utils import common_utils
from tools.test_utils.test_utils import test, test_classes
from tools.train_utils.train_utils import meta_active_train, train, active_train
from configs.config import cfg, cfg_from_yaml_file


def parse_config():
    argparser = argparse.ArgumentParser(description='arg parser')
    argparser.add_argument('--cfg_file', type=str, help='specify the config',
                           default='/cpfs2/user/wangjie/PythonProjects/Ml_Model/configs/test.yaml')
    argparser.add_argument('--tag', type=str, help='tag', default='default')
    argparser.add_argument('--epochs', type=int, default=None)
    args = argparser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    '''
    model_cfg = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [10, 32 * 2 * 2])
    ]
    '''
    model_cfg = [('conv2d', [64,3,3,3,1,1]),('bn', [64]),('relu', [True]),
                ('conv2d', [64,64,3,3,1,1]),('bn', [64]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('conv2d', [128,64,3,3,1,1]),('bn', [128]),('relu', [True]),
                ('conv2d', [128,128,3,3,1,1]),('bn', [128]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('conv2d', [256,128,3,3,1,1]),('bn', [256]),('relu', [True]),
                ('conv2d', [256,256,3,3,1,1]),('bn', [256]),('relu', [True]),
                ('conv2d', [256,256,3,3,1,1]),('bn', [256]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('conv2d', [512,256,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('flatten', []),
                ('linear', [512, 512]),
                ('relu', [True]),
                ('linear', [512, 512]),
                ('relu', [True]),
                ('linear', [10, 512])]

    cfg.MODEL = model_cfg
    return args, cfg


def main():
    args, cfg = parse_config()

    cfg.OPTIMIZATION.EPOCHS = cfg.OPTIMIZATION.EPOCHS if args.epochs is None else args.epochs

    # ------------------------------------------------------------------------
    output_dir = cfg.ROOT_DIR / 'output' / args.tag
    ckpt_dir = output_dir / 'ckpt'
    result_dir = output_dir / 'result'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)

    common_utils.fix_seeds()

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    # -----------------------create dataloader---------------------------
    train_dataset = Cifar10(split='train')
    meta_dataset = MetaCifar10(n_task=cfg.OPTIMIZATION.TASK_NUMS,
                               n_support=cfg.OPTIMIZATION.SUPPORT_SIZE,
                               n_query=cfg.OPTIMIZATION.QUERY_SIZE,
                               split='train')
    meta_dataloader = DataLoader(meta_dataset, cfg.OPTIMIZATION.TASK_BATCH, num_workers=0, collate_fn=meta_collate_fn)

    # init labeled train dataset
    init_labeled_index = np.random.choice(list(range(meta_dataset.nums_train_data)),
                                          cfg.OPTIMIZATION.INIT_LABELED_SIZE,
                                          replace=False).tolist()
    labeled_index_lists = {}
    # ---------------------------meta active train----------------------------
    model = Learner(cfg.MODEL)
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.META_LR)
    meta_al_entropy_labeled_index_list = meta_active_train(cfg=cfg.OPTIMIZATION,
                                                           model=model,
                                                           optimizer=optimizer,
                                                           meta_dataset=meta_dataset,
                                                           meta_dataloader=meta_dataloader,
                                                           train_dataset=train_dataset,
                                                           uncertainty_type='entropy',
                                                           init_labeled_index=init_labeled_index,
                                                           active_train_nums=cfg.OPTIMIZATION.ACTIVE_TRAIN_NUMS,
                                                           logger=logger,
                                                           result_dir=result_dir)
    labeled_index_lists['meta_al_entropy'] = meta_al_entropy_labeled_index_list
    model = Learner(cfg.MODEL)
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.META_LR)
    meta_al_margin_labeled_index_list = meta_active_train(cfg=cfg.OPTIMIZATION,
                                                          model=model,
                                                          optimizer=optimizer,
                                                          meta_dataset=meta_dataset,
                                                          meta_dataloader=meta_dataloader,
                                                          train_dataset=train_dataset,
                                                          uncertainty_type='margin',
                                                          init_labeled_index=init_labeled_index,
                                                          active_train_nums=cfg.OPTIMIZATION.ACTIVE_TRAIN_NUMS,
                                                          logger=logger,
                                                          result_dir=result_dir)
    labeled_index_lists['meta_al_margin'] = meta_al_margin_labeled_index_list
    # -------------------------active train-----------------------------------
    # entropy
    model = Learner(cfg.MODEL)
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.UPDATE_LR)
    al_entropy_labeled_index_list = active_train(cfg=cfg.OPTIMIZATION,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 train_dataset=train_dataset,
                                                 uncertainty_type='entropy',
                                                 init_labeled_index=init_labeled_index,
                                                 active_train_nums=cfg.OPTIMIZATION.ACTIVE_TRAIN_NUMS,
                                                 logger=logger,
                                                 result_dir=result_dir)
    labeled_index_lists['al_entropy'] = al_entropy_labeled_index_list
    # margin
    model = Learner(cfg.MODEL)
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.UPDATE_LR)
    al_margin_labeled_index_list = active_train(cfg=cfg.OPTIMIZATION,
                                                model=model,
                                                optimizer=optimizer,
                                                train_dataset=train_dataset,
                                                uncertainty_type='margin',
                                                init_labeled_index=init_labeled_index,
                                                active_train_nums=cfg.OPTIMIZATION.ACTIVE_TRAIN_NUMS,
                                                logger=logger,
                                                result_dir=result_dir)
    labeled_index_lists['al_margin'] = al_margin_labeled_index_list
    # -----------------------------random pick---------------------------------------
    labeled_index = init_labeled_index.copy()
    random_index_list = [labeled_index.copy()]
    total_num = len(train_dataset)
    for i in range(cfg.OPTIMIZATION.ACTIVE_TRAIN_NUMS):
        unlabeled_index = [x for x in list(range(total_num)) if x not in labeled_index]
        picked_index = np.random.choice(unlabeled_index, cfg.OPTIMIZATION.PICK_NUMS, replace=False)
        labeled_index.extend(picked_index)
        random_index_list.append(labeled_index.copy())
    labeled_index_lists['random'] = random_index_list
    # ---------------------------test----------------------------------
    logger.info('**************************start testing*************************************')
    train_dataset = Cifar10('train')
    test_dataset = Cifar10('test')
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=cfg.OPTIMIZATION.BATCH_SIZE,
                                 collate_fn=collate_fn)
    name = ['apl', 'atmb', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for al_type, index_lists in labeled_index_lists.items():
        for step, l_index in enumerate(index_lists):
            labeled_dataloader = DataLoader(train_dataset,
                                            batch_size=cfg.OPTIMIZATION.BATCH_SIZE,
                                            sampler=SubsetSequentialSampler(l_index),
                                            collate_fn=collate_fn)
            model = Learner(cfg.MODEL)
            optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.UPDATE_LR)
            model = train(cfg.OPTIMIZATION, model, optimizer, labeled_dataloader, tb_log=tb_log)
            acc_list = test_classes(model, test_dataloader, tb_log=tb_log)
            for i in range(len(acc_list)):
                tb_log.add_scalars('{}_acc'.format(name[i]), {al_type: acc_list[i]}, global_step=step)


if __name__ == '__main__':
    main()
