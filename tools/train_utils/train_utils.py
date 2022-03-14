"""
训练逻辑
"""
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm

from src.dataset import collate_fn, SubsetSequentialSampler, SubsetRandomSampler

device = torch.device('cuda')


def meta_train(cfg, model, optimizer, meta_dataloader):
    model = model.to(device)
    model.train()
    with tqdm.trange(cfg.META_TRAIN_EPOCHS, desc='meta train epochs', dynamic_ncols=True) as tbar:
        for cur_epoch in tbar:
            meta_train_one_epoch(cfg=cfg, model=model, optimizer=optimizer, meta_dataloader=meta_dataloader)
    return model


def meta_train_one_epoch(cfg, model, optimizer, meta_dataloader):
    pbar = tqdm.trange(len(meta_dataloader), desc='meta train iter', dynamic_ncols=True)
    for step, (x_spt, x_qry, y_spt, y_qry) in enumerate(meta_dataloader):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        task_num = x_spt.size(0)
        losses_q = 0
        for cur_task in range(task_num):
            logits = model(x_spt[cur_task], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[cur_task])
            # 利用support set计算网络的梯度
            grad = torch.autograd.grad(loss, model.parameters())
            fast_weights = list(map(lambda p: p[1] - cfg.UPDATE_LR * p[0], zip(grad, model.parameters())))
            for s in range(1, cfg.UPDATE_STEP):
                logits = model(x_spt[cur_task], vars=fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[cur_task])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - cfg.UPDATE_LR * p[0], zip(grad, fast_weights)))

            # 验证一下
            logits_q = model(x_qry[cur_task], fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry[cur_task], reduction='none')
            soft_logits_q = F.softmax(logits_q, dim=1)
            log_soft_logits_q = -torch.log(soft_logits_q)
            uncertainty = soft_logits_q.mul(log_soft_logits_q).sum(axis=1)
            soft_uncertainty = F.softmax(uncertainty, dim=0)
            soft_loss_q = F.softmax(loss_q, dim=0)
            log_soft_uncertainty = -torch.log(soft_uncertainty)
            uncertainty_loss_loss = soft_loss_q.mul(log_soft_uncertainty).sum()

            # 这里的loss有三种选择，一种是和前面一样，另一种是loss和uncertainty的loss，最后一种是混合的loss
            losses_q = losses_q + uncertainty_loss_loss + loss_q.sum() / x_qry[cur_task].size(0)

        pbar.update()
        pbar.set_postfix(losses_q=losses_q)

        optimizer.zero_grad()
        losses_q.backward()
        optimizer.step()
    pbar.close()


def train(cfg, model, optimizer, labeled_dataloader, tb_log=None):
    model.to(device)
    with tqdm.trange(cfg.EPOCHS, desc='train epochs', dynamic_ncols=True) as tbar:
        for i in tbar:
            pbar = tqdm.trange(len(labeled_dataloader), desc='train iter', dynamic_ncols=True)
            for step, (x, y) in enumerate(labeled_dataloader):
                x, y = x.to(device), y.to(device)
                logits = model(x, vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y)

                pbar.update()
                pbar.set_postfix(loss=loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return model


def meta_active_train(cfg, model, optimizer, meta_dataset, meta_dataloader,
                      train_dataset, uncertainty_type, init_labeled_index, active_train_nums,
                      logger, result_dir):
    logger.info("*********************Start meta active train {}****************************".format(uncertainty_type))
    labeled_index = init_labeled_index.copy()
    labeled_index_list = [labeled_index.copy()]
    logger.info("The init labeled dataset size: %d", len(init_labeled_index))
    result_dir = result_dir / 'meta_al_{}'.format(uncertainty_type)
    result_dir.mkdir(parents=True, exist_ok=True)
    for step in range(active_train_nums):
        logger.info("Start the %dth meta active train {} loop".format(uncertainty_type), step + 1)
        meta_dataset.set_labeled_index(labeled_index)
        unlabeled_index = meta_dataset.get_unlabeled_index()
        model = meta_train(cfg, model, optimizer, meta_dataloader)
        # 计算uncertainty，挑选数据
        unlabeled_dataloader = DataLoader(train_dataset,
                                          batch_size=cfg.BATCH_SIZE,
                                          sampler=SubsetSequentialSampler(unlabeled_index),
                                          collate_fn=collate_fn)
        if uncertainty_type == 'entropy':
            labeled_index, _, picked_index = entropy_uncertainty(model,
                                                                 unlabeled_dataloader,
                                                                 labeled_index,
                                                                 unlabeled_index,
                                                                 pick_nums=cfg.PICK_NUMS)
        elif uncertainty_type == 'margin':
            labeled_index, _, picked_index = margin_uncertainty(model,
                                                                unlabeled_dataloader,
                                                                labeled_index,
                                                                unlabeled_index,
                                                                pick_nums=cfg.PICK_NUMS)
        else:
            raise RuntimeError('there has no uncertainty type {}'.format(uncertainty_type))
        labeled_index_list.append(labeled_index.copy())
        with open(result_dir / "labeled_index_{}.txt".format(step + 1), 'w') as f:
            for index in labeled_index:
                f.write(str(index) + '\n')
        with open(result_dir / "picked_index_{}.txt".format(step + 1), 'w') as f:
            for index in picked_index:
                f.write(str(index) + '\n')
    logger.info("*********************meta active train {} done****************************".format(uncertainty_type))
    return labeled_index_list


def active_train(cfg, model, optimizer, train_dataset, uncertainty_type,
                 init_labeled_index, active_train_nums, logger, result_dir):
    logger.info("*********************Start active train {}****************************".format(uncertainty_type))
    labeled_index = init_labeled_index.copy()
    unlabeled_index = [x for x in list(range(len(train_dataset))) if x not in labeled_index]
    labeled_index_list = [labeled_index.copy()]
    logger.info("The init labeled dataset size: %d", len(init_labeled_index))
    result_dir = result_dir / 'al_{}'.format(uncertainty_type)
    result_dir.mkdir(parents=True, exist_ok=True)
    for step in range(active_train_nums):
        logger.info("Start the %dth active train {} loop".format(uncertainty_type), step + 1)
        labeled_dataloader = DataLoader(train_dataset,
                                        batch_size=cfg.BATCH_SIZE,
                                        sampler=SubsetRandomSampler(labeled_index),
                                        collate_fn=collate_fn)
        unlabeled_dataloader = DataLoader(train_dataset,
                                          batch_size=cfg.BATCH_SIZE,
                                          sampler=SubsetSequentialSampler(unlabeled_index),
                                          collate_fn=collate_fn)
        model = train(cfg, model, optimizer, labeled_dataloader)
        # 计算uncertainty，挑选数据
        if uncertainty_type == 'entropy':
            labeled_index, unlabeled_index, picked_index = entropy_uncertainty(model,
                                                                               unlabeled_dataloader,
                                                                               labeled_index,
                                                                               unlabeled_index,
                                                                               pick_nums=cfg.PICK_NUMS)
        elif uncertainty_type == 'margin':
            labeled_index, unlabeled_index, picked_index = margin_uncertainty(model,
                                                                              unlabeled_dataloader,
                                                                              labeled_index,
                                                                              unlabeled_index,
                                                                              pick_nums=cfg.PICK_NUMS)
        else:
            raise RuntimeError('there has no uncertainty type {}'.format(uncertainty_type))

        labeled_index_list.append(labeled_index.copy())
        with open(result_dir / "labeled_index_{}.txt".format(step + 1), 'w') as f:
            for index in labeled_index:
                f.write(str(index) + '\n')
        with open(result_dir / "picked_index_{}.txt".format(step + 1), 'w') as f:
            for index in picked_index:
                f.write(str(index) + '\n')
    logger.info("********************* active train {} done****************************".format(uncertainty_type))
    return labeled_index_list


def entropy_uncertainty(model, unlabeled_dataloader, labeled_index, unlabeled_index, pick_nums):
    samples_uncertainty = []
    for i, (x, _) in enumerate(unlabeled_dataloader):
        x = x.to(device)
        with torch.no_grad():
            logits = model(x, vars=None, bn_training=False)
            soft_logits = F.softmax(logits, dim=1)
            log_soft_logits = -torch.log(soft_logits)
            uncertainty = soft_logits.mul(log_soft_logits).sum(axis=1).cpu().numpy().tolist()
        samples_uncertainty.extend(uncertainty)
    # 熵越大，uncertainty越大
    sorted_indices = np.argsort(-np.array(samples_uncertainty))
    picked_index = list(np.array(unlabeled_index)[sorted_indices[:pick_nums]])
    labeled_index += picked_index
    unlabeled_index = list(np.array(unlabeled_index)[sorted_indices[pick_nums:]])
    return labeled_index, unlabeled_index, picked_index


def margin_uncertainty(model, unlabeled_dataloader, labeled_index, unlabeled_index, pick_nums):
    samples_uncertainty = []
    for i, (x, _) in enumerate(unlabeled_dataloader):
        x = x.to(device)
        with torch.no_grad():
            logits = model(x, vars=None, bn_training=False)
            soft_logits = F.softmax(logits, dim=1).cpu()
            sort_soft_logits = np.sort(soft_logits, axis=1)
            margin = sort_soft_logits[:, -1:] - sort_soft_logits[:, -2:-1]
        samples_uncertainty.extend(margin)
        # margin越小，uncertainty越大
    samples_uncertainty = np.squeeze(samples_uncertainty).tolist()
    sorted_indices = np.argsort(np.array(samples_uncertainty))
    picked_index = list(np.array(unlabeled_index)[sorted_indices[:pick_nums]])
    labeled_index += picked_index
    unlabeled_index = list(np.array(unlabeled_index)[sorted_indices[pick_nums:]])
    return labeled_index, unlabeled_index, picked_index
