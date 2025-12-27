import os
import time
import argparse
import datetime
import pandas as pd
from tqdm import tqdm as tqdm
from logger import create_logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils import *
from dataset import CLAdapterDataset
from build_model import CLAdapter_CLIP_ViT
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-mode', type=str, required=True)
    parser.add_argument('--finetune-mode', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nbatch_log', type=int, default=500)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_fold', type=int, default=0)
    parser.add_argument('--test_fold', type=int, default=1)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, required=True)
    args, _ = parser.parse_known_args()
    config = config_from_name(args.config_name)
    return args, config

def train_epoch(cur_epoch, model, train_loader, optimizer, criterion, scaler, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    bar = tqdm(train_loader)
    steps = 0
    for (images, labels) in bar:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True).long()
        if cur_epoch<=args.warmup_epochs:
            lr = get_warm_up_lr(cur_epoch, steps, args.warmup_epochs, args.init_lr, len(bar))
            set_lr(optimizer, lr)
        else:
            lr = get_train_epoch_lr(cur_epoch, steps, args.epochs, args.init_lr, len(bar))
            set_lr(optimizer, lr)
        with torch.cuda.amp.autocast():
            preds = model(images)
            loss = criterion(preds, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        reduced_loss = reduce_tensor(loss.data)
        losses.update(reduced_loss, images.size(0))
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        if args.local_rank==0:
            bar.set_description('lr: %.6f, loss_cur: %.5f, loss_avg: %.5f' % (lr, losses.val, losses.avg))
        if batch_time.count%args.nbatch_log==0 and args.local_rank==0:
            mu = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info('epoch: %d, iter: [%d/%d] || lr: %.6f, memory_used: %.0fMB, loss_cur: %.5f, loss_avg: %.5f, \
                        time_avg: %.3f, time_total: %.3f' % (cur_epoch, batch_time.count, len(train_loader), lr, mu, losses.val, losses.avg, batch_time.avg, batch_time.sum))
        steps += 1
    return losses.avg

def val_epoch(model, valid_loader, criterion):
    model.eval()
    bar = tqdm(valid_loader)
    with torch.no_grad():
        preds = []
        labels = []
        for (image, label) in bar:
            image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            pred = model(image)
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        loss = criterion(preds, labels)
        acc = accuracy(preds, labels, topk=(1, ))[0]
        prec = precision(preds, labels)
        reca = recall(preds, labels)
        f1 = f1_score(prec, reca)
        reduced_loss = reduce_tensor(loss)
        reduced_acc = reduce_tensor(acc)
        reduced_prec = reduce_tensor(prec)
        reduced_reca = reduce_tensor(reca)
        reduced_f1 = reduce_tensor(f1)
    return reduced_loss, reduced_acc, reduced_prec, reduced_reca, reduced_f1


def test_epoch(model, test_loader, criterion):
    model.eval()
    bar = tqdm(test_loader)
    with torch.no_grad():
        preds = []
        labels = []
        for (image, label) in bar:
            image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True).long()
            pred = model(image)
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        loss = criterion(preds, labels)
        acc = accuracy(preds, labels, topk=(1, ))[0]
        prec = precision(preds, labels)
        reca = recall(preds, labels)
        f1 = f1_score(prec, reca)
    return loss.item(), acc.item(), prec.item(), reca.item(), f1.item()
    
def main(config):
    df = pd.read_csv(args.csv_dir)
    is_malignant = 'malignant' in args.csv_dir
    dataset_train = CLAdapterDataset(is_malignant, df, args.val_fold, args.test_fold, 'train', config.MODEL.img_size, config.data_root)
    dataset_valid = CLAdapterDataset(is_malignant, df, args.val_fold, args.test_fold, 'valid', config.MODEL.img_size, config.data_root)
    dataset_test = CLAdapterDataset(is_malignant, df, args.val_fold, args.test_fold, 'test', config.MODEL.img_size, config.data_root)
    train_sampler = DistributedSampler(dataset_train)
    valid_sampler = DistributedSampler(dataset_valid)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True, sampler=valid_sampler, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    model = CLAdapter_CLIP_ViT(config)
    if config.MODEL.finetune != None:
        load_ckpt_finetune(config.MODEL.finetune, model, logger=logger, args=args)
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None, find_unused_parameters=True) #find_unused_parameters=True
    optimizer = get_optim_from_config(config, model, 'embed')
    criterion = nn.CrossEntropyLoss().cuda()
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    best_acc = -1
    args.epochs += 1
    for epoch in range(1, args.epochs):
        if args.local_rank==0:
            logger.info(f"----------[Epoch {epoch}]----------")
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion, scaler, args)
        val_loss, acc, val_prec, val_reca, val_f1, val_roc = val_epoch(model, valid_loader, criterion, config.MODEL.num_classes) # ROC
        if args.local_rank==0:
            logger.info(f"epoch: {epoch} || loss_train: {train_loss:.5f}, loss_val: {val_loss:.5f}, val_acc: {acc:.5f}, val_prec: {val_prec:.5f}, val_reca: {val_reca:.5f}, val_f1: {val_f1:.5f}, val_roc: {val_roc:.5f}")
            if acc >= best_acc:
                best_acc = acc
                save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.backbone.model_name}_best.pth')
                logger.info(f"Save best model to {save_path}, with best top1: {best_acc}")
                save_checkpoint(model, save_path)
                test_loss, test_acc, test_prec, test_reca, test_f1, test_roc = test_epoch(model, test_loader, criterion, config.MODEL.num_classes) # ROC
                logger.info(f"epoch: {epoch} || loss_test: {test_loss:.5f}, test_acc: {test_acc:.5f}, test_prec: {test_prec:.5f}, test_reca: {test_reca:.5f}, test_f1: {test_f1:.5f}, test_roc: {test_roc:.5f}")
            logger.info(f'Epoch {epoch} time cost: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
    if args.local_rank==0:
        logger.info(f"Best val acc: {best_acc}")
        logger.info(f"loss_test: {test_loss:.5f}, test_acc: {test_acc:.5f}, test_prec: {test_prec:.5f}, test_reca: {test_reca:.5f}, test_f1: {test_f1:.5f}")

if __name__ == '__main__':
    args, config = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    config.defrost()
    if 'malignant' in args.csv_dir:
        config.MODEL.num_classes = 4
        config.MODEL.output_dir += '/malignant'
    elif 'cotton' in args.csv_dir:
        config.MODEL.num_classes = 80
        config.MODEL.output_dir += '/agricultural_cotton'
    elif 'soyloc' in args.csv_dir:
        config.MODEL.num_classes = 200
        config.MODEL.output_dir += '/agricultural_soyloc'
    elif 'plant' in args.csv_dir:
        config.MODEL.num_classes = 4
        config.MODEL.output_dir += '/agricultural_plant_pathology'
    elif 'WHU-RS19' in args.csv_dir:
        config.MODEL.num_classes = 19
        config.MODEL.output_dir += '/RemoteSensing_plant_WHU-RS19'
    elif 'glass-insulator' in args.csv_dir:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/Industry_DefectSupervised_glass-insulator'
    elif 'lightning-rod-suspension' in args.csv_dir:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/Industry_DefectSupervised_lightning-rod-suspension'
    elif 'polymer-insulator-upper-shackle' in args.csv_dir:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/Industry_DefectSupervised_polymer-insulator-upper-shackle'
    elif 'vari-grip' in args.csv_dir:
        config.MODEL.num_classes = 3
        config.MODEL.output_dir += '/Industry_DefectSupervised_vari-grip'
    elif 'yoke-suspension' in args.csv_dir:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/Industry_DefectSupervised_yoke-suspension'
    elif 'KTH-TIPS2-b' in args.csv_dir:
        config.MODEL.num_classes = 11
        config.MODEL.output_dir += '/Material_KTH-TIPS2-b'
    elif 'tiny-imagenet' in args.csv_dir:
        config.MODEL.num_classes = 200
        config.MODEL.output_dir += '/tiny-imagenet'
    elif 'PACS' in args.csv_dir:
        config.MODEL.num_classes = 7
        config.MODEL.output_dir += '/OOD_PACS'
    else:
        config.MODEL.num_classes = 2
        config.MODEL.output_dir += '/gastric'
    if args.model_mode == 'conv':
        config.MODEL.backbone.out_dim = 1024
        config.MODEL.backbone.num_patch = 49
    elif args.model_mode == 'res_xcep':
        config.MODEL.backbone.out_dim = 2048
        config.MODEL.backbone.num_patch = 49
    else:
        config.MODEL.backbone.out_dim = 768
        config.MODEL.backbone.num_patch = 196
        # config.MODEL.backbone.out_dim = 1024
        # config.MODEL.backbone.num_patch = 576
    config.MODEL.m_mode = args.model_mode
    config.MODEL.f_mode = args.finetune_mode
    config.MODEL.output_dir += '/'+args.model_mode
    config.MODEL.output_dir += '/'+args.finetune_mode
    if config.MODEL.finetune is not None:
        args.init_lr /= 10
        config.MODEL.output_dir += '/'+config.MODEL.backbone.model_name+'-unfreeze'
    else:
        config.MODEL.output_dir += '/'+config.MODEL.backbone.model_name
    config.MODEL.img_size = args.image_size
    config.init_lr = args.init_lr
    config.batch_size = args.batch_size
    config.local_rank = args.local_rank
    config.world_size = args.world_size
    config.data_root = args.data_root
    config.freeze()
    torch.cuda.set_device(args.local_rank + args.gpu_id)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    set_seed(config.SEED)
    os.makedirs(config.MODEL.output_dir, exist_ok=True)
    logger = create_logger(output_dir=config.MODEL.output_dir, dist_rank=args.local_rank, name=f"{config.MODEL.backbone.model_name}")
    if args.local_rank==0:
        logger.info(config.dump())
    main(config)

