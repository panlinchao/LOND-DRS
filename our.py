import os
import pathlib
import time
import datetime
import argparse
import shutil
from typing import List
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import faiss
import torchnet
from data.noisy_cifar import *
from utils.core import *
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger
from utils.loss import *
from utils.contrastive_loss import *
from utils.module import MLPHead
from utils.plotter import plot_results
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
LOG_FREQ = 1


class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w, x_s
    

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, low_dim=128, dropout_rate=0.25, momentum=0.1, activation='tanh'):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.ova_head = MLPHead(256, mlp_scale_factor=1, projection_size=2*n_outputs, init_method='He', activation=activation)
        self.proj_head = MLPHead(256, mlp_scale_factor=1, projection_size=low_dim, init_method='He', activation=activation)
        self.prototypes = nn.Parameter(torch.zeros(n_outputs, low_dim), requires_grad=False)

    def init_prototypes(self, prototypes):
        self.prototypes.data = prototypes
        self.prototypes.requires_grad = True

    def forward(self, x, with_proto=False, temp=1.0):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        features_proj = self.proj_head(x)
        logits_open = self.ova_head(x)
        features_proj_norm = F.normalize(features_proj, dim=1)  # L2 normalize
        logits_proto = None
        if with_proto:
            prototypes = F.normalize(self.prototypes, dim=1)
            logits_proto = features_proj_norm.matmul(prototypes.t()) / temp
        return {'logits': None, 'logits_open': logits_open, 'logits_proto': logits_proto, 'features': features_proj_norm}


## Input interpolation functions
def mix_data_lab(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, index, lam


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def record_network_arch(result_dir, net, rotnet_head=None):
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())
        if rotnet_head is not None:
            f.writelines(rotnet_head.__repr__())


def build_logger(params):
    logger_root = f'Results/{params.synthetic_data}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, noise_condition, params.project, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


def build_model_optim_scheduler(params, device, build_scheduler=True):
    assert params.dataset.startswith('cifar')
    n_classes = int(params.n_classes * (1 - params.openset_ratio))
    if params.net == 'CNN':
        net = CNN(input_channel=3, n_outputs=n_classes, low_dim=params.low_dim, activation='leaky relu' if params.activation == 'l_relu' else params.activation)
    else:
        raise AssertionError(f'{params.net} network is not supported yet.')
    if params.opt == 'sgd':
        optimizer = build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    elif params.opt == 'adam':
        optimizer = build_adam_optimizer(net.parameters(), params.lr)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet.')
    if build_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)
    else:
        scheduler = None
    return net.to(device), optimizer, scheduler, n_classes
    
    
def build_dataset_loader(params):
    assert params.dataset.startswith('cifar')
    transform = build_transform(rescale_size=params.rescale_size, crop_size=params.crop_size)
    if params.dataset == 'cifar100':
        dataset = build_cifar100n_dataset(os.path.join(params.database, params.dataset), CLDataTransform(transform['cifar_train'], transform['cifar_train_strong_aug']), transform['cifar_test'], noise_type=params.noise_type, openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)
    elif params.dataset == 'cifar10':
        dataset = build_cifar10n_dataset(os.path.join(params.database, params.dataset), CLDataTransform(transform['cifar10_train'], transform['cifar10_train_strong_aug']), transform['cifar10_test'], noise_type=params.noise_type, openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)
    else:
        raise AssertionError(f'{params.dataset} dataset is not supported yet.')
    train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    return dataset, train_loader, test_loader


def wrapup_training(result_dir, best_accuracy):
    stats = get_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def compute_train_stats(net, train_loader, test_loader, sel_stats, n_classes, device):
    net.eval()
    with torch.no_grad():
        transform_bak = train_loader.dataset.transform
        train_loader.dataset.transform = test_loader.dataset.transform
        temploader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=16, shuffle=False, num_workers=4)
        for batch_idx, sample in enumerate(temploader):
            inputs = sample['data']
            index = sample['index']
            inputs = inputs.to(device)
            outputs = net(inputs)
            features = outputs['features']
            logits_open = outputs['logits_open']
            probs_open = F.softmax(logits_open.view(-1, 2, n_classes), 1)
            sel_stats['features'][index] = features.detach().cpu()
            sel_stats['ovaProbs'][index] = probs_open.detach().cpu()
    train_loader.dataset.transform = transform_bak


def compute_prototypes(features, S_OVA_Filter, labels, n_classes, device):
    prototypes = torch.zeros(n_classes, features.size(-1)).to(device)
    for i in range(n_classes):
        idx_class = labels == i
        idx_class_sel = torch.logical_and(idx_class, S_OVA_Filter)
        prototypes[i] = features[idx_class_sel].mean(0)
    return prototypes


def selected_update(soft_labels: torch.Tensor, labels: torch.Tensor, n_classes: int, cfg) -> torch.Tensor.bool:
    prob_temp = soft_labels[torch.arange(0, soft_labels.size(0)), labels]
    prob_temp[prob_temp <= 1e-2] = 1e-2
    prob_temp[prob_temp > (1-1e-2)] = 1-1e-2
    # --- margin
    soft_labels_tmp = soft_labels.clone()
    soft_labels_tmp[torch.arange(0, soft_labels.size(0)), labels] = 0
    top_k = torch.topk(soft_labels_tmp, k=cfg.top_k, dim=1, largest=True, sorted=False)[0]
    top_k_mean = torch.mean(top_k, dim=1)
    discrepancy_measure = prob_temp - top_k_mean
    agreement_measure = (torch.max(soft_labels, dim=1)[1] == labels).float().data.cpu()  

    ## select examples 
    num_clean_per_class = torch.zeros(n_classes)  # number of prediction for each class
    for i in range(n_classes):
        idx_class = labels == i
        num_clean_per_class[i] = torch.sum(agreement_measure[idx_class])
    if cfg.alpha_id == 0.5:
        num_samples2select_class = torch.median(num_clean_per_class)
    elif cfg.alpha_id == 1.0:
        num_samples2select_class = torch.max(num_clean_per_class)
    elif cfg.alpha_id == 0.0:
        num_samples2select_class = torch.min(num_clean_per_class)
    else:
        num_samples2select_class = torch.quantile(num_clean_per_class, cfg.alpha_id)
    
    agreement_measure = torch.zeros((len(labels),))
    for i in range(n_classes):
        idx_class = labels == i
        samplesPerClass = idx_class.sum()
        idx_class = (idx_class.float() == 1.0).nonzero().squeeze()
        discrepancy_class = discrepancy_measure[idx_class]  

        if num_samples2select_class >= samplesPerClass:
            k_corrected = samplesPerClass
        else:
            k_corrected = num_samples2select_class

        top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=True, sorted=False)[1]

        agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0
    selected_examples = agreement_measure.bool()
    return selected_examples


def sample_identification(net, n_classes, train_dataset, net_prototypes, sel_stats, device, cfg):
    sel_stats['sel_clean'] = sel_stats['sel_clean'] & False
    sel_stats['sel_noisy'] = sel_stats['sel_noisy'] & False
    sel_stats['sel_ood'] = sel_stats['sel_ood'] & False
    sel_stats['weights'] = sel_stats['weights'] * 0.0
    labels_corrected = torch.LongTensor(train_dataset.noisy_labels)
    labels_given = torch.LongTensor(train_dataset.noisy_labels)
    N = labels_given.size(0)
    tmp_range = torch.arange(0, N).long()

    # -------------- from neighbor's view, aggrate the predict probs
    features_np = sel_stats['features'].cpu().numpy()
    index = faiss.IndexFlatIP(features_np.shape[1])
    index.add(features_np)
    D, I = index.search(features_np, cfg.n_neighbors)
    neighbors = torch.LongTensor(I)
    neighbors_weights = torch.exp(torch.Tensor(D) / cfg.temperature)
    probs_neigh_corrected = torch.zeros(N, n_classes)
    probOVA_Norm = sel_stats['ovaProbs'][tmp_range, 1, :] / sel_stats['ovaProbs'][tmp_range, 1, :].sum(1).unsqueeze(-1)
    for n in range(N):
        probs_neigh_corrected[n] = (probOVA_Norm[neighbors[n, :]]*neighbors_weights[n].unsqueeze(-1)).sum(0)
    probs_neigh_corrected = probs_neigh_corrected / probs_neigh_corrected.sum(1).unsqueeze(-1)
    sel_stats['sel_clean'] = selected_update(probs_neigh_corrected, labels_corrected, n_classes, cfg) 
    sel_stats['weights'][sel_stats['sel_clean']] = torch.ones(sel_stats['sel_clean'].sum())

    if not net_prototypes.requires_grad:
        tmp_prototypes = compute_prototypes(sel_stats['features'], sel_stats['sel_clean'], labels_corrected, n_classes, device)
        net.init_prototypes(tmp_prototypes)

    # --- filter OOD sample
    ood_scores = sel_stats['ovaProbs'][tmp_range, 0, :]
    ood_scores_tmp = ood_scores.clone()
    ood_scores_tmp[tmp_range, labels_corrected] = -1e6
    other_max = torch.max(ood_scores_tmp, dim=1)[0]
    margin = sel_stats['ovaProbs'][tmp_range, 0, labels_corrected] - other_max  # max as ood
    margin[sel_stats['sel_clean']] = -1
    _, margin_idx = torch.sort(torch.abs(margin))
    num_clean = torch.sum(sel_stats['sel_clean']).item()
    num_candidate = (N - num_clean) * cfg.alpha_ood
    num_ood_sel = int(num_candidate)
    ood_sel_idx = margin_idx[:num_ood_sel]
    sel_stats['sel_ood'][ood_sel_idx] = True
    sel_stats['weights'][sel_stats['sel_ood']] = 1.0 - torch.abs(margin[ood_sel_idx])

    # --- weighted for rest data
    pred_merged = torch.argmax(probs_neigh_corrected, dim=-1)
    probs_merged_tmp = probs_neigh_corrected.clone()
    probs_merged_tmp[tmp_range, labels_corrected] = -1e6
    other_max = torch.max(probs_merged_tmp, dim=1)[0]
    margin_merged = probs_neigh_corrected[tmp_range, labels_corrected] - other_max
    sel_stats['sel_noisy'] = torch.logical_not(torch.logical_or(sel_stats['sel_clean'], sel_stats['sel_ood']))
    weights_ulb = margin_merged[sel_stats['sel_noisy']] + 1  # make sure the weights are positive
    weights_ulb = weights_ulb / weights_ulb.max()  # normalize the weights
    sel_stats['weights'][sel_stats['sel_noisy']] = weights_ulb
    labels_corrected[sel_stats['sel_noisy']] = pred_merged[sel_stats['sel_noisy']]

    labels_gt = torch.LongTensor(train_dataset.targets)
    labels_gt[labels_gt >= n_classes] = n_classes
    labels_given = F.one_hot(labels_given, n_classes)
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    acc_meter.reset()
    acc_meter.add(labels_given[sel_stats['sel_clean']], labels_gt[sel_stats['sel_clean']])
    correction_accuracy_ova = acc_meter.value()[0]
    correction_accuracy_ood = (labels_gt[sel_stats['sel_ood']] >= n_classes).float().mean().item()
    sel_nums = {
        'num_selected/ova': torch.sum(sel_stats['sel_clean']).item(),
        'num_selected/ood': torch.sum(sel_stats['sel_ood']).item(),
        'correction_accuracy/ova': correction_accuracy_ova,
        'correction_accuracy/ood': correction_accuracy_ood,
    }
    return labels_corrected, sel_nums


def warmup_train(net, optimizer, loss_funcs, train_loader, n_classes, device, cfg):
    train_accuracy = AverageMeter()
    train_loss = AverageMeter()
    pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='Warmup')
    net.train()
    optimizer.zero_grad()
    for it, sample in enumerate(pbar):
        curr_lr = [group['lr'] for group in optimizer.param_groups][0]
        x, x_s = sample['data']
        x, x_s = x.to(device), x_s.to(device)
        y = sample['label'].to(device)

        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(x, y, cfg.alpha, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(x_s, y, cfg.alpha, device)
        # ------------------ warmup for Proj head
        output_w_mix = net(img1)
        output_s_mix = net(img2)
        feat_w_mix = output_w_mix['features']
        feat_s_mix = output_s_mix['features']
        feat_w_mix = feat_w_mix.unsqueeze(1)
        feat_s_mix = feat_s_mix.unsqueeze(1)
        mv_features_mix = torch.cat([feat_w_mix, feat_s_mix], dim=1)  # for orginal order
        max_probs = torch.ones_like(y, dtype=torch.float32, device=device)
        loss_supcl = loss_funcs['BCL'](mv_features_mix, max_probs, labels=y_a1, mix_index_weak=mix_index1, mix_index_strong=mix_index2, lam1=lam1, lam2=lam2, reduction='none')

        # ------------------ warmup for OVA head
        given_labels = F.one_hot(y, n_classes)
        y_a1, y_b1 = given_labels, given_labels[mix_index1]
        y_a2, y_b2 = given_labels, given_labels[mix_index2]
        logit_w = output_w_mix['logits_open']
        logit_s = output_s_mix['logits_open']
        logits = torch.cat((logit_w, logit_s), dim=0)
        targets_1 = torch.cat((y_a1, y_a2), dim=0)
        targets_2 = torch.cat((y_b1, y_b2), dim=0)
        ones_vec = torch.ones((logit_w.size()[0],)).float().to(device)
        lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device)
        loss_cls = lam_vec * loss_funcs['OVA'](logits, targets_1) + (1 - lam_vec) *  loss_funcs['OVA'](logits, targets_2)
        loss_cls = loss_cls.mean()

        train_acc = accuracy(logit_w.view(logit_w.size(0), 2, -1)[:, 1, :], y, topk=(1,))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy.avg:3.2f}%; TrainLoss: {train_loss.avg:3.2f}')
        pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')

        loss = loss_supcl + loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_accuracy.update(train_acc[0], x.size(0))
        train_loss.update(loss.item(), x.size(0))
    return {'train/accuracy': train_accuracy.avg, 'train/loss': train_loss.avg}
 

def build_sel_loader(sel_stats, train_dataset, cfg):
    clean_idx_ova: List[int] = torch.nonzero(sel_stats['sel_clean']).squeeze(1).tolist()
    noisy_idx_ova: List[int] = torch.nonzero(sel_stats['sel_noisy']).squeeze(1).tolist()
    id_idx_ova: List[int] = torch.nonzero(torch.logical_not(sel_stats['sel_ood']))

    train_ova_clean_dataset = torch.utils.data.Subset(train_dataset, clean_idx_ova)
    train_ova_noisy_dataset = torch.utils.data.Subset(train_dataset, noisy_idx_ova)
    train_id_dataset = torch.utils.data.Subset(train_dataset, id_idx_ova)

    train_ova_clean_loader = torch.utils.data.DataLoader(train_ova_clean_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    train_ova_noisy_loader = torch.utils.data.DataLoader(train_ova_noisy_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    train_id_loader = torch.utils.data.DataLoader(train_id_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    return train_ova_clean_loader, train_ova_noisy_loader, train_id_loader


def linear_rampup(lambda_u, current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u*float(current)


def robust_train(net, optimizer, loss_funcs, train_loader, train_clean_loader, train_noisy_loader, train_id_loader, labels_corrected, sel_stats, n_classes, device, epoch, cfg):
    # meters   
    train_accuracy = AverageMeter()
    train_loss = AverageMeter()
    net.train()
    optimizer.zero_grad()
    train_clean_loader_iter = iter(train_clean_loader)
    train_noisy_loader_iter = iter(train_noisy_loader)
    train_id_loader_iter = iter(train_id_loader)
    pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='Robust')
    for it, sample in enumerate(pbar):
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy.avg:3.2f}%; TrainLoss: {train_loss.avg:3.2f}')
        try:
            sample_clean = next(train_clean_loader_iter)
        except:
            train_clean_loader_iter = iter(train_clean_loader)
            sample_clean = next(train_clean_loader_iter)
        try:
            sample_noisy = next(train_noisy_loader_iter)
        except:
            train_noisy_loader_iter = iter(train_noisy_loader)
            sample_noisy = next(train_noisy_loader_iter)
        try:
            sample_id = next(train_id_loader_iter)
        except:
            train_id_loader_iter = iter(train_id_loader)
            sample_id = next(train_id_loader_iter)
        # ----------------------- Predict pseudo labels for unlabeled data or data with noisy labels, label refinement for clean data
        x_w_noisy_ova, x_s_noisy_ova = sample_noisy['data']
        x_w_noisy_ova, x_s_noisy_ova = x_w_noisy_ova.to(device), x_s_noisy_ova.to(device)
        indices = sample_noisy['index']
        weights_noisy_ova = sel_stats['weights'][indices].to(device)
        x_w_clean_ova, x_s_clean_ova = sample_clean['data']
        x_w_clean_ova, x_s_clean_ova = x_w_clean_ova.to(device), x_s_clean_ova.to(device)
        indices = sample_clean['index']
        y_clean_ova = labels_corrected[indices].to(device)
        y_clean_ova_1hot = F.one_hot(y_clean_ova, n_classes)
        with torch.no_grad():
            ## Label guessing
            outputs_noisy_ova_w = net(x_w_noisy_ova, with_proto=True, temp=cfg.temperature)
            outputs_noisy_ova_s = net(x_s_noisy_ova, with_proto=True, temp=cfg.temperature)
            # --- OVA
            logits_noisy_ova_w = outputs_noisy_ova_w['logits_open']  # size: (bsz, 2*C)
            logits_noisy_ova_s = outputs_noisy_ova_s['logits_open']
            logits_noisy_ova_w = logits_noisy_ova_w.view(logits_noisy_ova_w.size(0), 2, -1)
            logits_noisy_ova_s = logits_noisy_ova_s.view(logits_noisy_ova_s.size(0), 2, -1)
            probs_noisy_ova_w = F.softmax(logits_noisy_ova_w, dim=1)
            probs_noisy_ova_s = F.softmax(logits_noisy_ova_s, dim=1)
            tmp_range = torch.arange(probs_noisy_ova_w.size(0))
            probs_noisy_ova_id_w = probs_noisy_ova_w[tmp_range, 1, :]
            probs_noisy_ova_id_w = probs_noisy_ova_id_w / torch.sum(probs_noisy_ova_id_w, dim=1, keepdim=True)
            probs_noisy_ova_id_s = probs_noisy_ova_s[tmp_range, 1, :]
            probs_noisy_ova_id_s = probs_noisy_ova_id_s / torch.sum(probs_noisy_ova_id_s, dim=1, keepdim=True)
            # --- Proto
            logits_noisy_ova_proto_w = outputs_noisy_ova_w['logits_proto']
            logits_noisy_ova_proto_s = outputs_noisy_ova_s['logits_proto']
            probs_noisy_ova_proto_w = F.softmax(logits_noisy_ova_proto_w, dim=1)
            probs_noisy_ova_proto_s = F.softmax(logits_noisy_ova_proto_s, dim=1)
            # --- avg for 1*C label
            pu = (probs_noisy_ova_id_w + probs_noisy_ova_id_s + probs_noisy_ova_proto_w + probs_noisy_ova_proto_s) / 4
            weights_factor = weights_noisy_ova * 1.0 / cfg.T  # size: (bsz,)
            ptu = pu.pow(weights_factor.view(-1, 1))  # The smaller the weights, the more uniform the distribution.
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            ## Label refinement
            outputs_clean_ova_w = net(x_w_clean_ova, with_proto=True, temp=cfg.temperature)
            outputs_clean_ova_s = net(x_s_clean_ova, with_proto=True, temp=cfg.temperature)
            logits_clean_ova_w = outputs_clean_ova_w['logits_open']
            logits_clean_ova_s = outputs_clean_ova_s['logits_open']
            logits_clean_ova_w = logits_clean_ova_w.view(logits_clean_ova_w.size(0), 2, -1)
            logits_clean_ova_s = logits_clean_ova_s.view(logits_clean_ova_s.size(0), 2, -1)
            probs_clean_ova_w = F.softmax(logits_clean_ova_w, dim=1)
            probs_clean_ova_s = F.softmax(logits_clean_ova_s, dim=1)
            tmp_range = torch.arange(probs_clean_ova_w.size(0))
            probs_clean_ova_id_w = probs_clean_ova_w[tmp_range, 1, :]
            weight_w, _ = torch.max(probs_clean_ova_id_w, dim=1)
            probs_clean_ova_id_s = probs_clean_ova_s[tmp_range, 1, :]
            weight_s, _ = torch.max(probs_clean_ova_id_s, dim=1)
            weight = (weight_w + weight_s) / 2
            weight = weight.view(-1, 1).type(torch.FloatTensor).cuda()
            logits_clean_ova_proto_w = outputs_clean_ova_w['logits_proto']
            logits_clean_ova_proto_s = outputs_clean_ova_s['logits_proto']
            probs_clean_ova_proto_w = F.softmax(logits_clean_ova_proto_w, dim=1)
            probs_clean_ova_proto_s = F.softmax(logits_clean_ova_proto_s, dim=1)
            px = (probs_clean_ova_proto_w + probs_clean_ova_proto_s) / 2
            px = weight * y_clean_ova_1hot + (1 - weight) * px
            ptx = px ** (1 / cfg.T)
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()
        # --- mix C labels
        id_inputs_w = torch.cat([x_w_clean_ova, x_w_noisy_ova], dim=0)
        id_inputs_s = torch.cat([x_s_clean_ova, x_s_noisy_ova], dim=0)
        id_targets = torch.cat([targets_x, targets_u], dim=0)
        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(id_inputs_w, id_targets, cfg.alpha, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(id_inputs_s, id_targets, cfg.alpha, device)

        output1 = net(img1, with_proto=True, temp=cfg.temperature)
        output2 = net(img2, with_proto=True, temp=cfg.temperature)

        # --- class-loss (OVA)
        logit_w = output1['logits_open']
        logit_s = output2['logits_open']
        # ------ for clean data
        preds = torch.cat((logit_w[:cfg.batch_size], logit_s[:cfg.batch_size]), dim=0)
        targets_1 = torch.cat((y_a1[:cfg.batch_size], y_a2[:cfg.batch_size]), dim=0)
        targets_2 = torch.cat((y_b1[:cfg.batch_size], y_b2[:cfg.batch_size]), dim=0)
        ones_vec = torch.ones((cfg.batch_size,)).float().to(device)
        lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device)
        loss_ova_lbl = lam_vec * loss_funcs['OVA'](preds, targets_1) + (1 - lam_vec) *  loss_funcs['OVA'](preds, targets_2)
        loss_ova_lbl = loss_ova_lbl.mean()

        # --- class-loss (Proto)
        logits_w = output1['logits_proto']
        logits_s = output2['logits_proto']
        probs_w = F.softmax(logits_w, dim=1)
        probs_s = F.softmax(logits_s, dim=1)
        # ------ for clean data
        logits_clean_id = torch.cat((logits_w[:cfg.batch_size], logits_s[:cfg.batch_size]), dim=0)
        loss_proto_lbl = -(
            lam_vec * torch.sum(F.log_softmax(logits_clean_id, dim=1) * targets_1, dim=1) +
            (1 - lam_vec) * torch.sum(F.log_softmax(logits_clean_id, dim=1) * targets_2, dim=1)
        )
        loss_proto_lbl = loss_proto_lbl.mean()
        # ------ for unlabeled / noisy data
        probs_ulb_id = torch.cat((probs_w[cfg.batch_size:], probs_s[cfg.batch_size:]), dim=0)
        targets_1 = torch.cat((y_a1[cfg.batch_size:], y_a2[cfg.batch_size:]), dim=0)
        targets_2 = torch.cat((y_b1[cfg.batch_size:], y_b2[cfg.batch_size:]), dim=0)
        loss_proto_ulb = (
            lam_vec * torch.sum(torch.abs((probs_ulb_id - targets_1) ** 2), dim=1) +
            (1 - lam_vec) * torch.sum(torch.abs((probs_ulb_id - targets_2) ** 2), dim=1)
        )
        loss_proto_ulb = loss_proto_ulb.mean()

        # --- BCL
        x_train, x_s_train = sample['data']
        x_train, x_s_train = x_train.to(device), x_s_train.to(device)         
        labels_hard_train = labels_corrected[sample['index']].to(device)
        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(x_train, labels_hard_train, cfg.alpha, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(x_s_train, labels_hard_train, cfg.alpha, device)
        # get mixed embeddings
        feat_w_mix = net(img1)['features'].unsqueeze(1)
        feat_s_mix = net(img2)['features'].unsqueeze(1)
        mv_features_mix = torch.cat([feat_w_mix, feat_s_mix], dim=1)  # for orginal order
        max_probs = sel_stats['weights'][sample['index']].to(device)
        select_vec = 1.0 - sel_stats['sel_ood'][sample['index']].to(device).float()
        loss_bcl = loss_funcs['BCL'](mv_features_mix, max_probs, labels=y_a1, mix_index_weak=mix_index1, mix_index_strong=mix_index2, lam1=lam1, lam2=lam2, reduction='mean', select_vec=select_vec)

        # ------------------------- OVA consistency loss
        x_train, x_s_train = sample_id['data']  
        x_train, x_s_train = x_train.to(device), x_s_train.to(device) 
        labels_hard_train = labels_corrected[sample_id['index']].to(device)
        logits_open_ori_w = net(x_train, with_proto=False)['logits_open']
        logits_open_ori_s = net(x_s_train, with_proto=False)['logits_open']
        logits_open_ori_w = logits_open_ori_w.view(logits_open_ori_w.size(0), 2, -1)
        logits_open_ori_s = logits_open_ori_s.view(logits_open_ori_s.size(0), 2, -1)
        logits_open_ori_w = F.softmax(logits_open_ori_w, 1)
        logits_open_ori_s = F.softmax(logits_open_ori_s, 1)
        loss_con = torch.mean(torch.sum(torch.sum(torch.abs(logits_open_ori_w - logits_open_ori_s)**2, 1), 1))
        probs_open_w = logits_open_ori_w.view(logits_open_ori_w.size(0), 2, -1)
        train_acc = accuracy(probs_open_w[:, 1, :], labels_hard_train, topk=(1,))

        lamb_proto_ulb = linear_rampup(1.0, epoch, cfg.warmup_epochs)
        loss = loss_ova_lbl + loss_proto_lbl + lamb_proto_ulb * loss_proto_ulb + cfg.lambda_bcl * loss_bcl + cfg.lambda_con * loss_con
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_accuracy.update(train_acc[0], x_train.size(0))
        train_loss.update(loss.item(), x_train.size(0))
    return {
        'train/accuracy': train_accuracy.avg, 
        'train/loss': train_loss.avg,
    }
    

def main(cfg, device):
    torch.set_float32_matmul_precision('high')
    init_seeds(cfg.seed)
    logger, result_dir = build_logger(cfg)
    net, optimizer, scheduler, n_classes = build_model_optim_scheduler(cfg, device, build_scheduler=False)
    lr_plan = build_lr_plan(cfg.lr, cfg.epochs, cfg.warmup_epochs, cfg.warmup_lr, decay=cfg.lr_decay, warmup_gradual=cfg.warmup_gradual)
    dataset, train_loader, test_loader = build_dataset_loader(cfg)
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu == -1:
                checkpoint = torch.load(cfg.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            net.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))


    # loss funcs
    loss_func_bcl = SoftSupMixupConLoss(temperature=cfg.temperature)
    loss_func_bcl = loss_func_bcl.to(device)
    loss_func_ova = SoftOVAOODLoss(reduction='none')
    loss_func_ova = loss_func_ova.to(device)
    loss_func_ova_all = OVAAllLoss()
    loss_func_ova_all = loss_func_ova_all.to(device)
    loss_funcs = {
        'BCL': loss_func_bcl,
        'OVA': loss_func_ova,
        'OVATe': loss_func_ova_all,
    }

    logger.msg(f"Categories: {n_classes}, Training Samples: {dataset['n_train_samples']}, Testing Samples: {dataset['n_test_samples']}")
    logger.msg(f"Noise Type: {dataset['train'].noise_type}, Openset Noise Ratio: {dataset['train'].openset_noise_ratio}, Closedset Noise Ratio: {dataset['train'].closeset_noise_rate}")
    logger.msg(f'Optimizer: {cfg.opt}')
    record_network_arch(result_dir, net)
    
    best_acc, best_epoch, is_best = 0, 0, False
    sel_stats = {
        'features': torch.zeros(dataset['n_train_samples'], cfg.low_dim),
        'ovaProbs': torch.zeros(dataset['n_train_samples'], 2, n_classes),
        'sel_clean': torch.zeros(dataset['n_train_samples']).bool(),
        'sel_noisy': torch.zeros(dataset['n_train_samples']).bool(),
        'sel_ood': torch.zeros(dataset['n_train_samples']).bool(),
        'weights': torch.zeros(dataset['n_train_samples']),
    }
    sel_nums = {
        'num_selected/ova': 0,
        'num_selected/ood': 0,
        'correction_accuracy/ova': 0,
        'correction_accuracy/ood': 0,
    }

    for epoch in range(cfg.start_epoch, cfg.epochs):
        start_time = time.time()
        is_best = False
        adjust_lr(optimizer, lr_plan[epoch])
        train_result = {'train/accuracy': 0, 'train/loss': 0}
        if epoch < cfg.warmup_epochs:
            train_result = warmup_train(net, optimizer, loss_funcs, train_loader, n_classes, device, cfg)
        else:
            compute_train_stats(net, train_loader, test_loader, sel_stats, n_classes, device)
            labels_corrected, sel_nums = sample_identification(net, n_classes, dataset['train'], net.prototypes, sel_stats, device, cfg)
            train_clean_loader, train_noisy_loader, train_id_loader = build_sel_loader(sel_stats, dataset['train'], cfg)
            train_result = robust_train(net, optimizer, loss_funcs, train_loader, train_clean_loader, train_noisy_loader, train_id_loader, labels_corrected, sel_stats, n_classes, device, epoch, cfg)
        if epoch >= cfg.warmup_epochs:
            train_result.update(sel_nums)
        eval_result = evaluate_ours(test_loader, loss_funcs['OVATe'], net, device)
        test_accuracy = eval_result['test/acc_closed_ova_top1']
        test_loss = eval_result['test/loss_ova']
        train_result.update(eval_result)
        logger.tb_log(train_result, epoch)
        if test_accuracy > best_acc:
            best_acc, best_epoch = test_accuracy, epoch + 1
            is_best = True
        # logging this epoch
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_result["train/loss"]:>6.4f} | '
                    f'train accuracy: {train_result["train/accuracy"]:>6.3f} | '
                    f'test loss: {test_loss:>6.4f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy: {best_acc:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt', layout='2x2')
        if cfg.save_model:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg.net,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint{}.pth.tar'.format(result_dir, epoch), best_file_name='{}/checkpoint_best.pth.tar'.format(result_dir))

    wrapup_training(result_dir, best_acc)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


def parse_args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of MDR')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--synthetic-data', type=str, default='cifar100nc')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default='0.8')
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--net', type=str, default='CNN')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--lr-decay', type=str, default='cosine')
    parser.add_argument('--warmup-lr', type=float, default=0.001)
    parser.add_argument('--warmup-gradual', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--warmup-lr-scale', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--T', type=float, default=0.5, help='sharpening temperature for probs')
    parser.add_argument('--alpha', type=float, default=4, help='hyper-parameter for Beta distribution')
    parser.add_argument('--n-neighbors', type=int, default=200, help='number of neighbors for KNN')
    parser.add_argument('--lambda-bcl', type=float, default=1, help='weight for BCL loss')
    parser.add_argument('--lambda-con', type=float, default=1, help='weight for Con loss')
    parser.add_argument('--top-k', type=int, default=2, help='top-k for sample selection')
    parser.add_argument('--alpha-id', type=float, default=0.9, help='fraction for clean sample selection')
    parser.add_argument('--alpha-ood', type=float, default=0.1, help='fraction for OOD sample selection')

    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--low-dim', type=int, default=128, help='projection dim')

    parser.add_argument('--start-epoch', default=0, type=int, 
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--log', type=str, default='MDR')
    parser.add_argument('--ablation', action='store_true')
    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    
    assert config.temperature <= 1 and config.temperature > 0, f'temperture for sharpening operation should be in (0, 1], but the currect value is {config.temperature}.'
    assert config.synthetic_data in ['cifar10nc', 'cifar100nc', 'cifar80no']
    assert config.noise_type in ['symmetric', 'asymmetric']
    config.openset_ratio = 0.0 if config.synthetic_data in ['cifar10nc', 'cifar100nc'] else 0.2
    if config.ablation:
        config.project = f'ablation/{config.project}'
    config.log_freq = LOG_FREQ
    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
