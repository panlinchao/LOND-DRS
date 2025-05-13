import torch
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from utils.meter import AverageMeter
import os


def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    """
    Computes the precision@k for the specified values of k in this mini-batch
    :param y_pred   : tensor, shape -> (batch_size, n_classes)
    :param y_actual : tensor, shape -> (batch_size)
    :param topk     : tuple
    :param return_tensor : bool, whether to return a tensor or a scalar
    :return:
        list, each element is a tensor with shape torch.Size([])
    """
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def accuracy_open(pred, target, topk=(1,), num_classes=5, return_tensor=False):
    """Computes the precision@k for the specified values of k,
    num_classes are the number of known classes.
    This function returns overall accuracy,
    accuracy to reject unknown samples,
    the size of unknown samples in this batch."""
    maxk = max(topk)
    batch_size = target.size(0)
    pred = pred.view(-1, 1)
    pred = pred.t()
    ind = (target == num_classes)
    unknown_size = len(ind)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if ind.sum() > 0:
        unk_corr = pred.eq(target).view(-1)[ind]
        acc = torch.sum(unk_corr).item() / unk_corr.size(0)
    else:
        acc = 0

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        # res.append(correct_k.mul_(100.0 / batch_size))
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res[0], acc, unknown_size


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)


def roc_id_ood(score_id, score_ood):
    id_all = np.r_[score_id, score_ood]
    Y_test = np.zeros(score_id.shape[0]+score_ood.shape[0])
    Y_test[score_id.shape[0]:] = 1
    return roc_auc_score(Y_test, id_all)


def evaluate(dataloader, model, dev, topk=(1,)):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)
            output = model(x)
            logits = output['logits']
            loss = torch.nn.functional.cross_entropy(logits, y)
            test_loss.update(loss.item(), x.size(0))
            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))
    return {'accuracy': test_accuracy.avg, 'loss': test_loss.avg}


def evaluate_softmax_ood(dataloader, model, dev, topk=(1,)):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)
            output = model(x)
            logits = output['logits']
            is_open = sample['is_open'].to(dev)
            targets_unk = is_open
            y[targets_unk] = int(logits.size(-1))
            targets_known = torch.logical_not(targets_unk)
            known_logits = logits[targets_known]
            known_targets = y[targets_known]
            loss = torch.nn.functional.cross_entropy(known_logits, known_targets)
            test_loss.update(loss.item(), x.size(0))
            acc = accuracy(known_logits, known_targets, topk)
            test_accuracy.update(acc[0], x.size(0))
            prob = torch.nn.functional.softmax(logits, dim=-1)
            known_score, pred_close = prob.max(1)
            if batch_idx == 0:
                known_all = known_score
                label_all = y
            else:
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, y], 0)
    roc_soft = compute_roc(-known_all.cpu().numpy(), label_all.cpu().numpy(), num_known=int(logits.size(-1)))
    return {'accuracy': test_accuracy.avg, 'loss': test_loss.avg, 'ROC Softmax': roc_soft}


def evaluate_with_oods(dataloader, loss_func, model, dev, topk=(1,), exp_cls=None):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc = AverageMeter()
    unk = AverageMeter()
    if exp_cls is not None:
        top1_exp = AverageMeter()
        top5_exp = AverageMeter()

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='OVA testing')):
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)
            output = model(x)
            # logits = output['logits']
            logits_open = output['logits_open']
            logits_open = logits_open.view(logits_open.size(0), 2, -1)
            prob_open = torch.nn.functional.softmax(logits_open, dim=1)
            prob_close = prob_open[:, 1, :]
            known_score, pred_close = prob_close.max(1)
            tmp_range = torch.arange(0, logits_open.size(0)).long().to(dev)
            unk_score = prob_open[tmp_range, 0, pred_close]
            is_open = sample['is_open'].to(dev)
            targets_unk = is_open
            y[targets_unk] = int(prob_open.size(-1))
            targets_known = torch.logical_not(targets_unk)
            if exp_cls is not None:
                targets_exp_known = torch.zeros_like(targets_known)
                for idx, yy in enumerate(y):
                    if yy in exp_cls:
                        targets_known[idx] = False
                        targets_exp_known[idx] = True
            known_pred = prob_close[targets_known]
            known_targets = y[targets_known]
            if len(known_pred) > 0:
                top1_known, top5_known = accuracy(known_pred, known_targets, topk=(1, 5), return_tensor=True)
                top1.update(top1_known.item(), len(known_pred))
                top5.update(top5_known.item(), len(known_pred))
            if exp_cls is not None:
                known_exp_pred = prob_close[targets_exp_known]
                known_exp_targets = y[targets_exp_known]
                if len(known_exp_pred) > 0:
                    top1_known_exp, top5_known_exp = accuracy(known_exp_pred, known_exp_targets, topk=(1, 5), return_tensor=True)
                    top1_exp.update(top1_known_exp.item(), len(known_exp_pred))
                    top5_exp.update(top5_known_exp.item(), len(known_exp_pred))
            ind_unk = unk_score > 0.5
            pred_close[ind_unk] = int(prob_open.size(-1))
            acc_all, unk_acc, size_unk = accuracy_open(pred_close, y, num_classes=int(prob_open.size(-1)), return_tensor=True)
            acc.update(acc_all.item(), x.shape[0])
            unk.update(unk_acc, size_unk)

            if batch_idx == 0:
                unk_all = unk_score
                known_all = known_score
                label_all = y
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, y], 0)

            # loss = torch.nn.functional.cross_entropy(logits_open[:, 1, :], y)
            loss = loss_func(output['logits_open'][targets_known], known_targets)
            test_loss.update(loss.item(), x.size(0))
            # acc = accuracy(logits_open[:, 1, :], y, topk)
            # test_accuracy.update(acc[0], x.size(0))
    # ROC calculation
    unk_all = unk_all.cpu().numpy()
    known_all = known_all.cpu().numpy()
    label_all = label_all.cpu().numpy()
    roc = compute_roc(unk_all, label_all, num_known=int(prob_open.size(-1)))
    roc_soft = compute_roc(-known_all, label_all, num_known=int(prob_open.size(-1)))
    ind_known = np.where(label_all < int(prob_open.size(-1)))[0]
    id_score = unk_all[ind_known]
    result = {
        'Test loss': test_loss.avg,
        'Closed acc': top1.avg,
        'Overall acc': acc.avg,
        'Unk acc': unk.avg,
        'ROC': roc,
        'ROC Softmax': roc_soft,
        'Id score': id_score
    }
    if exp_cls is not None:
        result['Closed acc Exp'] = top1_exp.avg
    return result


def open_detection_indexes(scores, close_samples, thresh=None):
    """
    :param scores: np.array, shape -> (n_samples,), larger score means more likely to be closed
    :param close_samples: np.array, shape -> (n_samples,), 1 for closed, 0 for open
    :param thresh: float, threshold for open detection
    """
    if np.isnan(scores).any() or np.isinf(scores).any():
        return {"auroc" : -1}
    fpr, tpr, thresholds = metrics.roc_curve(close_samples, scores) 
    auroc = metrics.auc(fpr,tpr)
    precision, recall, _ = metrics.precision_recall_curve(close_samples, scores)
    aupr_in = metrics.auc(recall,precision)
    precision, recall, _ = metrics.precision_recall_curve(np.bitwise_not(close_samples), -scores)
    aupr_out = metrics.auc(recall,precision)

    det_acc = .5 * (tpr + 1.-fpr).max()
    tidx = np.abs(np.array(tpr) - 0.95).argmin()
    if thresh is None:    
        thresh = thresholds[tidx]
    predicts = scores >= thresh
    ys = close_samples
    accuracy = metrics.accuracy_score(ys,predicts)
    f1 = metrics.f1_score(ys,predicts)
    recall = metrics.recall_score(ys,predicts)
    precision = metrics.precision_score(ys,predicts)
    fpr_at_tpr95 = fpr[tidx]
    return {
        "auroc" : auroc,
        "auprIN" : aupr_in,
        "auprOUT" : aupr_out,
        "accuracy" : accuracy,
        'f1' : f1,
        'recall' : recall,
        'precision' : precision,
        "fpr@tpr95" : fpr_at_tpr95,
        "bestdetacc" : det_acc,
    }


def evaluate_with_oods_proto(dataloader, loss_func, model, dev, cfg, test_proto=False, result_dir=None, epoch=0, topk=(1,)):
    """
    Proto:
        Accuracy: top1 accuracy on known classes, top1 accuracy on unknown classes, overall accuracy
        test_loss: loss on known classes
    OVA:
        Accuracy: top1 accuracy on known classes, top1 accuracy on unknown classes, overall accuracy
        test_loss: loss on known classes
    """
    model.eval()
    test_loss_ova = AverageMeter()
    test_loss_proto = AverageMeter()
    # Close performance metric
    top1_ova = AverageMeter()
    top5_ova = AverageMeter()
    top1_proto = AverageMeter()
    top5_proto = AverageMeter()
    # Open performance metric
    acc_overall = AverageMeter()
    acc_unk = AverageMeter()

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='Testing')):
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)
            if test_proto:
                output = model(x, with_proto=True, temp=cfg.temperature)
                logits_proto = output['logits_proto']
                logits_open = output['logits_open']
                logits_open = logits_open.view(logits_open.size(0), 2, -1)
                prob_proto = torch.nn.functional.softmax(logits_proto, dim=1)
                prob_open = torch.nn.functional.softmax(logits_open, dim=1)
            else:
                output = model(x)
                logits_open = output['logits_open']
                logits_open = logits_open.view(logits_open.size(0), 2, -1)
                tmp_range = torch.arange(0, logits_open.size(0)).long().to(dev)
                logits_proto = logits_open[tmp_range, 1, :]
                prob_proto = torch.nn.functional.softmax(logits_proto, dim=1)
                prob_open = torch.nn.functional.softmax(logits_open, dim=1)
            # logits_proto = output['logits_proto']
            # logits_open = output['logits_open']
            # logits_open = logits_open.view(logits_open.size(0), 2, -1)
            # prob_proto = torch.nn.functional.softmax(logits_proto, dim=1)
            # prob_open = torch.nn.functional.softmax(logits_open, dim=1)

            prob_close_proto = prob_proto
            known_score_proto, pred_close_proto = prob_close_proto.max(1)

            prob_close_ova = prob_open[:, 1, :]
            known_score_ova, pred_close_ova = prob_close_ova.max(1)

            tmp_range = torch.arange(0, logits_open.size(0)).long().to(dev)
            
            unk_score_ova = prob_open[tmp_range, 0, pred_close_ova]
            unk_score_protoOva = prob_open[tmp_range, 0, pred_close_proto]
            
            is_open = sample['is_open'].to(dev)
            targets_unk = is_open
            y[targets_unk] = int(prob_open.size(-1))  # set the label of unknown samples to the number of classes
            targets_known = torch.logical_not(targets_unk)
            
            prob_known_proto = prob_close_proto[targets_known]
            prob_known_ova = prob_close_ova[targets_known]
            known_targets = y[targets_known]
            if len(prob_known_proto) > 0:
                top1_known, top5_known = accuracy(prob_known_proto, known_targets, topk=(1, 5), return_tensor=True)
                top1_proto.update(top1_known.item(), len(prob_known_proto))
                top5_proto.update(top5_known.item(), len(prob_known_proto))
            
            if len(prob_known_ova) > 0:
                top1_known, top5_known = accuracy(prob_known_ova, known_targets, topk=(1, 5), return_tensor=True)
                top1_ova.update(top1_known.item(), len(prob_known_ova))
                top5_ova.update(top5_known.item(), len(prob_known_ova))
            
            ind_unk = unk_score_protoOva > 0.5
            pred_close_proto[ind_unk] = int(prob_open.size(-1))
            cur_acc_all, cur_acc_unk, size_unk = accuracy_open(pred_close_proto, y, num_classes=int(prob_open.size(-1)), return_tensor=True)
            acc_overall.update(cur_acc_all.item(), x.shape[0])
            acc_unk.update(cur_acc_unk, size_unk)

            if batch_idx == 0:
                logits_open_all = logits_open.detach().cpu()
                logits_proto_all = logits_proto.detach().cpu()
                prob_close_ova_all = prob_close_ova
                prob_close_proto_all = prob_close_proto
                score_protoOva = unk_score_protoOva
                score_ova = unk_score_ova
                score_msp = known_score_proto
                label_all = y
            else:
                logits_open_all = torch.cat([logits_open_all, logits_open.detach().cpu()], 0)
                logits_proto_all = torch.cat([logits_proto_all, logits_proto.detach().cpu()], 0)
                prob_close_ova_all = torch.cat([prob_close_ova_all, prob_close_ova], 0)
                prob_close_proto_all = torch.cat([prob_close_proto_all, prob_close_proto], 0)
                score_protoOva = torch.cat([score_protoOva, unk_score_protoOva], 0)
                score_ova = torch.cat([score_ova, unk_score_ova])
                score_msp = torch.cat([score_msp, known_score_proto], 0)
                label_all = torch.cat([label_all, y], 0)

            # loss = torch.nn.functional.cross_entropy(logits_open[:, 1, :], y)
            # loss_ova = loss_func(output['logits_open'][targets_known], known_targets)
            # test_loss_ova.update(loss_ova.item(), targets_known.sum().item())
            # loss_proto = torch.nn.functional.cross_entropy(logits_proto[targets_known], known_targets)
            # test_loss_proto.update(loss_proto.item(), targets_known.sum().item())
            # acc = accuracy(logits_open[:, 1, :], y, topk)
            # test_accuracy.update(acc[0], x.size(0))
    
    logits_proto_all = logits_proto_all.cpu().numpy()
    logits_open_all = logits_open_all.cpu().numpy()
    prob_close_ova_all = prob_close_ova_all.cpu().numpy()
    prob_close_proto_all = prob_close_proto_all.cpu().numpy()
    score_protoOva = score_protoOva.cpu().numpy()
    score_ova = score_ova.cpu().numpy()
    score_msp = score_msp.cpu().numpy()
    label_all = label_all.cpu().numpy()
    close_samples = np.ones_like(label_all, dtype=np.bool_)
    is_ood = label_all >= int(prob_open.size(-1))
    close_samples[is_ood] = False
    if result_dir is not None:
        stats = {
            "logits_proto": logits_proto_all,
            "logits_open": logits_open_all,
            "labels": label_all,
        }
        if not os.path.exists(os.path.join(result_dir, 'Stats')):
            os.makedirs(os.path.join(result_dir, 'Stats'))
        torch.save(stats, os.path.join(result_dir, 'Stats', f'test_stats_{epoch}.pth'))

    # # --- correct & incorrect
    # pred_close_ova_all = np.argmax(prob_close_ova_all, axis=1)
    # pred_close_proto_all = np.argmax(prob_close_proto_all, axis=1)
    # iid_correct_ova = np.zeros_like(label_all, dtype=np.bool_)
    # iid_correct_ova[~is_ood] = pred_close_ova_all[~is_ood] == label_all[~is_ood]
    # iid_incorrect_ova = np.zeros_like(label_all, dtype=np.bool_)
    # iid_incorrect_ova[~is_ood] = pred_close_ova_all[~is_ood] != label_all[~is_ood]
    # iid_correct_proto = np.zeros_like(label_all, dtype=np.bool_)
    # iid_correct_proto[~is_ood] = pred_close_proto_all[~is_ood] == label_all[~is_ood]
    # iid_incorrect_proto = np.zeros_like(label_all, dtype=np.bool_)
    # iid_incorrect_proto[~is_ood] = pred_close_proto_all[~is_ood] != label_all[~is_ood]

    score_protoOva = 1.0 - score_protoOva
    score_ova = 1.0 - score_ova
    open_detection_protoOva = open_detection_indexes(score_protoOva, close_samples)
    open_detection_protoOva_thresh = open_detection_indexes(score_protoOva, close_samples, thresh=0.5)
    open_detection_ova = open_detection_indexes(score_ova, close_samples)
    open_detection_ova_thresh = open_detection_indexes(score_ova, close_samples, thresh=0.5)
    open_detection_msp = open_detection_indexes(score_msp, close_samples)
    if test_proto:
        result = {
            'test/loss_ova': test_loss_ova.avg,
            'test/loss_proto': test_loss_proto.avg,
            'test/acc_closed_ova_top1': top1_ova.avg,
            'test/acc_closed_ova_top5': top5_ova.avg,
            'test/acc_closed_proto': top1_proto.avg,
            'test/acc_closed_proto_top5': top5_proto.avg,
            'test/acc_ovarall_0.5thre': acc_overall.avg,
            'test/acc_unknown_0.5thre': acc_unk.avg,
            'test/open_auroc_protoOVA': open_detection_protoOva["auroc"],
            'test/open_auroc_ova': open_detection_ova["auroc"],
            'test/open_auroc_msp': open_detection_msp["auroc"],
            'test/open_f1_protoOVA': open_detection_protoOva["f1"],
            'test/open_f1_ova': open_detection_ova["f1"],
            'test/open_f1_msp': open_detection_msp["f1"],
            'test/open_f1_protoOVA_0.5thre': open_detection_protoOva_thresh["f1"],
            'test/open_f1_ova_0.5thre': open_detection_ova_thresh["f1"],
            'test/open_fpr95_protoOVA': open_detection_protoOva["fpr@tpr95"],
            'test/open_fpr95_ova': open_detection_ova["fpr@tpr95"],
            'test/open_fpr95_msp': open_detection_msp["fpr@tpr95"],
        }
    else:
        result = {
            'test/loss_ova': test_loss_ova.avg,
            'test/loss_proto': 0,
            'test/acc_closed_ova_top1': top1_ova.avg,
            'test/acc_closed_ova_top5': top5_ova.avg,
            'test/acc_closed_proto': 0,
            'test/acc_closed_proto_top5': 0,
            'test/acc_ovarall_0.5thre': acc_overall.avg,
            'test/acc_unknown_0.5thre': acc_unk.avg,
            'test/open_auroc_protoOVA': open_detection_protoOva["auroc"],
            'test/open_auroc_ova': open_detection_ova["auroc"],
            'test/open_auroc_msp': open_detection_msp["auroc"],
            'test/open_f1_protoOVA': open_detection_protoOva["f1"],
            'test/open_f1_ova': open_detection_ova["f1"],
            'test/open_f1_msp': open_detection_msp["f1"],
            'test/open_f1_protoOVA_0.5thre': open_detection_protoOva_thresh["f1"],
            'test/open_f1_ova_0.5thre': open_detection_ova_thresh["f1"],
            'test/open_fpr95_protoOVA': open_detection_protoOva["fpr@tpr95"],
            'test/open_fpr95_ova': open_detection_ova["fpr@tpr95"],
            'test/open_fpr95_msp': open_detection_msp["fpr@tpr95"],
        }
    return result


def evaluate_ours(dataloader, loss_func, model, dev):
    """
    OVA:
        Accuracy: top1 accuracy on known classes, top1 accuracy on unknown classes, overall accuracy
        test_loss: loss on known classes
    """
    model.eval()
    test_loss_ova = AverageMeter()
    top1_ova = AverageMeter()
    top5_ova = AverageMeter()


    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='Testing')):
            x = sample['data'].to(dev)
            y = sample['label'].to(dev)

            output = model(x)
            logits_open = output['logits_open']
            logits_open = logits_open.view(logits_open.size(0), 2, -1)
            prob_open = torch.nn.functional.softmax(logits_open, dim=1)

            prob_close_ova = prob_open[:, 1, :]
            
            is_open = sample['is_open'].to(dev)
            targets_unk = is_open
            y[targets_unk] = int(prob_open.size(-1))  # set the label of unknown samples to the number of classes
            targets_known = torch.logical_not(targets_unk)
            
            prob_known_ova = prob_close_ova[targets_known]
            known_targets = y[targets_known]
            
            if len(prob_known_ova) > 0:
                top1_known, top5_known = accuracy(prob_known_ova, known_targets, topk=(1, 5), return_tensor=True)
                top1_ova.update(top1_known.item(), len(prob_known_ova))
                top5_ova.update(top5_known.item(), len(prob_known_ova))

            loss_ova = loss_func(output['logits_open'][targets_known], known_targets)
            test_loss_ova.update(loss_ova.item(), targets_known.sum().item())

    result = {
        'test/loss_ova': test_loss_ova.avg,
        'test/acc_closed_ova_top1': top1_ova.avg,
        'test/acc_closed_ova_top5': top5_ova.avg,
    }
    return result