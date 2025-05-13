# This loss is adopted from https://github.com/TencentYoutuResearch/Classification-SemiCLS/blob/main/loss/soft_supconloss.py
""" The Code is under Tencent Youtu Public Rule
Part of the code is adopted form SupContrast as in the comment in the class
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SoftSupMixupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SoftSupMixupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, max_probs, labels=None, mix_index_weak=None, mix_index_strong=None, lam1=None, lam2=None, mask=None, reduction="mean", select_vec=None, select_matrix=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None and select_vec is not None:
            # --- Supervised contrastive learning
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            labels_major = labels
            labels_minor_w = labels[mix_index_weak]
            labels_minor_s = labels[mix_index_strong]
            labels_major = labels_major.contiguous().view(-1, 1)
            labels_minor_w = labels_minor_w.contiguous().view(-1, 1)
            labels_minor_s = labels_minor_s.contiguous().view(-1, 1)
            mask_major = torch.eq(labels_major, labels_major.T).float().to(device)
            mask_minor_w = torch.eq(labels_minor_w, labels_major.T).float().to(device)  # can't contrast with the other views, thus need a unsup term
            mask_minor_s = torch.eq(labels_minor_s, labels_major.T).float().to(device)
            #max_probs = max_probs.reshape((batch_size,1))
            max_probs_major = max_probs
            max_probs_minor_w = max_probs[mix_index_weak]
            max_probs_minor_s = max_probs[mix_index_strong]
            max_probs_major = max_probs_major.contiguous().view(-1, 1)
            max_probs_minor_w = max_probs_minor_w.contiguous().view(-1, 1)
            max_probs_minor_s = max_probs_minor_s.contiguous().view(-1, 1)
            score_mask_major = torch.matmul(max_probs_major, max_probs_major.T)  # --- foreground weighting
            score_mask_minor_w = torch.matmul(max_probs_minor_w, max_probs_major.T)
            score_mask_minor_s = torch.matmul(max_probs_minor_s, max_probs_major.T)
            # Set diagional to 1 to be same with eq(8) as in issue
            # https://github.com/TencentYoutuResearch/Classification-SemiCLS/issues/9
            # Not that our results in paper doesn't have following line and should
            # mathematically be better after adding.
            score_mask_major = score_mask_major.fill_diagonal_(1)  # all ones
            score_mask_minor_w = score_mask_minor_w.fill_diagonal_(1)
            score_mask_minor_s = score_mask_minor_s.fill_diagonal_(1)
            
            mask_major = mask_major.mul(score_mask_major)  # [bsz, bsz]
            mask_minor_w = mask_minor_w.mul(score_mask_minor_w)
            mask_minor_s = mask_minor_s.mul(score_mask_minor_s)

            # build select matrix for major labels and minor labels
            def build_sel_mat(select_vec_input_left, select_vec_input_right):
                """
                select_vec_input_*: (bsz, ), indicating that one sample is ID (1) or OOD (0), 
                where OOD needs to be filtered.
                select_mat: (bsz, bsz), indicating where each sample can contrast with and OOD only contrasts with itself.
                """
                select_mask = torch.clone(select_vec_input_left).float()
                select_mask[select_vec_input_left == 0] = -1
                select_elements = torch.eq(select_mask.reshape([-1, 1]), select_vec_input_right.reshape([-1, 1]).T).float()
                select_elements += torch.eye(select_vec_input_left.shape[0]).to(device)
                select_elements[select_elements > 1] = 1
                select_mat = torch.ones(select_vec_input_left.shape[0]).to(device) * select_elements
                return select_mat
            # major: ID labels is ok; minor: only the pair of two samples is ID labels is ok
            # ------ major
            select_vec_major = select_vec
            select_mat_major = build_sel_mat(select_vec_major, select_vec_major)

            # ------ minor
            select_vec_minor_w = select_vec[mix_index_weak]
            select_vec_minor_s = select_vec[mix_index_strong]
            select_mat_minor_w = build_sel_mat(select_vec_minor_w, select_vec_major)
            select_mat_minor_s = build_sel_mat(select_vec_minor_s, select_vec_major)
            
            mask_sup_major = mask_major.mul(select_mat_major).repeat(2, 2)
            mask_sup_minor = torch.cat([mask_minor_w.mul(select_mat_minor_w).repeat(1, 2), 
                                    mask_minor_s.mul(select_mat_minor_s).repeat(1, 2)], dim=0)
            
            # --- Unsupervised contrastive learning
            # ------ minor
            labels_unsup = torch.arange(batch_size).long().unsqueeze(1).to(device) # If no labels used, label is the index in mini-batch
            mask_unsup_minor_w = torch.eq(labels_unsup[mix_index_weak], labels_unsup.T).float()
            mask_unsup_minor_s = torch.eq(labels_unsup[mix_index_strong], labels_unsup.T).float()
            mask_unsup_minor = torch.cat([mask_unsup_minor_w.repeat(1, 2), mask_unsup_minor_s.repeat(1, 2)], dim=0)
            # ------ major
            mask_unsup_major = torch.eye(batch_size, dtype=torch.float32).repeat(2, 2).to(device)
        elif labels is not None and select_matrix is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs, max_probs.T)
            # Some may find that the line 59 is different with eq(6)
            # Acutuall the final mask will set weight=0 when i=j, following Eq(8) in paper
            # For more details, please see issue 9

            # Set diagional to 1 to be same with eq(8) as in issue 9
            # https://github.com/TencentYoutuResearch/Classification-SemiCLS/issues/9
            # Not that our results in paper doesn't have following line and should
            # mathematically be better after adding.
            score_mask = score_mask.fill_diagonal_(1)
            mask = mask.mul(score_mask) * select_matrix

        elif labels is not None:
            # --- Supervised contrastive learning
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            labels_major = labels
            labels_minor_w = labels[mix_index_weak]
            labels_minor_s = labels[mix_index_strong]
            labels_major = labels_major.contiguous().view(-1, 1)
            labels_minor_w = labels_minor_w.contiguous().view(-1, 1)
            labels_minor_s = labels_minor_s.contiguous().view(-1, 1)
            mask_major = torch.eq(labels_major, labels_major.T).float().to(device)
            mask_minor_w = torch.eq(labels_minor_w, labels_major.T).float().to(device)
            mask_minor_s = torch.eq(labels_minor_s, labels_major.T).float().to(device)
            #max_probs = max_probs.reshape((batch_size,1))
            max_probs_major = max_probs
            max_probs_minor_w = max_probs[mix_index_weak]
            max_probs_minor_s = max_probs[mix_index_strong]
            max_probs_major = max_probs_major.contiguous().view(-1, 1)
            max_probs_minor_w = max_probs_minor_w.contiguous().view(-1, 1)
            max_probs_minor_s = max_probs_minor_s.contiguous().view(-1, 1)
            score_mask_major = torch.matmul(max_probs_major, max_probs_major.T)  # --- foreground weighting
            score_mask_minor_w = torch.matmul(max_probs_minor_w, max_probs_major.T)
            score_mask_minor_s = torch.matmul(max_probs_minor_s, max_probs_major.T)
            # Set diagional to 1 to be same with eq(8) as in issue
            # https://github.com/TencentYoutuResearch/Classification-SemiCLS/issues/9
            # Not that our results in paper doesn't have following line and should
            # mathematically be better after adding.
            score_mask_major = score_mask_major.fill_diagonal_(1)  # all ones
            score_mask_minor_w = score_mask_minor_w.fill_diagonal_(1)
            score_mask_minor_s = score_mask_minor_s.fill_diagonal_(1)
            
            mask_major = mask_major.mul(score_mask_major)  # [bsz, bsz]
            mask_minor_w = mask_minor_w.mul(score_mask_minor_w)
            mask_minor_s = mask_minor_s.mul(score_mask_minor_s)

            mask_sup_major = mask_major.repeat(2, 2)
            mask_sup_minor = torch.cat([mask_minor_w.repeat(1, 2), mask_minor_s.repeat(1, 2)], dim=0)

            # --- Unsupervised contrastive learning
            # ------ minor
            labels_unsup = torch.arange(batch_size).long().unsqueeze(1).to(device) # If no labels used, label is the index in mini-batch
            mask_unsup_minor_w = torch.eq(labels_unsup[mix_index_weak], labels_unsup.T).float()
            mask_unsup_minor_s = torch.eq(labels_unsup[mix_index_strong], labels_unsup.T).float()
            mask_unsup_minor = torch.cat([mask_unsup_minor_w.repeat(1, 2), mask_unsup_minor_s.repeat(1, 2)], dim=0)
            # ------ major
            mask_unsup_major = torch.eye(batch_size, dtype=torch.float32).repeat(2, 2).to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, dims]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # [bsz*n_views, bsz*n_views]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)  # [bsz*anchor_count, bsz*contrast_count]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask_sup_major),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_sup_major = mask_sup_major * logits_mask  # only positive is nonzero, remove self-contrast case
        mask_sup_minor = mask_sup_minor * logits_mask
        mask_unsup_major = mask_unsup_major * logits_mask
        mask_unsup_minor = mask_unsup_minor * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # Denominator term
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos_sup_major = (mask_sup_major * log_prob).sum(1) / (mask_sup_major.sum(1) + mask_unsup_major.sum(1))
        mean_log_prob_pos_unsup_major = (mask_unsup_major * log_prob).sum(1) / (mask_sup_major.sum(1) + mask_unsup_major.sum(1))
        mean_log_prob_pos_sup_minor = (mask_sup_minor * log_prob).sum(1) / (mask_sup_minor.sum(1) + mask_unsup_minor.sum(1))
        mean_log_prob_pos_unsup_minor = (mask_unsup_minor * log_prob).sum(1) / (mask_sup_minor.sum(1) + mask_unsup_minor.sum(1))
        
        # loss: anchor_count*bsz, [weak,strong]^T
        ## Weight first and second mixup term (both data views) with the corresponding mixing weight

        ##First mixup term. First mini-batch part. Unsupervised + supervised loss separated
        loss1a = -lam1 * mean_log_prob_pos_unsup_major[:int(len(mean_log_prob_pos_unsup_major) / 2)] - lam1 * mean_log_prob_pos_sup_major[:int(len(mean_log_prob_pos_sup_major) / 2)]
        ##First mixup term. Second mini-batch part. Unsupervised + supervised loss separated
        loss1b = -lam2 * mean_log_prob_pos_unsup_major[int(len(mean_log_prob_pos_unsup_major) / 2):] - lam2 * mean_log_prob_pos_sup_major[int(len(mean_log_prob_pos_sup_major) / 2):]
        ## All losses for first mixup term
        loss1 = torch.cat((loss1a, loss1b))

        ##Second mixup term. First mini-batch part. Unsupervised + supervised loss separated
        loss2a = -(1.0 - lam1) * mean_log_prob_pos_unsup_minor[:int(len(mean_log_prob_pos_unsup_minor) / 2)] - (1.0 - lam1) * mean_log_prob_pos_sup_minor[:int(len(mean_log_prob_pos_sup_minor) / 2)]
        ##Second mixup term. Second mini-batch part. Unsupervised + supervised loss separated
        loss2b = -(1.0 - lam2) * mean_log_prob_pos_unsup_minor[int(len(mean_log_prob_pos_unsup_minor) / 2):] - (1.0 - lam2) * mean_log_prob_pos_sup_minor[int(len(mean_log_prob_pos_sup_minor) / 2):]
        ## All losses secondfor first mixup term
        loss2 = torch.cat((loss2a, loss2b))

        ## Final loss (summation of mixup terms after weighting)
        loss = loss1 + loss2

        loss = loss.view(2, batch_size).mean()

        return loss