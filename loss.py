import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def loss_function(x, recon_x, mu, logvar, reconloss, NegSample, device, neg_num, weight):
    if NegSample:
        where_zeros = torch.zeros(x.shape).to(device)
        pos_score = torch.where(x > 0, F.sigmoid(recon_x), where_zeros)
        false_score = torch.where(x == 0.0, F.sigmoid(recon_x), where_zeros)
        bpr_loss = 0 
        for i in range(x.shape[0]):
            pos = pos_score[i].nonzero().view(1,-1)
            pos_only_score = pos_score[i][pos]
            pos_only_score = pos_only_score.view(-1, 1)

            neg_samples = torch.randint(0, x.shape[1], size=(1, neg_num))
            neg_samples = torch.from_numpy(np.setdiff1d(neg_samples.cpu().numpy(),pos.cpu().numpy())).long().unsqueeze(0)
            while neg_samples.shape[1] != neg_num:
                new_sam = torch.randint(0, x.shape[1], size=(1, neg_num - neg_samples.shape[1]))
                neg_samples = torch.cat((neg_samples, new_sam), 1)
            neg_samples = neg_samples.clone().detach()
            neg_ids_u = neg_samples.to(device)
            neg_only_score = false_score[i][neg_ids_u]
            bpr_loss += - torch.sum(F.logsigmoid(pos_only_score - neg_only_score)) / (neg_num * pos.shape[1])
            if i == 0:
                neg_ids = neg_ids_u.unsqueeze(0)
            else:
                neg_ids = torch.cat((neg_ids, neg_ids_u.unsqueeze(0)), 0)

        neg_ids = neg_ids.squeeze()
        # print(neg_ids)
        mask = torch.zeros(x.shape, dtype=torch.uint8).to(device).scatter(1, neg_ids, 1)
        neg_x = torch.masked_select(x, mask)
        neg_score = torch.masked_select(F.sigmoid(recon_x), mask)

        if reconloss == 'gauss':
            mse = nn.MSELoss(reduction='none')
            Recon_loss_pos = torch.mean(torch.sum(mse(pos_score, x), dim=1))
            Recon_loss_neg = torch.sum(mse(neg_score, neg_x)) / (neg_num * x.shape[0])
            Recon_loss = Recon_loss_neg + Recon_loss_pos
        elif reconloss == 'bern':
            bce = nn.BCELoss(reduction='none')
            Recon_loss_pos = torch.mean(torch.sum(bce(pos_score, x), dim=1))
            Recon_loss_neg = torch.sum(bce(neg_score, neg_x)) / (neg_num * x.shape[0])
            Recon_loss = Recon_loss_neg + Recon_loss_pos
        else:
            log_softmax_var = F.log_softmax(recon_x, dim=-1)
            Recon_loss = - torch.mean(torch.sum(log_softmax_var * x, dim=-1))
        # Recon_loss = Recon_loss +  weight * bpr_loss/x.shape[0]
        # Recon_loss = bpr_loss / x.shape[0]
    else:
        if reconloss == 'gauss':
            mse = nn.MSELoss(reduction='none')
            # Recon_loss = F.mse_loss(F.sigmoid(recon_x), x)
            Recon_loss = torch.mean(torch.sum(mse(recon_x, x), dim=1))
        elif reconloss == 'bern':
            bce = nn.BCEWithLogitsLoss(reduction='none')
            # Recon_loss = F.binary_cross_entropy_with_logits(recon_x, x)
            Recon_loss = torch.mean(torch.sum(bce(recon_x, x), dim=1))
        else:
            log_softmax_var = F.log_softmax(recon_x, dim=-1)
            Recon_loss = - torch.mean(torch.sum(log_softmax_var * x, dim=-1))
            
    KLD = KL_divergence(mu, logvar)
    return Recon_loss, KLD

def KL_divergence(mu, logvar):
    # KLD = 0.5 * torch.sum(-1 + sigma.pow(2) + mu.pow(2) - torch.log(1e-8 + sigma.pow(2)), 1)
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    # KLD = torch.mean(KLD) 
    return KLD

 
def mf_loss(pos_score, neg_score, pos_weight):
    # mse = nn.MSELoss()
    # pa = torch.ones_like(pos_score)
    # na = torch.zeros_like(neg_score)
    # loss = pos_weight * mse(pa, F.sigmoid(pos_score)) + mse(na, F.sigmoid(neg_score))
    neg_score = neg_score.view(1, -1)
    pos_score = pos_score.view(-1, 1)
    loss = - torch.sum(F.logsigmoid(pos_score - neg_score)) / (neg_score.shape[1]*pos_score.shape[0])
    return loss
        

