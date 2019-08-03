import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from dataloader import *
from model import *
from model_baseline import *
import time 
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
import torch
from datetime import datetime
from loss import *
from evaluation import *
from dataset_total import *

class Trainer(object):
    def __init__(self, args):
        self.args = args
        if self.args.condition:
            self.hidden_vecs = {}
            for i in range(self.args.class_num):
                hidden = "h_{}".format(i+1)
                # self.hidden_vecs[hidden] = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)
                self.hidden_vecs[hidden] = torch.zeros(self.args.condition_size).to(self.args.device)

    def train(self):
        ### DAE
        if self.args.baseline == 'DAE':
            p_dims = self.args.dims
            # p_dims = [self.args.latent_size, self.args.hidden_size_vae, self.args.input_size_vae]
            model = MultiDAE(p_dims)
            optimizer = optim.Adam(model.parameters(), lr = self.args.lr_vae, weight_decay=0.0)
        elif self.args.baseline == 'MF':
            model = MF(6040, 3355, 100)
            optimizer = optim.SparseAdam(model.parameters(), lr = self.args.lr_mf)
            # optimizer = optim.SGD(model.parameters(), lr=1e-6, weight_decay=1e-5)
        elif self.args.baseline == 'HPrior':
            model = HPrior_VAE(self.args.dims, self.args.hidden_size_rnn, self.args.dataset,\
                                    self.args.input_size_rnn, self.args.activation)
            optimizer = optim.Adam(model.parameters(), lr = self.args.lr_vae, weight_decay=0.0)
        else:
            model = VAE_RNN_rec(self.args.dims, self.args.input_size_rnn, self.args.embedding_size,\
                                self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                                self.args.hidden_size_rnn, self.args.condition, self.args.dataset, self.args.activation, self.args.freeze, self.args.attn, self.args.condition_size)
            optimizer = {
            'encoder' : optim.Adam(model.encoder.parameters(), lr=self.args.lr_vae, weight_decay=0.0),
            'decoder' : optim.Adam(model.decoder.parameters(), lr=self.args.lr_vae, weight_decay=0.0)
            }
        
        model = model.to(self.args.device)
        if self.args.data_dir == './data/ml-1m':
            dataloader = ItemRatingLoader(self.args.data_dir)
        elif self.args.data_dir == './data/amazon' or self.args.data_dir == './data/amazon_min20_woman'\
                or self.args.data_dir == './data/amazon_min20_woman_fix' or self.args.dataset=='amazon_min10_woman':
            dataloader = AmazonRatingLoader(self.args.data_dir, self.args.dims[0])

        if self.args.condition:
            optimizer['RNNEncoder'] = optim.Adam(model.RNNEncoder.parameters(), lr=self.args.lr_rnn, weight_decay=0.0)
            # weight = torch.FloatTensor([0.18, 0.28, 0.54]).to(args.device)
            # CEloss = nn.CrossEntropyLoss(weight = weight)
            CEloss = nn.CrossEntropyLoss()

        if self.args.load_model:
            model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))
            self.args.timestamp = self.args.load_model[:10]
        if self.args.condition and self.args.load_pretrained:
            model.RNNEncoder.load_state_dict(torch.load(self.args.pretrained_dir + '/' + self.args.load_pretrained + '/' + 'model.pt'))
            print("loaded pretrained model")
        
        writer = SummaryWriter(self.args.log_dir + "/" + self.args.timestamp + "_" + self.args.config)
        # if self.args.baseline == 'MF':
        #     train_data_rating = dataloader.load_train_data_mf(os.path.join(self.args.data_dir, 'train.csv'), os.path.join(self.args.data_dir, 'valid_tr.csv'),\
        #                                                             self.args.batch_size, int(374215/self.args.batch_size)+1)
        #     N = 374215
        if not self.args.baseline == 'MF':
            train_data_rating = dataloader.load_train_data(os.path.join(self.args.data_dir, 'train.csv'))
            N = train_data_rating.shape[0]
        
            idxlist = np.array(range(N))
            idx_pe = np.random.permutation(len(idxlist))
            idxlist = idxlist[idx_pe]
        b_ndcg100 = 0.0
        update_count = 0.0
        for e in range(self.args.epoch):
            model.train()
            total_loss = 0
            if self.args.baseline == 'MF':
                train_data_rating = dataloader.load_train_data_mf(os.path.join(self.args.data_dir, 'train.csv'), os.path.join(self.args.data_dir, 'valid_tr.csv'),os.path.join(self.args.data_dir, 'test_tr.csv'),
                                                                        self.args.batch_size, int(374215/self.args.batch_size)+1)
                N = 374215
            if self.args.condition or self.args.baseline == 'HPrior':
                train_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'train', self.args.batch_size, idx_pe)
            
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition or self.args.baseline == 'HPrior':
                    order, item_feature, label = next(train_data_item)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    x_unorder = train_data_rating[idxlist[st_idx:end_idx]]
                    X = x_unorder[order]
                else:
                    end_idx = min(st_idx + self.args.batch_size, N)
                    if self.args.baseline == 'MF':
                        d = next(train_data_rating)
                        pos = d[0]
                        neg = d[1]
                    else:
                        X = train_data_rating[idxlist[st_idx:end_idx]]
                if not self.args.baseline == 'MF':
                    if sparse.isspmatrix(X):
                        X = X.toarray()
                    X = X.astype('float32')   

                if self.args.condition:
                    optimizer["RNNEncoder"].zero_grad()
                    output, h = model.RNNEncoder(item_feature.to(self.args.device))
                    rnn_loss = CEloss(output, label.to(self.args.device))
                    rnn_loss.backward(retain_graph=True)
                    # rnn_loss.backward()
                    # optimizer["RNNEncoder"].step()
                    # self.make_condition(h, label.data)
                    self.make_condition(h, label.data)


                if self.args.baseline:
                    optimizer.zero_grad()
                else:
                    optimizer["encoder"].zero_grad()
                    optimizer["decoder"].zero_grad()
                
                if self.args.condition:
                    if self.args.test_hidden == 'onehot':
                        h = self.tooh(label, self.args.class_num).to(self.args.device)
                    model_input = (torch.FloatTensor(X).to(self.args.device), F.sigmoid(h))
                    recon, mu, logvar = model(model_input)
                elif self.args.baseline == 'HPrior':
                    recon, mu, logvar = model(torch.FloatTensor(X).to(self.args.device), item_feature.to(self.args.device))
                else:
                    if self.args.baseline == 'DAE':
                        recon = model(torch.FloatTensor(X).to(self.args.device))
                    elif self.args.baseline == 'MF':
                        pos = list(zip(*pos))
                        user = torch.LongTensor(pos[0]).to(self.args.device)
                        item = torch.LongTensor(pos[1]).to(self.args.device)
                        neg = list(zip(*neg))
                        user_neg = torch.LongTensor(neg[0]).to(self.args.device)
                        item_neg = torch.LongTensor(neg[1]).to(self.args.device)
                        ps, ns = model(user, item, user_neg, item_neg)
                        mfloss = mf_loss(ps, ns, 30)
                    else:
                        recon, mu, logvar = model(torch.FloatTensor(X).to(self.args.device))
                
                if self.args.baseline == 'DAE':
                    log_softmax_var = F.log_softmax(recon, dim=-1)
                    recon_loss = - torch.mean(torch.sum(log_softmax_var * torch.FloatTensor(X).to(self.args.device), dim=-1))
                if not self.args.baseline or self.args.baseline == 'HPrior':
                    recon_loss, kld = loss_function(torch.FloatTensor(X).to(self.args.device), recon, mu, logvar, self.args.dist, \
                                                                self.args.negsample, self.args.device, self.args.neg_num, self.args.bpr_weight)
                if self.args.anneal_steps > 0:
                    anneal = min(self.args.anneal_cap, 1. * update_count / self.args.anneal_steps)
                    update_count  += 1
                else:
                    anneal = self.args.anneal_cap
                if self.args.baseline == 'DAE':
                    vae_loss = recon_loss
                elif self.args.baseline == 'MF':
                    vae_loss = mfloss
                else:
                    vae_loss = recon_loss + anneal * kld
                vae_loss.backward()

                if self.args.baseline:
                    optimizer.step()
                else:
                    optimizer["encoder"].step()
                    optimizer["decoder"].step()

                if self.args.condition:
                    optimizer["RNNEncoder"].step()
                # r10, r20, r50, r100, ndcg10, ndcg50, ndcg100, ndcg150, auc = self.test(model, anneal)
                # tensorboard
                if self.args.condition:
                    writer.add_scalar("Train rnn loss", rnn_loss, i + e*N/self.args.batch_size)
                writer.add_scalar("Train vae loss", vae_loss, i + e*N/self.args.batch_size)
                if not self.args.baseline =='MF':
                    writer.add_scalar("Recon loss", recon_loss, i + e*N/self.args.batch_size)
                if not self.args.baseline:
                    writer.add_scalar("KLD", kld, i + e*N/self.args.batch_size)
                
                if i % 20 == 0:
                    if not self.args.baseline:
                        print(f"recon : {recon_loss.item():.3} | kld : {kld.item():.3}")
                    if self.args.condition:
                        print(f"epoch : {e} | train_vae_loss : {vae_loss.item():.3} | train_rnn_loss : {rnn_loss.item():.3}", 
                        f"[{i*self.args.batch_size} / {N}","(",f"{(i/N*self.args.batch_size)*100:.3} %", ")]")
                    else:
                        print(f"epoch : {e} | train_vae_loss : {vae_loss.item():.3}", 
                        f"[{i*self.args.batch_size} / {N}","(",f"{(i/N*self.args.batch_size)*100:.3} %", ")]")
                total_loss += vae_loss
            # save model
            # torch.save(model.state_dict(), self.args.log_dir + '/' + self.args.timestamp + '_' + self.args.config + '/model.pt')
            # print("model saved!")
            print(f"epoch : {e} | train vae loss : {total_loss / (N/self.args.batch_size):.3} ")
            if self.args.condition:
                #save condition per epoch for evaluation
                for j in range(self.args.class_num):
                    hidden = "h_{}".format(j+1)
                    torch.save(self.hidden_vecs[hidden], f"{self.args.hiddenvec_dir}/{hidden}.pt")
                print("hidden vector saved!")
            # test per epoch
            r10, r20, r50, r100, ndcg10, ndcg50, ndcg100, ndcg150, auc = self.test(model, anneal)
            if ndcg50 > b_ndcg100 :
                torch.save(model.state_dict(), self.args.log_dir + '/' + self.args.timestamp + '_' + self.args.config + '/model.pt')
                print("model saved!")
                b_ndcg100 = ndcg50
            # tensorboard
            # writer.add_scalar("Test_loss", test_loss, e)
            writer.add_scalar("Test_Recall10", r10, e)
            writer.add_scalar("Test_Recall20", r20, e)
            writer.add_scalar("Test_Recall50", r50, e)
            writer.add_scalar("Test_Recall100", r100, e)
            writer.add_scalar("Test_NDCG10", ndcg10, e)
            writer.add_scalar("Test_NDCG50", ndcg50, e)
            writer.add_scalar("Test_NDCG100", ndcg100, e)
            writer.add_scalar("Test_NDCG150", ndcg150, e)
            writer.add_scalar("Test_AUC", auc, e)

    def test(self, model, anneal):
        model.eval()
        if self.args.data_dir == './data/ml-1m':
            dataloader = ItemRatingLoader(self.args.data_dir)
            if self.args.baseline == 'MF':
                tr_data_rating = dataloader.load_tr_te_data_mf(os.path.join(self.args.data_dir, 'fixed2_valid_te.csv'), 1)
                N = 177 * self.args.batch_size
            else:
                tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed2_valid_tr.csv'), os.path.join(self.args.data_dir, 'fixed2_valid_te.csv'))
        elif self.args.data_dir == './data/amazon' or self.args.data_dir =='./data/amazon_min20_woman' \
                            or self.args.data_dir =='./data/amazon_min20_woman_fix' or self.args.dataset=='amazon_min10_woman':
            dataloader = AmazonRatingLoader(self.args.data_dir, self.args.dims[0])
            if self.args.baseline == 'MF':
                tr_data_rating = dataloader.load_tr_te_data_mf(os.path.join(self.args.data_dir, 'valid_te.csv'), 1)
                N = 177 * self.args.batch_size
            else:
                tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'valid_tr.csv'), os.path.join(self.args.data_dir, 'valid_te.csv'))
                # print(tr_data_rating.shape[0])
        # if self.args.baseline == 'MF':
            # tr_data_rating = dataloader.load_tr_te_data_mf(os.path.join(self.args.data_dir, 'fixed_valid_te.csv'), 1)
            # N = 177 * self.args.batch_size
        if not self.args.baseline == 'MF':
            N = tr_data_rating.shape[0]
            idxlist = np.array(range(N))
            np.random.seed(98765)
            idx_pe = np.random.permutation(len(idxlist))
            idxlist = idxlist[idx_pe]
        if self.args.condition or self.args.baseline == 'HPrior':  
            valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'valid', self.args.batch_size, idx_pe)
        
        test_loss = 0 
        with torch.no_grad():
            r20_list, r50_list,r10_list, r100_list, ndcg_list50, ndcg_list10, ndcg_list100, ndcg_list150, auc_list \
                                                                                    = [], [], [], [], [],[], [], [], []
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition or self.args.baseline == 'HPrior':
                    order, item_feature, label = next(valid_data_item)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    x_tr_unorder = tr_data_rating[idxlist[st_idx:end_idx]]
                    X_tr = x_tr_unorder[order]
                    x_te_unorder = te_data_rating[idxlist[st_idx:end_idx]]
                    X_te = x_te_unorder[order]
                else:
                    if not self.args.baseline == 'MF':
                        end_idx = min(st_idx + self.args.batch_size, N)
                        X_tr = tr_data_rating[idxlist[st_idx:end_idx]]
                        X_te = te_data_rating[idxlist[st_idx:end_idx]]
                    
                    else:
                        dd = next(tr_data_rating)
                        X_te = dd[0]
                        X_pre = dd[1]

                if not self.args.baseline == 'MF':
                    if sparse.isspmatrix(X_tr):
                        X_tr = X_tr.toarray()
                    X_tr = X_tr.astype('float32') 
                if self.args.condition:
                    if self.args.test_hidden == 'trained':
                        h = []
                        for b_c in label:
                            for j in range(self.args.class_num):
                                hidden = "h_{}".format(j+1)
                                if b_c == j:
                                    h.append(self.hidden_vecs[hidden].unsqueeze(0))
                        hidden = torch.cat(h, 0)
                    elif self.args.test_hidden == 'onehot':
                        hidden = self.tooh(label, self.args.class_num).to(self.args.device)
                    else:
                        _, hidden = model.RNNEncoder(item_feature.to(self.args.device))
                    model_input = (torch.FloatTensor(X_tr).to(self.args.device), F.sigmoid(hidden))
                    recon, mu, logvar = model(model_input)
                elif self.args.baseline == 'HPrior':
                    recon, mu, logvar = model(torch.FloatTensor(X_tr).to(self.args.device), item_feature.to(self.args.device))
                else:
                    if not self.args.baseline:
                        recon, mu, logvar = model(torch.FloatTensor(X_tr).to(self.args.device))
                    elif self.args.baseline == 'MF':
                        pre = [list(t) for t in zip(*X_pre)]
                        user = torch.LongTensor(list(set(pre[0]))).to(self.args.device)
                        item = torch.LongTensor(pre[1]).to(self.args.device)
                        recon = model(user, item, None, None)
                        recon = recon.view(1,-1)
                    else:
                        recon = model(torch.FloatTensor(X_tr).to(self.args.device))
                if not self.args.baseline or self.args.baseline == 'HPrior': 
                    recon_loss, kld = loss_function(torch.FloatTensor(X_tr).to(self.args.device), recon, mu, logvar, self.args.dist, \
                                                    self.args.negsample, self.args.device, self.args.neg_num, self.args.bpr_weight)
                    loss = recon_loss + anneal * kld
                    test_loss += loss

                if not self.args.baseline == 'MF':
                    recon[X_tr.nonzero()] = -np.inf
                # print(X_te.shape)
                ndcg_list50.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                ndcg_list100.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=100))
                ndcg_list10.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
                ndcg_list150.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=150))
                r20_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=20))
                r50_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                r10_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
                r100_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=100))
                auc_list.append(AUC_score(recon.cpu().detach().numpy(), X_te))
                # print(ndcg_list100)
                # print(AUC_score(recon.cpu().detach().numpy(), X_te))
                

            ndcg_list50 = np.concatenate(ndcg_list50)
            ndcg_list100 = np.concatenate(ndcg_list100)
            ndcg_list150 = np.concatenate(ndcg_list150)
            ndcg_list10 = np.concatenate(ndcg_list10)
            r20_list = np.concatenate(r20_list)
            r10_list = np.concatenate(r10_list)
            r50_list = np.concatenate(r50_list)       
            r100_list = np.concatenate(r100_list) 
            auc_list = np.asarray(auc_list)      

            if not self.args.baseline:
                print(f"test loss : {test_loss / (N/self.args.batch_size):.3}")
            print("Test NDCG@10=%.5f (%.5f)" % (ndcg_list10.mean(), np.std(ndcg_list10) / np.sqrt(len(ndcg_list10))))
            print("Test NDCG@50=%.5f (%.5f)" % (ndcg_list50.mean(), np.std(ndcg_list50) / np.sqrt(len(ndcg_list50))))
            print("Test NDCG@100=%.5f (%.5f)" % (ndcg_list100.mean(), np.std(ndcg_list100) / np.sqrt(len(ndcg_list100))))
            print("Test NDCG@150=%.5f (%.5f)" % (ndcg_list150.mean(), np.std(ndcg_list150) / np.sqrt(len(ndcg_list150))))
            print("Test Recall@10=%.5f (%.5f)" % (r10_list.mean(), np.std(r10_list) / np.sqrt(len(r10_list))))
            print("Test Recall@20=%.5f (%.5f)" % (r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))))
            print("Test Recall@50=%.5f (%.5f)" % (r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))))
            print("Test Recall@100=%.5f (%.5f)" % (r100_list.mean(), np.std(r100_list) / np.sqrt(len(r100_list))))
            print("Test AUC=%.5f (%.5f)" % (auc_list.mean(), np.std(auc_list) / np.sqrt(len(auc_list))))
        return np.mean(r10_list), np.mean(r20_list), np.mean(r50_list), np.mean(r100_list), \
                    np.mean(ndcg_list10), np.mean(ndcg_list50),  np.mean(ndcg_list100), np.mean(ndcg_list150), np.mean(auc_list)
        # return test_loss / (N/self.args.batch_size), np.mean(r20_list), np.mean(r50_list), np.mean(ndcg_list)
    
    def test_testset(self):
        if self.args.condition:
            for k in range(self.args.class_num):
                hidden = "h_{}".format(k+1)
                self.hidden_vecs[hidden]= torch.load(f"{self.args.hiddenvec_dir}/{hidden}.pt")
        if self.args.baseline == 'DAE':
            p_dims = self.args.dims
            # p_dims = [self.args.latent_size, self.args.hidden_size_vae, self.args.input_size_vae]
            model = MultiDAE(p_dims)
        elif self.args.baseline == 'HPrior':
            model = HPrior_VAE(self.args.dims, self.args.hidden_size_rnn, self.args.dataset,\
                                    self.args.input_size_rnn, self.args.activation)
        elif self.args.baseline == 'MF':
            model = MF(6040, 3355, 100)
        else:
            model = VAE_RNN_rec(self.args.dims, self.args.input_size_rnn, self.args.embedding_size,\
                                self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                                self.args.hidden_size_rnn, self.args.condition, self.args.dataset, self.args.activation, self.args.freeze, self.args.attn, self.args.condition_size)
        model = model.to(self.args.device)

        model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))
        model.eval()
        if self.args.data_dir == './data/ml-1m':
            dataloader = ItemRatingLoader(self.args.data_dir)
            if self.args.baseline == 'MF':
                tr_data_rating = dataloader.load_tr_te_data_mf(os.path.join(self.args.data_dir, 'fixed2_test_te.csv'), 1)
                N = 177 * self.args.batch_size
            else:#fixed_valid / fixed_test
                # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed2_test_tr.csv'), \
                # os.path.join(self.args.data_dir, 'fixed2_test_te.csv'))
                tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed_valid_tr.csv'), \
                os.path.join(self.args.data_dir, 'fixed_valid_te.csv'))
        elif self.args.data_dir == './data/amazon' or self.args.data_dir == './data/amazon_min20_woman' or self.args.data_dir =='./data/amazon_min20_woman_fix' or self.args.dataset=='amazon_min10_woman':
            dataloader = AmazonRatingLoader(self.args.data_dir, self.args.dims[0])
            # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'test_tr.csv'),\
                                                                    #  os.path.join(self.args.data_dir, 'test_te.csv'))
            tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'valid_tr.csv'),\
                                                                     os.path.join(self.args.data_dir, 'valid_te.csv'))

        if not self.args.baseline == 'MF':
            N = tr_data_rating.shape[0]
            idxlist = np.array(range(N))
            np.random.seed(98765)
            idx_pe = np.random.permutation(len(idxlist))
            idxlist = idxlist[idx_pe]
        if self.args.condition or self.args.baseline == 'HPrior':  
            valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'valid', self.args.batch_size, idx_pe)
            # valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'test', self.args.batch_size, idx_pe)
            if self.args.test_hidden == 'fixed':
                tr_data_hidden = dataloader.fixed_hidden(int(N/self.args.batch_size)+1, 'valid', self.args.batch_size, idx_pe)
            
        with torch.no_grad():
            r20_list, r50_list,r5_list,r10_list, r100_list, ndcg_list50, ndcg_list5,ndcg_list10, ndcg_list100, ndcg_list150, auc_list = [], [], [], [], [], [],[], [], [], [], []
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition or self.args.baseline == 'HPrior':
                    order, item_feature, label = next(valid_data_item)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    x_tr_unorder = tr_data_rating[idxlist[st_idx:end_idx]]
                    X_tr = x_tr_unorder[order]
                    x_te_unorder = te_data_rating[idxlist[st_idx:end_idx]]
                    X_te = x_te_unorder[order]
                else:
                    if not self.args.baseline == 'MF':
                        end_idx = min(st_idx + self.args.batch_size, N)
                        X_tr = tr_data_rating[idxlist[st_idx:end_idx]]
                        X_te = te_data_rating[idxlist[st_idx:end_idx]]

                    else:
                        dd = next(tr_data_rating)
                        X_te = dd[0]
                        X_pre = dd[1]
                if not self.args.baseline == 'MF':
                    if sparse.isspmatrix(X_tr):
                        X_tr = X_tr.toarray()
                    X_tr = X_tr.astype('float32')

                if self.args.condition:
                    if self.args.test_hidden == 'trained':
                        print('use trained hidden vector')
                        h = []
                        for b_c in label:
                            for j in range(self.args.class_num):
                                hidden = "h_{}".format(j+1)
                                if b_c == j:
                                    h.append(self.hidden_vecs[hidden].unsqueeze(0))
                        hidden = torch.cat(h, 0)
                        # _, hidden = model.RNNEncoder(item_feature.to(self.args.device))
                    elif self.args.test_hidden == 'onehot':
                        hidden = self.tooh(label, self.args.class_num).to(self.args.device)
                    elif self.args.test_hidden == 'fixed':
                        hidden = next(tr_data_hidden)
                    else:
                        _, hidden = model.RNNEncoder(item_feature.to(self.args.device))
                    if len(hidden.shape) == 1:
                        hidden = hidden.unsqueeze(0)
                    model_input = (torch.FloatTensor(X_tr).to(self.args.device), F.sigmoid(hidden))
                    recon, _, _ = model(model_input)
                elif self.args.baseline == 'HPrior':
                    recon, _, _ = model(torch.FloatTensor(X_tr).to(self.args.device), item_feature.to(self.args.device))
                else:
                    if not self.args.baseline:
                        recon, _, _ = model(torch.FloatTensor(X_tr).to(self.args.device))
                    elif self.args.baseline == 'MF':
                        pre = [list(t) for t in zip(*X_pre)]
                        user = torch.LongTensor(list(set(pre[0]))).to(self.args.device)
                        item = torch.LongTensor(pre[1]).to(self.args.device)
                        recon = model(user, item, None, None)
                        recon = recon.view(1,-1)
                    else:
                        recon = model(torch.FloatTensor(X_tr).to(self.args.device))
                # if not self.args.baseline: 
                    # recon_loss, kld = loss_function(torch.FloatTensor(X_tr).to(self.args.device), recon, mu, logvar, self.args.dist)
                if not self.args.baseline == 'MF':
                    recon[X_tr.nonzero()] = -np.inf
                ndcg_list50.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                ndcg_list100.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=100))
                ndcg_list10.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
                ndcg_list5.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=5))
                ndcg_list150.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=150))
                r20_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=20))
                r50_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                r10_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
                r5_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=5))
                r100_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=100))
                auc_list.append(AUC_score(recon.cpu().detach().numpy(), X_te))

            ndcg_list50 = np.concatenate(ndcg_list50)
            ndcg_list100 = np.concatenate(ndcg_list100)
            ndcg_list150 = np.concatenate(ndcg_list150)
            ndcg_list10 = np.concatenate(ndcg_list10)
            ndcg_list5 = np.concatenate(ndcg_list5)
            r20_list = np.concatenate(r20_list)
            r10_list = np.concatenate(r10_list)
            r50_list = np.concatenate(r50_list)       
            r5_list = np.concatenate(r5_list)       
            r100_list = np.concatenate(r100_list)       
            auc_list = np.asarray(auc_list)    
            # if not self.args.baseline:
                # print(f"test loss : {test_loss / (N/self.args.batch_size):.3}")
            print("Test NDCG@5=%.5f (%.5f)" % (ndcg_list5.mean(), np.std(ndcg_list5) / np.sqrt(len(ndcg_list5))))
            print("Test NDCG@10=%.5f (%.5f)" % (ndcg_list10.mean(), np.std(ndcg_list10) / np.sqrt(len(ndcg_list10))))
            print("Test NDCG@50=%.5f (%.5f)" % (ndcg_list50.mean(), np.std(ndcg_list50) / np.sqrt(len(ndcg_list50))))
            print("Test NDCG@100=%.5f (%.5f)" % (ndcg_list100.mean(), np.std(ndcg_list100) / np.sqrt(len(ndcg_list100))))
            print("Test NDCG@150=%.5f (%.5f)" % (ndcg_list150.mean(), np.std(ndcg_list150) / np.sqrt(len(ndcg_list150))))
            print("Test Recall@5=%.5f (%.5f)" % (r5_list.mean(), np.std(r5_list) / np.sqrt(len(r5_list))))
            print("Test Recall@10=%.5f (%.5f)" % (r10_list.mean(), np.std(r10_list) / np.sqrt(len(r10_list))))
            print("Test Recall@20=%.5f (%.5f)" % (r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))))
            print("Test Recall@50=%.5f (%.5f)" % (r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))))
            print("Test Recall@100=%.5f (%.5f)" % (r100_list.mean(), np.std(r100_list) / np.sqrt(len(r100_list))))
            print("Test AUC=%.5f (%.5f)" % (auc_list.mean(), np.std(auc_list) / np.sqrt(len(auc_list))))


    def tooh(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
        labels: (LongTensor) class labels, sized [N,].
        num_classes: (int) number of classes.

        Returns:
        (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes) 
        return y[labels]

    def check_condition(self, period):
        model = VAE_RNN_rec(self.args.dims, self.args.input_size_rnn, self.args.embedding_size,\
                                self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                                self.args.hidden_size_rnn, self.args.condition, self.args.dataset, self.args.activation, self.args.freeze, self.args.attn)
        model = model.to(self.args.device)

        if self.args.load_model:
            model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))

        if self.args.data_dir == './data/ml-1m':
            dataloader = ItemRatingLoader(self.args.data_dir)
        elif self.args.data_dir == './data/amazon':
            dataloader = AmazonRatingLoader(self.args.data_dir, self.args.dims[0])
        # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed2_test_tr.csv'), os.path.join(self.args.data_dir, 'fixed2_test_te.csv'))
        tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'test_tr.csv'), os.path.join(self.args.data_dir, 'test_te.csv'))
        N = tr_data_rating.shape[0]
        idxlist = np.array(range(N))
        # print(N)
        np.random.seed(98764)
        idx_pe = np.random.permutation(len(idxlist))

        idxlist = idxlist[idx_pe]
        # print(idxlist[:self.args.batch_size])
        # if self.args.condition:  
        valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'test', self.args.batch_size, idx_pe)
        with torch.no_grad():
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                order, item_feature, label = next(valid_data_item)
                end_idx = min(st_idx + self.args.batch_size, N)
                x_tr_unorder = tr_data_rating[idxlist[st_idx:end_idx]]
                X_tr = x_tr_unorder[order]
                x_te_unorder = te_data_rating[idxlist[st_idx:end_idx]]
                X_te = x_te_unorder[order]
                # print(label.item())
                if not label.item() == period:
                    continue
                else:
                    if sparse.isspmatrix(X_tr):
                        X_tr = X_tr.toarray()
                    X_tr = X_tr.astype('float32')
                    for k in range(self.args.class_num):
                        hidden = "h_{}".format(k+1)
                        self.hidden_vecs[hidden]= torch.load(f"{self.args.hiddenvec_dir}/{hidden}.pt")
                    hs = {}
                    for _ in range(self.args.batch_size):
                        for k in range(self.args.class_num):
                            hidden = "h_{}".format(k+1)
                            hs.setdefault(hidden, []).append(self.hidden_vecs[hidden].unsqueeze(0))
                    ndcg_result={}
                    for j in range(self.args.class_num):
                        hidden = "h_{}".format(j+1)
                        hv = torch.cat(hs[hidden],0)
                        model_input = (torch.FloatTensor(X_tr).to(self.args.device), hv)
                        recon, _, _ = model(model_input)
                        topk = show_recommended_items(recon.cpu().detach().numpy(), k=50)
                        with open(f'./result/amazon/qual/topk{j}.pkl', 'wb') as f:
                            pickle.dump(topk, f)   
                        recon[X_tr.nonzero()] = -np.inf
                        nd_name = "ndcg_list50_{}".format(j+1)
                        ndcg_result.setdefault(nd_name, []).append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
            ndcg_final_result={}
            for j in range(self.args.class_num): 
                nd_name = "ndcg_list50_{}".format(j+1)
                ndcg_final_result[nd_name] = np.concatenate(ndcg_result[nd_name])
            return ndcg_final_result

    def make_condition(self, h, label):
        for i in range(len(label)):
            for k in range(self.args.class_num):
                hidden = "h_{}".format(k+1)
                if label[i] == k:
                    self.hidden_vecs[hidden] = (self.hidden_vecs[hidden] + h[i].squeeze()) / 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', default=[33310, 300, 100], type=list)
    parser.add_argument('--input-size_rnn', default=4096, type=int)
    parser.add_argument('--hidden-size_rnn', default=100, type=int)
    parser.add_argument('--condition-size', default=50, type=int)
    parser.add_argument('--embedding-size', default=500, type=int)
    parser.add_argument('--seq-len', default=20, type=int)
    parser.add_argument('--num-layer', default=1, type=int)
    parser.add_argument('--dropout-rate', default=0.2, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--num-workers', default=15, type=int)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--anneal-steps', default=200000, type=int)
    parser.add_argument('--anneal-cap', default=0.2, type=float)
    parser.add_argument('--class-num', default=5, type=int)
    parser.add_argument('--lr_rnn', default=0.001, type=float)
    parser.add_argument('--lr_vae', default=0.001, type=float)
    parser.add_argument('--lr_mf', default=0.0003, type=float)
    parser.add_argument('--bpr-weight', default=1.0, type=float)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--negsample', action='store_true')
    parser.add_argument('--neg-num', default=50, type=int)
    parser.add_argument('--dist', default='multi', type=str)
    parser.add_argument('--test-hidden', default='basic', type=str)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--dataset', default='amazon_min20_woman_fix', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--memo', default='', type=str)
    parser.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M"), type=str)
    parser.add_argument('--condition', action="store_false")
    parser.add_argument('--load-model', default=None, type=str)
    parser.add_argument('--load-pretrained', default=None, type=str)
    parser.add_argument('--baseline', default=None, type=str)
    parser.add_argument('--test',  action='store_true')

    args = parser.parse_args()

    if args.baseline:
        args.condition = False
    if args.dataset == 'amazon' or args.dataset == 'amazon_min20_woman' or args.dataset =='amazon_min20_woman_fix' or args.dataset=='amazon_min10_woman':
        args.class_num = 5
        args.input_size_rnn = 4096
        args.embedding_size = 500
        if args.dataset == 'amazon_min20_woman':
            args.dims[0] = 16736
            args.dims[1] = 500
        elif args.dataset == 'amazon_min20_woman_fix':
            args.dims[0] = 15931
            args.dims[1] = 500
        elif args.dataset == 'amazon_min10_woman':
            args.dims[0] = 31842
            args.dims[1] = 1000
        else:
            args.dim[0] = 14253
    elif args.dataset == 'ml-1m':
        args.class_num = 3
        args.input_size_rnn = 1128
        args.dims[0] = 3355
        # args.dims[1] = 600
        # args.dims[2] = 200
        args.dims = [3355, 100]
        args.embedding_size =300

    args.data_dir = f'./data/{args.dataset}'
    args.log_dir = f'./result/{args.dataset}/joint_train'
    args.pretrained_dir = f'./result/{args.dataset}/pretrain'
    args.hiddenvec_dir = f'./pretrained/{args.dataset}/joint'
    
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config_list = [args.embedding_size, args.seq_len, args.hidden_size_rnn, args.bidirectional, args.lr_vae, \
                        args.lr_rnn, args.negsample, args.dist, args.condition, args.anneal_cap, args.baseline, args.attn, args.memo]
    print("using",args.device)
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    
    trainer = Trainer(args)
    if args.test:
        trainer.test_testset()
    else:
        trainer.train()
    # for i in range(args.class_num):
    #     result = trainer.check_condition(i)
    #     for j in range(args.class_num):
    #         nd_name = "ndcg_list50_{}".format(j+1)
    #         print(f"{i}-{j} {result[nd_name].mean()}")