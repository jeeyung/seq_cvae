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
from data5 import *

class Trainer(object):
    def __init__(self, args):
        self.args = args
        if self.args.condition:
            self.hidden_vecs = {}
            for i in range(self.args.class_num):
                hidden = "h_{}".format(i+1)
                self.hidden_vecs[hidden] = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)

    def train(self):
        ### DAE
        if self.args.baseline == 'DAE':
            p_dims = [self.args.latent_size, self.args.hidden_size_vae, self.args.input_size_vae]
            model = MultiDAE(p_dims)
            optimizer = optim.Adam(model.parameters(), lr = self.args.lr_vae, weight_decay=0.0)
        else:
            model = VAE_RNN_rec(self.args.dims, self.args.input_size_rnn, self.args.embedding_size,\
                                self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                                self.args.hidden_size_rnn, self.args.condition, self.args.data_dir, self.args.activation)
            optimizer = {
            'encoder' : optim.Adam(model.encoder.parameters(), lr=self.args.lr_vae, weight_decay=0.0),
            'decoder' : optim.Adam(model.decoder.parameters(), lr=self.args.lr_vae, weight_decay=0.0)
            }
        
        model = model.to(self.args.device)
        dataloader = ItemRatingLoader(self.args.data_dir)
        
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
        train_data_rating = dataloader.load_train_data(os.path.join(self.args.data_dir, 'train.csv'))
        N = train_data_rating.shape[0]
        
        idxlist = np.array(range(N))

        # np.random.seed(98765)
        idx_pe = np.random.permutation(len(idxlist))
        idxlist = idxlist[idx_pe]

        update_count = 0.0
        for e in range(self.args.epoch):
            model.train()
            total_loss = 0
            if self.args.condition:
                train_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'train', self.args.batch_size, idx_pe)
            
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition:
                    order, item_feature, label = next(train_data_item)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    x_unorder = train_data_rating[idxlist[st_idx:end_idx]]
                    X = x_unorder[order]
                else:
                    end_idx = min(st_idx + self.args.batch_size, N)
                    X = train_data_rating[idxlist[st_idx:end_idx]]

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
                    model_input = (torch.FloatTensor(X).to(self.args.device), h)
                    recon, mu, logvar = model(model_input)
                else:
                    if self.args.baseline:
                        recon = model(torch.FloatTensor(X).to(self.args.device))
                    else:
                        recon, mu, logvar = model(torch.FloatTensor(X).to(self.args.device))
                
                if self.args.baseline:
                    log_softmax_var = F.log_softmax(recon, dim=-1)
                    recon_loss = - torch.mean(torch.sum(log_softmax_var * torch.FloatTensor(X).to(self.args.device), dim=-1))
                else:
                    recon_loss, kld = loss_function(torch.FloatTensor(X).to(self.args.device), recon, mu, logvar, self.args.dist)
                if self.args.anneal_steps > 0:
                    anneal = min(self.args.anneal_cap, 1. * update_count / self.args.anneal_steps)
                    update_count  += 1
                else:
                    anneal = self.args.anneal_cap
                if self.args.baseline:
                    vae_loss = recon_loss
                else:
                    vae_loss = recon_loss + anneal * kld
                vae_loss.backward()

                if self.args.baseline:
                    optimizer.step()
                else:
                    optimizer["encoder"].step()
                    optimizer["decoder"].step()

                if self.args.condition and self.args.joint_train:
                    optimizer["RNNEncoder"].step()
                # r20, r50, ndcg, rmse = self.test(model, anneal)
                # tensorboard
                if self.args.condition:
                    writer.add_scalar("Train rnn loss", rnn_loss, i + e*N/self.args.batch_size)
                writer.add_scalar("Train vae loss", vae_loss, i + e*N/self.args.batch_size)
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
            torch.save(model.state_dict(), self.args.log_dir + '/' + self.args.timestamp + '_' + self.args.config + '/model.pt')
            print("model saved!")
            print(f"epoch : {e} | train vae loss : {total_loss / (N/self.args.batch_size):.3} ")
            if self.args.condition:
                #save condition per epoch for evaluation
                for j in range(self.args.class_num):
                    hidden = "h_{}".format(j+1)
                    torch.save(self.hidden_vecs[hidden], f"{self.args.hiddenvec_dir}/{hidden}.pt")
                print("hidden vector saved!")
            # test per epoch
            r10, r20, r50, ndcg10, ndcg50, ndcg100 = self.test(model, anneal)
            # tensorboard
            # writer.add_scalar("Test_loss", test_loss, e)
            writer.add_scalar("Test_Recall10", r10, e)
            writer.add_scalar("Test_Recall20", r20, e)
            writer.add_scalar("Test_Recall50", r50, e)
            writer.add_scalar("Test_NDCG10", ndcg10, e)
            writer.add_scalar("Test_NDCG50", ndcg50, e)
            writer.add_scalar("Test_NDCG100", ndcg100, e)

    def test(self, model, anneal):
        model.eval()
        dataloader = ItemRatingLoader(self.args.data_dir)
        
        # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed2_valid_tr.csv'), os.path.join(self.args.data_dir, 'fixed2_valid_te.csv'))
        tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'valid_tr.csv'), os.path.join(self.args.data_dir, 'valid_te.csv'))
        N = tr_data_rating.shape[0]
        idxlist = np.array(range(N))
        np.random.seed(98765)
        idx_pe = np.random.permutation(len(idxlist))
        idxlist = idxlist[idx_pe]
        if self.args.condition:  
            valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'valid', self.args.batch_size, idx_pe)
        
        test_loss = 0 
        with torch.no_grad():
            r20_list, r50_list,r10_list, ndcg_list50, ndcg_list10, ndcg_list100 = [], [], [], [], [],[]
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition:
                    order, item_feature, label = next(valid_data_item)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    x_tr_unorder = tr_data_rating[idxlist[st_idx:end_idx]]
                    X_tr = x_tr_unorder[order]
                    x_te_unorder = te_data_rating[idxlist[st_idx:end_idx]]
                    X_te = x_te_unorder[order]
                else:
                    end_idx = min(st_idx + self.args.batch_size, N)
                    X_tr = tr_data_rating[idxlist[st_idx:end_idx]]
                    X_te = te_data_rating[idxlist[st_idx:end_idx]]
                
                if sparse.isspmatrix(X_tr):
                    X_tr = X_tr.toarray()
                X_tr = X_tr.astype('float32')

                if self.args.condition:
                    if self.args.test_hidden == 'trained':
                        h = []
                        for b_c in label:
                            for j in range(self.args.class_num):
                                hidden = "h_{}".format(j+1)
                                if b_c == i:
                                    h.append(self.hidden_vecs[hidden].unsqueeze(0))
                        hidden = torch.cat(h, 0)
                    elif self.args.test_hidden == 'onehot':
                        hidden = self.tooh(label, self.args.class_num).to(self.args.device)
                    else:
                        _, hidden = model.RNNEncoder(item_feature.to(self.args.device))
                    model_input = (torch.FloatTensor(X_tr).to(self.args.device), hidden)
                    recon, mu, logvar = model(model_input)
                else:
                    if not self.args.baseline:
                        recon, mu, logvar = model(torch.FloatTensor(X_tr).to(self.args.device))
                    else:
                        recon = model(torch.FloatTensor(X_tr).to(self.args.device))
                if not self.args.baseline: 
                    recon_loss, kld = loss_function(torch.FloatTensor(X_tr).to(self.args.device), recon, mu, logvar, self.args.dist)
                    loss = recon_loss + anneal * kld
                    test_loss += loss

                recon[X_tr.nonzero()] = -np.inf
                # print(Precision_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
                ndcg_list50.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                ndcg_list100.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=100))
                ndcg_list10.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
                r20_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=20))
                r50_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                r10_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
            
            ndcg_list50 = np.concatenate(ndcg_list50)
            ndcg_list100 = np.concatenate(ndcg_list100)
            ndcg_list10 = np.concatenate(ndcg_list10)
            r20_list = np.concatenate(r20_list)
            r10_list = np.concatenate(r10_list)
            r50_list = np.concatenate(r50_list)       

            if not self.args.baseline:
                print(f"test loss : {test_loss / (N/self.args.batch_size):.3}")
            print("Test NDCG@10=%.5f (%.5f)" % (ndcg_list10.mean(), np.std(ndcg_list10) / np.sqrt(len(ndcg_list10))))
            print("Test NDCG@50=%.5f (%.5f)" % (ndcg_list50.mean(), np.std(ndcg_list50) / np.sqrt(len(ndcg_list50))))
            print("Test Recall@20=%.5f (%.5f)" % (r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))))
            print("Test Recall@50=%.5f (%.5f)" % (r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))))
        return np.mean(r10_list), np.mean(r20_list), np.mean(r50_list), np.mean(ndcg_list10), np.mean(ndcg_list50),  np.mean(ndcg_list100)
        # return test_loss / (N/self.args.batch_size), np.mean(r20_list), np.mean(r50_list), np.mean(ndcg_list)
    
    def test_testset(self):
        if self.args.baseline == 'DAE':
            p_dims = [self.args.latent_size, self.args.hidden_size_vae, self.args.input_size_vae]
            model = MultiDAE(p_dims)
        else:
            model = VAE_RNN_rec(self.args.dims, self.args.input_size_rnn, self.args.embedding_size,\
                                self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                                self.args.hidden_size_rnn, self.args.condition, self.args.data_dir, self.args.activation)
        model = model.to(self.args.device)

        model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))
        dataloader = ItemRatingLoader(self.args.data_dir)
        
        # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed_test_tr.csv'), os.path.join(self.args.data_dir, 'fixed_test_te.csv'))
        tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'test_tr.csv'),\
                                                                     os.path.join(self.args.data_dir, 'test_te.csv'))
        N = tr_data_rating.shape[0]
        idxlist = np.array(range(N))

        np.random.seed(98765)
        idx_pe = np.random.permutation(len(idxlist))
        idxlist = idxlist[idx_pe]
        if self.args.condition:  
            valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'test', self.args.batch_size, idx_pe)
        
        with torch.no_grad():
            r20_list, r50_list,r10_list, ndcg_list50, ndcg_list10, ndcg_list100 = [], [], [], [], [],[]
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition:
                    order, item_feature, label = next(valid_data_item)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    x_tr_unorder = tr_data_rating[idxlist[st_idx:end_idx]]
                    X_tr = x_tr_unorder[order]
                    x_te_unorder = te_data_rating[idxlist[st_idx:end_idx]]
                    X_te = x_te_unorder[order]
                else:
                    end_idx = min(st_idx + self.args.batch_size, N)
                    
                    X_tr = tr_data_rating[idxlist[st_idx:end_idx]]
                    X_te = te_data_rating[idxlist[st_idx:end_idx]]
                
                if sparse.isspmatrix(X_tr):
                    X_tr = X_tr.toarray()
                X_tr = X_tr.astype('float32')

                if self.args.condition:
                    if self.args.test_hidden == 'trained':
                        h = []
                        for b_c in label:
                            for j in range(self.args.class_num):
                                hidden = "h_{}".format(j+1)
                                if b_c == i:
                                    h.append(self.hidden_vecs[hidden].unsqueeze(0))
                        hidden = torch.cat(h, 0)
                    elif self.args.test_hidden == 'onehot':
                        hidden = self.tooh(label, self.args.class_num).to(self.args.device)
                    else:
                        _, hidden = model.RNNEncoder(item_feature.to(self.args.device))
                    model_input = (torch.FloatTensor(X_tr).to(self.args.device), hidden)
                    recon, _, _ = model(model_input)
                else:
                    if not self.args.baseline:
                        recon, mu, logvar = model(torch.FloatTensor(X_tr).to(self.args.device))
                    else:
                        recon = model(torch.FloatTensor(X_tr).to(self.args.device))
                # if not self.args.baseline: 
                    # recon_loss, kld = loss_function(torch.FloatTensor(X_tr).to(self.args.device), recon, mu, logvar, self.args.dist)

                recon[X_tr.nonzero()] = -np.inf
                ndcg_list50.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                ndcg_list100.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=100))
                ndcg_list10.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
                r20_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=20))
                r50_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                r10_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
                
            ndcg_list50 = np.concatenate(ndcg_list50)
            ndcg_list100 = np.concatenate(ndcg_list100)
            ndcg_list10 = np.concatenate(ndcg_list10)
            r20_list = np.concatenate(r20_list)
            r10_list = np.concatenate(r10_list)
            r50_list = np.concatenate(r50_list)       

            # if not self.args.baseline:
                # print(f"test loss : {test_loss / (N/self.args.batch_size):.3}")
            print("Test NDCG@10=%.5f (%.5f)" % (ndcg_list10.mean(), np.std(ndcg_list10) / np.sqrt(len(ndcg_list10))))
            print("Test NDCG@50=%.5f (%.5f)" % (ndcg_list50.mean(), np.std(ndcg_list50) / np.sqrt(len(ndcg_list50))))
            print("Test NDCG@100=%.5f (%.5f)" % (ndcg_list100.mean(), np.std(ndcg_list100) / np.sqrt(len(ndcg_list100))))
            print("Test Recall@10=%.5f (%.5f)" % (r10_list.mean(), np.std(r10_list) / np.sqrt(len(r10_list))))
            print("Test Recall@20=%.5f (%.5f)" % (r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))))
            print("Test Recall@50=%.5f (%.5f)" % (r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))))


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
                                self.args.hidden_size_rnn, self.args.condition, self.args.data_dir, self.args.activation)
        model = model.to(self.args.device)

        if self.args.load_model:
            model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))

        dataloader = ItemRatingLoader(self.args.data_dir)
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
                hidden = "h_{}".format(i+1)
                if label[i] == k:
                    self.hidden_vecs[hidden] = (self.hidden_vecs[hidden] + h[i].squeeze()) / 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', default=[14984, 300, 100], type=list)
    parser.add_argument('--input-size_rnn', default=4096, type=int)
    parser.add_argument('--hidden-size_rnn', default=50, type=int)
    parser.add_argument('--embedding-size', default=500, type=int)
    parser.add_argument('--seq-len', default=20, type=int)
    parser.add_argument('--num-layer', default=1, type=int)
    parser.add_argument('--dropout-rate', default=0.1, type=float)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--num-workers', default=15, type=int)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--anneal-steps', default=200000, type=int)
    parser.add_argument('--anneal-cap', default=0.2, type=float)
    parser.add_argument('--class-num', default=5, type=int)
    parser.add_argument('--lr_rnn', default=0.001, type=float)
    parser.add_argument('--lr_vae', default=0.001, type=float)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dist', default='multi', type=str)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--data-dir', default='./data/amazon', type=str)
    parser.add_argument('--log-dir', default='./result/amazon/joint_train', type=str)
    parser.add_argument('--pretrained-dir',default='./result/amazon/pretrain', type=str)
    parser.add_argument('--hiddenvec-dir',default='./pretrained/amazon/joint', type=str)
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
    if args.data_dir == './data/amazon':
        args.class_num = 5
        args.input_size_rnn = 4096
    elif args.data_dir == './data/ml-1m':
        args.class_num = 3
        args.input_size_rnn = 1128

    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config_list = [args.embedding_size, args.seq_len, args.hidden_size_rnn, args.bidirectional, args.lr_vae, \
                        args.lr_rnn, args.dist, args.condition, args.anneal_cap, args.baseline, args.memo]
    print("using",args.device)
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    
    trainer = Trainer(args)
    if args.test:
        trainer.test_testset()
    else:
        trainer.train()
    for i in args.class_num:
        result = trainer.check_condition(i)
        for j in args.class_num:
            nd_name = "ndcg_list50_{}".format(j+1)
            print(f"{i}-{j} {result[nd_name].mean()}")