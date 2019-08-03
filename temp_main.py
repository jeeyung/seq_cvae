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
from data3 import ItemRatingLoader
# from model_baseline import *
''' It is the similar to main_renew2.py'''

class Trainer(object):
    def __init__(self, args):
        self.args = args
        if self.args.condition:
            self.h_1 = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)
            self.h_2 = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)
            self.h_3 = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)
            # self.h_4 = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)

    def train(self):
        ### DAE
        if self.args.baseline == 'DAE':
            p_dims = [self.args.latent_size, self.args.hidden_size_vae, self.args.input_size_vae]
            model = MultiDAE(p_dims)
            optimizer = optim.Adam(model.parameters(), lr = self.args.lr_vae, weight_decay=0.0)
        else:
            model = VAE_RNN_rec(self.args.input_size_vae, self.args.input_size_rnn, self.args.embedding_size,\
                                self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                                self.args.hidden_size_vae, self.args.hidden_size_rnn, self.args.latent_size, \
                                    self.args.condition, self.args.data_dir, self.args.pretrain)
            optimizer = {
            'encoder' : optim.Adam(model.encoder.parameters(), lr=self.args.lr_vae, weight_decay=0.0),
            'decoder' : optim.Adam(model.decoder.parameters(), lr=self.args.lr_vae, weight_decay=0.0)
            }
        
        model = model.to(self.args.device)
        # train_F = True
        dataloader = ItemRatingLoader(self.args.data_dir)
        
        if self.args.condition:
            optimizer['RNNEncoder'] = optim.Adam(model.RNNEncoder.parameters(), lr=self.args.lr_rnn, weight_decay=0.0)
            # weight = torch.FloatTensor([0.96, 0.023, 0.017]).to(args.device)
            weight = torch.FloatTensor([0.18, 0.28, 0.54]).to(args.device)
            CEloss = nn.CrossEntropyLoss(weight = weight)
            # CEloss = nn.CrossEntropyLoss()

        if self.args.load_model:
            model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))
            self.args.timestamp = self.args.load_model[:10]
        if self.args.condition and self.args.load_pretrained:
            model.RNNEncoder.load_state_dict(torch.load(self.args.pretrained_dir + '/' + self.args.load_pretrained + '/' + 'model.pt'))
            print("loaded pretrained model")
        
        writer = SummaryWriter(self.args.log_dir + "/" + self.args.timestamp + "_" + self.args.config)
        train_data_rating = dataloader.load_train_data(os.path.join(self.args.data_dir, 'train.csv'))
        # train_data_rating = dataloader.load_train_data(os.path.join(self.args.data_dir, 'small_train.csv'))
        # train_data_rating = dataloader.load_train_data(os.path.join(self.args.data_dir, 'ver2/train.csv'))
        N = train_data_rating.shape[0]
        
        idxlist = np.array(range(N))

        # np.random.seed(98765)
        idx_pe = np.random.permutation(len(idxlist))
        idxlist = idxlist[idx_pe]

        update_count = 0.0
        for e in range(self.args.epoch):
            model.train()
            total_loss = 0
            train_data_item = dataloader.fixed_hidden(int(N/args.batch_size)+1, 'train', args.batch_size, idx_pe)
            # if self.args.condition:
                # train_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'train', self.args.batch_size, idx_pe)
            
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition:
                    h = next(train_data_item)
                    # print(h.shape)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    X = train_data_rating[idxlist[st_idx:end_idx]]
                    # X = x_unorder[order]
                else:
                    end_idx = min(st_idx + self.args.batch_size, N)
                    X = train_data_rating[idxlist[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')   

                # if self.args.condition:
                    # optimizer["RNNEncoder"].zero_grad()
                    # output, _ = model.RNNEncoder(item_feature.to(self.args.device))
                    # rnn_loss = CEloss(output, label.to(self.args.device))
                    # rnn_loss.backward(retain_graph=True)
                    # rnn_loss.backward()
                    # optimizer["RNNEncoder"].step()
                    # self.make_condition(h, label.data)
                    # self.make_condition(h, label.data)
                
                if self.args.baseline:
                    optimizer.zero_grad()
                else:
                    optimizer["encoder"].zero_grad()
                    optimizer["decoder"].zero_grad()
                
                # if self.args.condition and self.args.joint_train:
                #     optimizer["RNNEncoder"].zero_grad()
                
                # if self.args.condition and self.args.avg_condition:
                #     h=[]
                #     for b_c in label:
                #         if b_c == 0:
                #             h.append(self.h_1.unsqueeze(0))
                #         elif b_c == 1:
                #             h.append(self.h_2.unsqueeze(0))
                #         else:
                #             h.append(self.h_3.unsqueeze(0))
                #     h = torch.cat(h, 0)
                    # print(h.shape)
                if self.args.condition:
                    # h = self.tooh(label, 3).to(self.args.device)
                    # print(h.shape)
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

                # if self.args.condition and self.args.joint_train:
                    # optimizer["RNNEncoder"].step()
                # r20, r50, ndcg, rmse = self.test(model, anneal)
                # tensorboard
                # if self.args.condition:
                    # writer.add_scalar("Train rnn loss", rnn_loss, i + e*N/self.args.batch_size)
                writer.add_scalar("Train vae loss", vae_loss, i + e*N/self.args.batch_size)
                writer.add_scalar("Recon loss", recon_loss, i + e*N/self.args.batch_size)
                if not self.args.baseline:
                    writer.add_scalar("KLD", kld, i + e*N/self.args.batch_size)
                
                if i % 20 == 0:
                    if not self.args.baseline:
                        print(f"recon : {recon_loss.item():.3} | kld : {kld.item():.3}")
                    if self.args.condition:
                        # print(f"epoch : {e} | train_vae_loss : {vae_loss.item():.3} | train_rnn_loss : {rnn_loss.item():.3}", 
                        # f"[{i*self.args.batch_size} / {N}","(",f"{(i/N*self.args.batch_size)*100:.3} %", ")]")
                        pass
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
                torch.save(self.h_1, f"./pretrained/ml-1m/joint/h_1.pt")
                torch.save(self.h_2, f"./pretrained/ml-1m/joint/h_2.pt")
                torch.save(self.h_3, f"./pretrained/ml-1m/joint/h_3.pt")
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
        # train_F = False
        dataloader = ItemRatingLoader(self.args.data_dir)
        
        # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed_valid_tr.csv'), os.path.join(self.args.data_dir, 'fixed_valid_te.csv'))
        tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed2_valid_tr.csv'), os.path.join(self.args.data_dir, 'fixed2_valid_te.csv'))
        # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'ver2/validation_tr.csv'), os.path.join(self.args.data_dir, 'ver2/validation_te.csv'))
        N = tr_data_rating.shape[0]
        idxlist = np.array(range(N))
        # print(len(idxlist))
        np.random.seed(98765)
        idx_pe = np.random.permutation(len(idxlist))
        idxlist = idxlist[idx_pe]
        if self.args.condition:  
            # valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'valid', self.args.batch_size, idx_pe)
            valid_data_item = dataloader.fixed_hidden(int(N/args.batch_size)+1, 'valid', args.batch_size, idx_pe)
        # np.random.shuffle(idxlist)
        test_loss = 0 
        # test_rmse = 0
        with torch.no_grad():
            r20_list, r50_list,r10_list, ndcg_list50, ndcg_list10, ndcg_list100 = [], [], [], [], [],[]
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition:
                    hidden = next(valid_data_item)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    X_tr= tr_data_rating[idxlist[st_idx:end_idx]]
                    # X_tr = x_tr_unorder[order]
                    X_te= te_data_rating[idxlist[st_idx:end_idx]]
                    # X_te = x_te_unorder[order]
                else:
                    end_idx = min(st_idx + self.args.batch_size, N)
                    
                    X_tr = tr_data_rating[idxlist[st_idx:end_idx]]
                    X_te = te_data_rating[idxlist[st_idx:end_idx]]
                
                if sparse.isspmatrix(X_tr):
                    X_tr = X_tr.toarray()
                X_tr = X_tr.astype('float32')

                if self.args.condition:
                    # _, hidden = model.RNNEncoder(item_feature.to(self.args.device))
                    # h = []
                    # for b_c in label:
                    #     if b_c == 0:
                    #         h.append(self.h_1.unsqueeze(0))
                    #     elif b_c == 1:
                    #         h.append(self.h_2.unsqueeze(0))
                    #     else:
                    #         h.append(self.h_3.unsqueeze(0))
                    # hidden = torch.cat(h, 0)
                    # hidden = self.tooh(label, 3).to(self.args.device)
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
            print("Test NDCG@50=%.5f (%.5f)" % (ndcg_list50.mean(), np.std(ndcg_list50) / np.sqrt(len(ndcg_list50))))
            print("Test NDCG@10=%.5f (%.5f)" % (ndcg_list10.mean(), np.std(ndcg_list10) / np.sqrt(len(ndcg_list10))))
            print("Test Recall@20=%.5f (%.5f)" % (r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))))
            print("Test Recall@50=%.5f (%.5f)" % (r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))))
            # print(f"Test rmse : {test_rmse / (N/self.args.batch_size):.3}")
        return np.mean(r10_list), np.mean(r20_list), np.mean(r50_list), np.mean(ndcg_list10), np.mean(ndcg_list50),  np.mean(ndcg_list100)
        # return np.mean(r20_list), np.mean(r50_list), np.mean(ndcg_list)
        # return test_loss / (N/self.args.batch_size), np.mean(r20_list), np.mean(r50_list), np.mean(ndcg_list)
    
    def test_testset(self):
        if self.args.baseline == 'DAE':
            p_dims = [self.args.latent_size, self.args.hidden_size_vae, self.args.input_size_vae]
            model = MultiDAE(p_dims)
        else:
            model = VAE_RNN_rec(self.args.input_size_vae, self.args.input_size_rnn, self.args.embedding_size,\
                                    self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                                    self.args.hidden_size_vae, self.args.hidden_size_rnn, self.args.latent_size, \
                                        self.args.condition, self.args.data_dir, self.args.pretrain)
        model = model.to(self.args.device)

        model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))
        dataloader = ItemRatingLoader(self.args.data_dir)
        
        tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed2_test_tr.csv'), os.path.join(self.args.data_dir, 'fixed2_test_te.csv'))
        # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed_test_tr.csv'), os.path.join(self.args.data_dir, 'fixed_test_te.csv'))
        # tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'ver2/validation_tr.csv'), os.path.join(self.args.data_dir, 'ver2/validation_te.csv'))
        N = tr_data_rating.shape[0]
        idxlist = np.array(range(N))

        np.random.seed(98765)
        idx_pe = np.random.permutation(len(idxlist))
        idxlist = idxlist[idx_pe]
        if self.args.condition:  
            # valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'test', self.args.batch_size, idx_pe)
            valid_data_item = dataloader.fixed_hidden(int(N/args.batch_size)+1, 'test', args.batch_size, idx_pe)
        
        # np.random.shuffle(idxlist)
        test_loss = 0 
        # test_rmse = 0
        with torch.no_grad():
            r20_list, r50_list,r10_list, ndcg_list50, ndcg_list10, ndcg_list100 = [], [], [], [], [],[]
            for i, st_idx in enumerate(range(0, N, self.args.batch_size)):
                if self.args.condition:
                    hidden = next(valid_data_item)
                    end_idx = min(st_idx + self.args.batch_size, N)
                    X_tr = tr_data_rating[idxlist[st_idx:end_idx]]
                    # X_tr = x_tr_unorder[order]
                    X_te = te_data_rating[idxlist[st_idx:end_idx]]
                    # X_te = x_te_unorder[order]
                else:
                    end_idx = min(st_idx + self.args.batch_size, N)
                    
                    X_tr = tr_data_rating[idxlist[st_idx:end_idx]]
                    X_te = te_data_rating[idxlist[st_idx:end_idx]]
                
                if sparse.isspmatrix(X_tr):
                    X_tr = X_tr.toarray()
                X_tr = X_tr.astype('float32')

                if self.args.condition:
                    # h = []
                    # for b_c in label:
                    #     if b_c == 0:
                    #         h.append(self.h_1.unsqueeze(0))
                    #     elif b_c == 1:
                    #         h.append(self.h_2.unsqueeze(0))
                    #     else:
                    #         h.append(self.h_3.unsqueeze(0))
                    # hidden = torch.cat(h, 0)
                    # hidden = self.tooh(label, 3).to(self.args.device)
                    # _, hidden = model.RNNEncoder(item_feature.to(self.args.device))
                    model_input = (torch.FloatTensor(X_tr).to(self.args.device), hidden)
                    recon, mu, logvar = model(model_input)
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
            print("Test NDCG@50=%.5f (%.5f)" % (ndcg_list50.mean(), np.std(ndcg_list50) / np.sqrt(len(ndcg_list50))))
            print("Test NDCG@10=%.5f (%.5f)" % (ndcg_list10.mean(), np.std(ndcg_list10) / np.sqrt(len(ndcg_list10))))
            print("Test NDCG@100=%.5f (%.5f)" % (ndcg_list100.mean(), np.std(ndcg_list100) / np.sqrt(len(ndcg_list100))))
            print("Test Recall@20=%.5f (%.5f)" % (r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))))
            print("Test Recall@50=%.5f (%.5f)" % (r50_list.mean(), np.std(r50_list) / np.sqrt(len(r50_list))))
            print("Test Recall@10=%.5f (%.5f)" % (r10_list.mean(), np.std(r10_list) / np.sqrt(len(r10_list))))



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
        model = VAE_RNN_rec(self.args.input_size_vae, self.args.input_size_rnn, self.args.embedding_size,\
                                self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                                self.args.hidden_size_vae, self.args.hidden_size_rnn, self.args.latent_size, \
                                    self.args.condition, self.args.data_dir, self.args.pretrain)
        model = model.to(self.args.device)

        if self.args.load_model:
            model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))

        dataloader = ItemRatingLoader(self.args.data_dir)
        tr_data_rating, te_data_rating = dataloader.load_tr_te_data(os.path.join(self.args.data_dir, 'fixed2_test_tr.csv'), os.path.join(self.args.data_dir, 'fixed2_test_te.csv'))
        N = tr_data_rating.shape[0]
        idxlist = np.array(range(N))
        # print(N)
        np.random.seed(98764)
        idx_pe = np.random.permutation(len(idxlist))

        idxlist = idxlist[idx_pe]
        # print(idxlist[:self.args.batch_size])
        # if self.args.condition:  
        valid_data_item = dataloader.load_sequence_data_generator(int(N/self.args.batch_size)+1, 'test', self.args.batch_size, idx_pe)
        ndcg_list50_1, ndcg_list50_2, ndcg_list50_3 = [],[],[]
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
                    self.h_1 = torch.load("./pretrained/ml-1m/joint/h_1.pt")
                    self.h_2 = torch.load("./pretrained/ml-1m/joint/h_2.pt")
                    self.h_3 = torch.load("./pretrained/ml-1m/joint/h_3.pt")
                    h1 = []
                    h2 = []
                    h3 = []
                    for _ in range(self.args.batch_size):
                        h1.append(self.h_1.unsqueeze(0))
                        h2.append(self.h_2.unsqueeze(0))
                        h3.append(self.h_3.unsqueeze(0))
                    
                    hidden1 = torch.cat(h1, 0)
                    hidden2 = torch.cat(h2, 0)
                    hidden3 = torch.cat(h3, 0)
                    # print(torch.FloatTensor(X_tr).to(self.args.device).shape)
                    # print(hidden1.shape)
                    model_input = (torch.FloatTensor(X_tr).to(self.args.device), hidden1)
                    recon1, _, _ = model(model_input)
                    topk1 = show_recommended_items(recon1.cpu().detach().numpy(), k=50)

                    model_input = (torch.FloatTensor(X_tr).to(self.args.device), hidden2)
                    recon2, _, _ = model(model_input)
                    topk2 = show_recommended_items(recon2.cpu().detach().numpy(), k=50)

                    model_input = (torch.FloatTensor(X_tr).to(self.args.device), hidden3)
                    recon3, _, _ = model(model_input)
                    topk3 = show_recommended_items(recon3.cpu().detach().numpy(), k=50)
                    with open('./result/ml-1m/qual/topk1.pkl', 'wb') as f:
                        pickle.dump(topk1, f)

                    with open('./result/ml-1m/qual/topk2.pkl', 'wb') as f:
                        pickle.dump(topk2, f)

                    with open('./result/ml-1m/qual/topk3.pkl', 'wb') as f:
                        pickle.dump(topk3, f)
                        print('save')
                    break
                    # with open('./result/ml-1m/qual/x_te.pkl', 'wb') as f:
                    #     pickle.dump(X_te.toarray().nonzero(), f)
                    # print(topk1, topk2, topk3, X_te.toarray().nonzero())
                    
                    recon1[X_tr.nonzero()] = -np.inf
                    recon2[X_tr.nonzero()] = -np.inf
                    recon3[X_tr.nonzero()] = -np.inf
                    ndcg_list50_1.append(Recall_at_k_batch(recon1.cpu().detach().numpy(), X_te, k=50))
                    ndcg_list50_2.append(Recall_at_k_batch(recon2.cpu().detach().numpy(), X_te, k=50))
                    ndcg_list50_3.append(Recall_at_k_batch(recon3.cpu().detach().numpy(), X_te, k=50))

                    # ndcg_list50_2_1.append(NDCG_binary_at_k_batch(recon1.cpu().detach().numpy(), X_te, k=50))
                    # ndcg_list50_2_2.append(NDCG_binary_at_k_batch(recon2.cpu().detach().numpy(), X_te, k=50))
                    # ndcg_list50_2_3.append(NDCG_binary_at_k_batch(recon3.cpu().detach().numpy(), X_te, k=50))

                    # ndcg_list50_3_1.append(NDCG_binary_at_k_batch(recon1.cpu().detach().numpy(), X_te, k=50))
                    # ndcg_list50_3_2.append(NDCG_binary_at_k_batch(recon2.cpu().detach().numpy(), X_te, k=50))
                    # ndcg_list50_3_3.append(NDCG_binary_at_k_batch(recon3.cpu().detach().numpy(), X_te, k=50))

                    # r20_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=20))
                    # r50_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=50))
                    # r10_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), X_te, k=10))
            
            ndcg_list50_1 = np.concatenate(ndcg_list50_1)
            ndcg_list50_2 = np.concatenate(ndcg_list50_2)
            ndcg_list50_3 = np.concatenate(ndcg_list50_3)
            # print(ndcg_list50_1)
            return ndcg_list50_1.mean(), ndcg_list50_2.mean(), ndcg_list50_3.mean()
                    # r20_list = np.concatenate(r20_list)
                    # r10_list = np.concatenate(r10_list)
                    # r50_list = np.concatenate(r50_list)
                    # print(ndcg_list50_1, ndcg_list50_2, ndcg_list50_3)

            # ndcg_list100 = np.concatenate(ndcg_list100)
            # ndcg_list10 = np.concatenate(ndcg_list10)
            # r20_list = np.concatenate(r20_list)
            # r10_list = np.concatenate(r10_list)
            # r50_list = np.concatenate(r50_list)  

    def make_condition(self, h, label):
        for i in range(len(label)):
            if label[i] == 0:
                self.h_1 = (self.h_1 +h[i].squeeze()) / 2
            elif label[i] == 1:
                self.h_2 = (self.h_2 +h[i].squeeze()) / 2
            else:
                self.h_3 = (self.h_3 +h[i].squeeze()) / 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input-size_vae', default=356, type=int)
    parser.add_argument('--input-size_vae', default=3355, type=int)
    # parser.add_argument('--input-size_vae', default=9731, type=int)
    parser.add_argument('--input-size_rnn', default=1128, type=int)
    parser.add_argument('--latent-size', default=50, type=int)
    parser.add_argument('--hidden-size_rnn', default=50, type=int)
    parser.add_argument('--hidden-size_vae', default=100, type=int)
    # parser.add_argument('--hidden-size_vae1', default=800, type=int)
    # parser.add_argument('--hidden-size_vae2', default=300, type=int)
    parser.add_argument('--embedding-size', default=300, type=int)
    parser.add_argument('--seq-len', default=20, type=int)
    parser.add_argument('--num-layer', default=1, type=int)
    parser.add_argument('--dropout-rate', default=0.1, type=float)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--num-workers', default=15, type=int)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--anneal-steps', default=200000, type=int)
    parser.add_argument('--anneal-cap', default=0.2, type=float)
    parser.add_argument('--class-num', default=3, type=int)
    # parser.add_argument('--test-ratio', default=0.2, type=float)
    # parser.add_argument('--eval-ratio', default=0.2, type=float)
    parser.add_argument('--lr_rnn', default=0.001, type=float)
    parser.add_argument('--lr_vae', default=0.001, type=float)
    # parser.add_argument('--ld', default=0.1, type=float)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dist', default='multi', type=str)
    parser.add_argument('--data-dir', default='./data/ml-1m', type=str)
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--log-dir', default='./result/ml-1m/joint_train', type=str)
    parser.add_argument('--pretrained-dir',default='./result/ml-1m/pretrain', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--memo', default='', type=str)
    parser.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M"), type=str)
    parser.add_argument('--condition', action="store_false")
    parser.add_argument('--joint-train', action="store_true")
    parser.add_argument('--load-model', default=None, type=str)
    parser.add_argument('--load-pretrained', default=None, type=str)
    parser.add_argument('--avg-condition', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--baseline', default=None, type=str)
    parser.add_argument('--test',  action='store_true')

        

    args = parser.parse_args()
    if args.baseline:
        args.condition = False
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config_list = [args.embedding_size, args.seq_len, args.hidden_size_rnn, args.bidirectional, args.hidden_size_vae,args.latent_size, args.lr_vae, \
                        args.lr_rnn, args.dist, args.joint_train, args.avg_condition, args.condition, args.anneal_cap, args.baseline, args.memo]
    print("using",args.device)
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    
    trainer = Trainer(args)
    if args.test:
        trainer.test_testset()
    else:
        trainer.train()
    # mean_1_1, mean_1_2, mean_1_3 = trainer.check_condition(0)
    # mean_2_1, mean_2_2, mean_2_3 = trainer.check_condition(1)
    # mean_3_1, mean_3_2, mean_3_3 = trainer.check_condition(2)
    # print(mean_1_1,mean_1_2,mean_1_3,mean_2_1,mean_2_2,mean_2_3,mean_3_1,mean_3_2, mean_3_3)
    # trainer.test_testset()