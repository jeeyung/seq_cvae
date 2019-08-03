import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from dataloader import *
from model import *
import time 
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
import torch
from datetime import datetime
from loss import *
from evaluation import *

class Trainer(object):
    def __init__(self, args):
        self.args = args
        if self.args.condition:
            self.h_1 = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)
            self.h_2 = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)
            self.h_3 = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)
            self.h_4 = torch.zeros(self.args.hidden_size_rnn).to(self.args.device)


    def train(self):
        writer = SummaryWriter(self.args.log_dir + "/" + self.args.timestamp + "_" + self.args.config)
        model = VAE_RNN_rec(self.args.input_size_vae, self.args.input_size_rnn, self.args.embedding_size,\
                            self.args.num_layer, self.args.dropout_rate, self.args.bidirectional, self.args.class_num,\
                            self.args.hidden_size_vae, self.args.hidden_size_rnn, self.args.latent_size, self.args.condition)
        model = model.to(self.args.device)
        train_F = True
        if self.args.condition:
            train_loader = JointDataLoader(self.args.data_dir, self.args.batch_size, train_F, self.args.test_ratio, \
                                                self.args.eval_ratio, self.args.seq_len, self.args.num_workers)
        else:
            train_loader = RatingDataLoader(self.args.data_dir, self.args.batch_size, train_F, self.args.test_ratio, self.args.eval_ratio)
        optimizer = {
            'encoder' : optim.RMSprop(model.encoder.parameters(), lr=self.args.lr_vae, alpha=0.99, eps=1e-08),
            'decoder' : optim.RMSprop(model.decoder.parameters(), lr=self.args.lr_vae, alpha=0.99, eps=1e-08)
        }
        if self.args.condition:
            optimizer['RNNEncoder'] = optim.RMSprop(model.RNNEncoder.parameters(), lr=self.args.lr_rnn, alpha=0.99, eps=1e-08)
            weight = torch.Tensor([0.65, 0.25, 0.08, 0.02]).to(self.args.device)
            CEloss = nn.CrossEntropyLoss(weight = weight)

        if self.args.load_model:
            model.load_state_dict(torch.load(self.args.log_dir + '/' + self.args.load_model + '/' + 'model.pt'))
            self.args.timestamp = self.args.load_model[:10]
        if self.args.condition and self.args.load_pretrained:
            model.RNNEncoder.load_state_dict(torch.load(self.args.pretrained_dir + '/' + self.args.load_pretrained + '/' + 'model.pt'))
            print("loaded pretrained model")

        update_count = 0.0
        for e in range(self.args.epoch):
            total_loss = 0
            r20_list, r50_list, ndcg_list = [], [], []
            for i, batch in enumerate(train_loader):
                if self.args.condition:
                    optimizer["RNNEncoder"].zero_grad()
                    output, h = model.RNNEncoder(batch["item_feature"].to(self.args.device))
                    rnn_loss = CEloss(output, batch["label"].to(self.args.device))
                    rnn_loss.backward(retain_graph=True)
                    # rnn_loss.backward()
                    optimizer["RNNEncoder"].step()
                    self.make_condition(h, batch["label"].data)
                
                optimizer["encoder"].zero_grad()
                optimizer["decoder"].zero_grad()
                if self.args.condition and self.args.joint_train:
                    optimizer["RNNEncoder"].zero_grad()
                
                if self.args.condition and self.args.avg_condition:
                    for b_c in batch["label"]:
                        if b_c == 1:
                            h.append(self.h_1.unsqueeze(0))
                        elif b_c == 2:
                            h.append(self.h_2.unsqueeze(0))
                        elif b_c == 3:
                            h.append(self.h_3.unsqueeze(0))
                        else:
                            h.append(self.h_4.unsqueeze(0))
                    h = torch.cat(h, 0)
                if self.args.condition:
                    model_input = (batch["rating"].to(self.args.device), h)
                    recon, mu, logvar = model(model_input)
                else:
                    recon, mu, logvar = model(batch["rating"].to(self.args.device))
                # print(recon)
                recon_loss, kld = loss_function(batch["rating"].to(self.args.device), recon, mu, logvar, self.args.dist)
                if self.args.anneal_steps > 0:
                    anneal = min(self.args.anneal_cap, 1. * update_count / self.args.anneal_steps)
                    update_count  += 1
                else:
                    anneal = self.args.anneal_cap
                
                vae_loss = recon_loss + anneal * kld
                vae_loss.backward()
                optimizer["encoder"].step()
                optimizer["decoder"].step()

                if self.args.condition and self.args.joint_train:
                    optimizer["RNNEncoder"].step()
                # tensorboard
                if self.args.condition:
                    writer.add_scalar("Train rnn loss", rnn_loss, i + e*len(train_loader))
                writer.add_scalar("Train vae loss", vae_loss, i + e*len(train_loader))
                writer.add_scalar("Recon loss", recon_loss, i + e*len(train_loader))
                writer.add_scalar("KLD", kld, i + e*len(train_loader))
                # for evaluation
                ndcg_list.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), batch["rating"].cpu().detach().numpy(), k=100))
                r20_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), batch["rating"].cpu().detach().numpy(), k=20))
                r50_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), batch["rating"].cpu().detach().numpy(), k=50))
                
                if i % 20 == 0:
                    print(f"recon : {recon_loss.item():.3} | kld : {kld.item():.3}")
                    print(f"epoch : {e} | train_vae_loss : {vae_loss.item():.3}", 
                    f"[{i*self.args.batch_size} / {len(train_loader)*self.args.batch_size}","(",f"{(i/len(train_loader))*100:.3} %", ")]")
                total_loss += vae_loss
            # save model
            torch.save(model.state_dict(), self.args.log_dir + '/' + self.args.timestamp + '_' + self.args.config + '/model.pt')
            print("model saved!")
            # evaluation
            ndcg_list = np.concatenate(ndcg_list)
            r20_list = np.concatenate(r20_list)
            r50_list = np.concatenate(r50_list)        
            print(f"epoch : {e} | train vae loss : {total_loss / len(train_loader):.3} ")
            print("Train NDCG@100=%.5f (%.5f)" % (np.mean(ndcg_list), np.std(ndcg_list) / np.sqrt(len(ndcg_list))))
            print("Train Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
            print("Train Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
            if self.args.condition:
                #save condition per epoch for evaluation
                torch.save(self.h_1, f"./pretrained/joint/h_1.pt")
                torch.save(self.h_2, f"./pretrained/joint/h_2.pt")
                torch.save(self.h_3, f"./pretrained/joint/h_3.pt")
                torch.save(self.h_4, f"./pretrained/joint/h_4.pt")
            # test per epoch
            test_loss, r20, r50, ndcg = self.test(model, anneal)
            # tensorboard
            writer.add_scalar("Test_loss", test_loss, e)
            writer.add_scalar("Test_Recall20", r20, e)
            writer.add_scalar("Test_Recall50", r50, e)
            writer.add_scalar("Test_NDCG", ndcg, e)
        
    def test(self,model, anneal):
        train_F = False
        if self.args.condition:
            test_loader = JointDataLoader(self.args.data_dir, self.args.batch_size, train_F, self.args.test_ratio, \
                                                self.args.eval_ratio, self.args.seq_len, self.args.num_workers) 
        else:
            test_loader = RatingDataLoader(self.args.data_dir, self.args.batch_size, train_F, self.args.test_ratio, self.args.eval_ratio)
        test_loss = 0 

        with torch.no_grad():
            r20_list, r50_list, ndcg_list = [], [], []

            for i, batch in enumerate(test_loader):
                if self.args.condition:
                    h = []
                    for b_c in batch["label"]:
                        if b_c == 1:
                            h.append(self.h_1.unsqueeze(0))
                        elif b_c == 2:
                            h.append(self.h_2.unsqueeze(0))
                        elif b_c == 3:
                            h.append(self.h_3.unsqueeze(0))
                        else:
                            h.append(self.h_4.unsqueeze(0))
                    hidden = torch.cat(h, 0)
                    model_input = (batch["test_te"].to(self.args.device), hidden)
                    recon, mu, logvar = model(model_input)
                else:
                    recon, mu, logvar = model(batch["test_te"].to(self.args.device))
                recon_loss, kld = loss_function(batch["test_te"].to(self.args.device), recon, mu, logvar, self.args.dist)
                loss = recon_loss + anneal * kld
                test_loss += loss

                ndcg_list.append(NDCG_binary_at_k_batch(recon.cpu().detach().numpy(), batch["test_eval"].cpu().detach().numpy(), k=100))
                r20_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), batch["test_eval"].cpu().detach().numpy(), k=20))
                r50_list.append(Recall_at_k_batch(recon.cpu().detach().numpy(), batch["test_eval"].cpu().detach().numpy(), k=50))
            
            ndcg_list = np.concatenate(ndcg_list)
            r20_list = np.concatenate(r20_list)
            r50_list = np.concatenate(r50_list)                

            print(f"test loss : {test_loss / len(test_loader):.3}")
            print("Test NDCG@100=%.5f (%.5f)" % (np.mean(ndcg_list), np.std(ndcg_list) / np.sqrt(len(ndcg_list))))
            print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
            print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
        return test_loss/len(test_loader), np.mean(r20_list), np.mean(r50_list), np.mean(ndcg_list)

    def make_condition(self, h, label):
        for i in range(len(label)):
            if label[i] == 0:
                self.h_1 += h[i].squeeze()
            elif label[i] == 1:
                self.h_2 += h[i].squeeze()
            elif label[i] == 2:
                self.h_3 += h[i].squeeze()
            else:
                self.h_4 += h[i].squeeze()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size_vae', default=9731, type=int)
    parser.add_argument('--input-size_rnn', default=1128, type=int)
    parser.add_argument('--latent-size', default=20, type=int)
    parser.add_argument('--hidden-size_rnn', default=100, type=int)
    parser.add_argument('--hidden-size_vae', default=600, type=int)
    parser.add_argument('--embedding-size', default=200, type=int)
    parser.add_argument('--seq-len', default=20, type=int)
    parser.add_argument('--num-layer', default=1, type=int)
    parser.add_argument('--dropout-rate', default=0.1, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--num-workers', default=15, type=int)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--anneal-steps', default=200000, type=int)
    parser.add_argument('--anneal-cap', default=0.5, type=int)
    parser.add_argument('--class-num', default=4, type=int)
    parser.add_argument('--test-ratio', default=0.2, type=float)
    parser.add_argument('--eval-ratio', default=0.2, type=float)
    parser.add_argument('--lr_rnn', default=0.001, type=float)
    parser.add_argument('--lr_vae', default=0.01, type=float)
    # parser.add_argument('--ld', default=0.1, type=float)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dist', default='bern', type=str)
    parser.add_argument('--data-dir', default='./data/ml-20m', type=str)
    parser.add_argument('--log-dir', default='./result/ml-20m/joint_train', type=str)
    parser.add_argument('--pretrained-dir',default='./result/ml-20m/pretrain', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--memo', default='', type=str)
    parser.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M"), type=str)
    parser.add_argument('--condition', action="store_false")
    parser.add_argument('--joint-train', action="store_true")
    parser.add_argument('--load-model', default=None, type=str)
    parser.add_argument('--load-pretrained', default=None, type=str)
    parser.add_argument('--avg-condition', action='store_true')

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config_list = [args.embedding_size, args.seq_len, args.hidden_size_rnn, args.bidirectional, args.hidden_size_vae, args.latent_size, args.lr_vae, \
                        args.lr_rnn, args.dist, args.joint_train, args.avg_condition, args.anneal_cap]
    print("using",args.device)
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    
    trainer = Trainer(args)
    trainer.train()