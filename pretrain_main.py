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

def train(args):
    writer = SummaryWriter(args.log_dir + "/" + args.timestamp + "_" + args.config)
    if args.data_dir == './data/ml-1m':
        model = RNNEncoder(args.input_size, args.embedding_size, args.hidden_size,\
                        args.num_layer, args.dropout_rate, args.bidirectional, args.class_num, args.dataset, args.condition_size)
        train_loader = ItemFeatureDataLoader_ml1m(args.data_dir, args.seq_len, args.batch_size, True, args.test_ratio)
    else:
        model = RNNEncoder_amazon(args.input_size, args.embedding_size, args.hidden_size,\
                            args.num_layer, args.dropout_rate, args.bidirectional, args.class_num, args.dataset, args.item_num, args.activation)
        train_loader = ItemFeatureDataLoader_amazon(args.data_dir, args.seq_len, args.batch_size, True, args.test_ratio)
    model = model.to(args.device)
    optimizer =  optim.Adam(model.parameters(), lr=args.lr)
    if args.data_dir == './data/ml-1m':
        weight = torch.FloatTensor([2.5, 3, 2]).to(args.device)
        CEloss = nn.CrossEntropyLoss(weight = weight)
    else:
        CEloss = nn.CrossEntropyLoss()
    
    for e in range(args.epoch):
        total_loss = 0
        total_correct = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            output, _ = model((batch["feature"]).to(args.device))
            loss = CEloss(output, (batch["label"]).to(args.device))
            loss.backward()
            optimizer.step()
            predicted = torch.argmax(output, 1)
            # print(batch["label"], predicted)
            correct = (predicted==batch["label"].to(args.device)).sum().item()
            # test_loss = test(args, model, CEloss)
            # print(correct)
            writer.add_scalar("Train loss", loss, i + e*len(train_loader))
            
            if i % 10 == 0:
                print(f"epoch : {e} | train_loss : {loss.item():.3}", 
                f"| {i*args.batch_size} / {len(train_loader)*args.batch_size}","(",f"{(i/len(train_loader))*100:.3} %", ")")
            total_correct += correct
            total_loss += loss
        
        print(f"epoch : {e} | train loss : {total_loss / len(train_loader):.3} | accuracy : {total_correct/(len(train_loader.dataset))*100:0.4} %")
        writer.add_scalar("Accuracy", total_correct/(len(train_loader.dataset))*100, e)
        test_loss = test(args, model, CEloss)
        writer.add_scalar("Test loss", test_loss, e)
        torch.save(model.state_dict(), args.log_dir + '/' + args.timestamp + '_' + args.config + '/model.pt')
        print("pretrained model saved")
        save_pretrained(args, model)
    
def test(args, model, CEloss):
    if args.data_dir == './data/ml-1m':
        test_loader= ItemFeatureDataLoader_ml1m(args.data_dir, args.seq_len, args.batch_size, True, args.test_ratio)
    else:
        test_loader = ItemFeatureDataLoader_amazon(args.data_dir, args.seq_len, args.batch_size, True, args.test_ratio)
    test_loss = 0 
    confusion_matrix = torch.zeros(args.class_num, args.class_num, dtype=torch.int32)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            output, _ = model(batch["feature"].to(args.device))
            loss = CEloss(output, batch["label"].to(args.device))
            test_loss += loss
            predicted = torch.argmax(output, 1)
            for t, p in zip(batch["label"].view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        print("confusion matrix\n", confusion_matrix)
        print(f"test loss : {test_loss / len(test_loader):.3}")
          
    return test_loss/len(test_loader)

def save_pretrained(args, model):
    if args.data_dir == './data/ml-1m':
        test_loader= ItemFeatureDataLoader_ml1m(args.data_dir, args.seq_len, 1, True, args.test_ratio)
    else:
        test_loader = ItemFeatureDataLoader_amazon(args.data_dir, args.seq_len, 1, True, args.test_ratio)
    hidden_vecs={}
    for i in range(args.class_num):
        hidden = "h_{}".format(i+1)
        hidden_vecs[hidden] = torch.zeros(args.condition_size).to(args.device)

    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            _, h = model(batch["feature"].to(args.device))
            for i in range(args.class_num):
                hidden = "h_{}".format(i+1)
                if batch["label"].data == i:
                    hidden_vecs[hidden] = (hidden_vecs[hidden] + h.squeeze()) / 2

    for j in range(args.class_num):
        hidden = "h_{}".format(j+1)
        torch.save(hidden_vecs[hidden], f"{args.pretrained_dir}/{hidden}.pt")
    print("pretrain hidden vector saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=1128, type=int)
    parser.add_argument('--embedding-size', default=300, type=int)
    parser.add_argument('--seq-len', default=20, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--condition-size', default=50, type=int)
    parser.add_argument('--item-num', default=33310, type=int)
    parser.add_argument('--num-layer', default=1, type=int)
    parser.add_argument('--dropout-rate', default=0.1, type=float)
    parser.add_argument('--class-num', default=3, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch-size', default=50, type=int)
    parser.add_argument('--test-ratio', default=0.2, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--dataset', default='amazon_min20_woman_fix', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--pretrain', action='store_false')
    parser.add_argument('--memo', default='', type=str)
    parser.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M"), type=str)

    args = parser.parse_args()
    args.data_dir = f'./data/{args.dataset}'
    args.log_dir = f'./result/{args.dataset}/pretrain'
    args.pretrained_dir = f'./pretrained/{args.dataset}'
    if args.dataset == 'amazon' or args.dataset == 'amazon_min20_woman' or args.dataset == 'amazon_min20_woman_fix' or args.dataset == 'amazon_min10_woman':
        args.class_num = 5
        args.input_size = 4096
        args.embedding_size = 500
        if args.dataset =='amazon_min10_woman':
            args.item_num = 31842 
    elif args.dataset == 'ml-1m':
        args.class_num = 3
        args.input_size = 1128
        args.item_num = 3355
        args.embedding_size =300
    
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("using ", args.device)
    config_list = [args.hidden_size, args.class_num, args.embedding_size, args.lr, args.bidirectional, args.dataset, args.memo]
    
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    
    train(args)