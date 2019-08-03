import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from dataloader import *
import time 
from dataloader import *
from torch.distributions.multivariate_normal import MultivariateNormal

class MultiDAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        self.dims = self.p_dims + self.q_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

class MF(nn.Module):
    def __init__(self, user_num, item_num, latent_dim):
        super(MF, self).__init__()
        self.user_latent = nn.Embedding(user_num, latent_dim, sparse=True)
        self.item_latent = nn.Embedding(item_num, latent_dim, sparse=True)
        self.user_biases = torch.nn.Embedding(user_num, 
                                              1,
                                              sparse=True)
        self.item_biases = torch.nn.Embedding(item_num,
                                              1,
                                              sparse=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item, user_neg, item_neg):
        if not isinstance(user_neg, torch.Tensor):
            u = self.user_latent(user)
            i = self.item_latent(item)
            pos_score = self.user_biases(user) + self.item_biases(item)
            pos_score = pos_score.squeeze()
            pos_score += (u * i).sum(1)
            # pos_score = self.sigmoid(pos_score)
            return pos_score

        u = self.user_latent(user)
        i = self.item_latent(item)
        pos_score = self.user_biases(user) + self.item_biases(item)
        pos_score = pos_score.squeeze()
        pos_score += (u * i).sum(1)

        n_u = self.user_latent(user_neg)
        n_i = self.item_latent(item_neg)
        neg_score = self.user_biases(user_neg) + self.item_biases(item_neg)
        neg_score = neg_score.squeeze()
        neg_score += (n_u*n_i).sum(1)
        # pos_score = self.sigmoid(pos_score)
        # neg_score = self.sigmoid(neg_score)
        return pos_score, neg_score

class HPrior_VAE(nn.Module):
    def __init__(self, dims, hidden_size_rnn, dataset, input_size, activation='tanh'):
        super(HPrior_VAE, self).__init__()
        self.dims = dims
        self.dataset = dataset
        if os.path.exists(f'./pretrained/{self.dataset}/embed_weight.pkl'):
            with open(f'./pretrained/{self.dataset}/embed_weight.pkl', 'rb') as f:
                embed = pickle.load(f)
        self.embedding = nn.Embedding.from_pretrained(embed)
        self.input_linear = nn.Linear(input_size, dims[-1])
        self.encoder = VAEEncoder(self.dims, hidden_size_rnn, activation)
        self.decoder = VAEDecoder(self.dims[::-1], hidden_size_rnn, activation)
        # embed = self._load_embedding()
        self.init_weights()

    def _load_embedding(self):
        for movie_id in range(1,3356):
            if movie_id % 3000 == 0:
                print('embedding is being made :', movie_id)
            with open(f'./data/{self.dataset}/genome/{movie_id - 1}.pkl', 'rb') as f:
                d = pickle.load(f)
                if movie_id-1==0:
                    sequence_data = torch.FloatTensor(d).unsqueeze(0)
                else:
                    sequence_data = torch.cat((sequence_data, torch.FloatTensor(d).unsqueeze(0)),0)
                f.close()
        padding = torch.zeros(1128).unsqueeze(0)
        embed_weight = torch.cat((padding,sequence_data), 0)
        with open(f'./pretrained/{self.dataset}/embed_weight.pkl', 'wb') as f:
            pickle.dump(embed_weight, f)
        return embed_weight

    def forward(self, x, item_feature):
        # print(item_feature.shape)
        item_feature,_ = pad_packed_sequence(item_feature)
        # print(torch.transpose(item_feature, 0,1).shape)
        item = self.input_linear(self.embedding(torch.transpose(item_feature, 0,1)))
        mean = torch.mean(item, dim=1)
        deviation = torch.std(item, dim=1)
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar, mean, deviation)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def reparam(self, mu, logvar, mean, deviation):
        if self.training:
            for k in range(len(deviation)):
                if k ==0:
                    devi = torch.diag(deviation[k]).unsqueeze(0)
                else:
                    devi = torch.cat((devi, torch.diag(deviation[k]).unsqueeze(0)), 0)
            m = MultivariateNormal(mean, devi)
            prior = m.sample()
            std = torch.exp(0.5 * logvar)
            return prior.mul(std).add_(mu)
        else:
            return mu

    def init_weights(self):
        for layer in self.encoder.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.decoder.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class VAEEncoder(nn.Module):
    def __init__(self, dims, condition_size, activation='tanh'):
        super(VAEEncoder, self).__init__()
        d = dims.copy() 
        d[-1] = d[-1] * 2
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(d[:-1], d[1:])])

        self.activation = activation
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        self.latent_size = dims[-1]
        self.drop = nn.Dropout(0.5)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.normalize(x)
        x = self.drop(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1 :
                x  = self.activations[self.activation](x)
            else:
                gaussian_params = x
        mu = gaussian_params[:,:self.latent_size]
        logvar = gaussian_params[:, self.latent_size:]
        return mu, logvar
        
class VAEDecoder(nn.Module):
    def __init__(self, dims, condition_size, activation):
        super(VAEDecoder, self).__init__()
        self.activation = activation
        d = dims.copy()
        
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(d[:-1], d[1:])])
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1 :
                x  = self.activations[self.activation](x)
            else:
                result = x
        # result = self.layer_2(self.drop_2(self.tanh(self.layer_1(latent))))
        return result

