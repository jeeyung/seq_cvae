import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import numpy as np
from dataloader import *
import time 

class RNNEncoder_amazon(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layer, dropout_p, bidirectional,\
                                     class_num, dataset, item_num, activation, freeze):
        super(RNNEncoder_amazon, self).__init__()
        self.dataset = dataset
        self.item_num = item_num
        self.bidirectional = bidirectional
        self.activation = activation
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        if os.path.exists(f'./pretrained/{self.dataset}/embed_weight.pkl'):
            with open(f'./pretrained/{self.dataset}/embed_weight.pkl', 'rb') as f:
                weight = pickle.load(f)
            print('completed loading embedding!')
        else:        
            weight = self._load_weight()

        # self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=freeze)

        self.input_linear = nn.Linear(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layer, \
                                    dropout=dropout_p, bidirectional=bidirectional, batch_first=True)
        self.output_linear = nn.Linear(hidden_size, class_num)

    def _load_weight(self):
        check_pad = []
        for item_id in range(0, self.item_num):
            if item_id % 3000 == 0:
                print('embedding is being made :', item_id)
            try:
                with open(f'./data/{self.dataset}/image_cnn/{item_id}.pkl', 'rb') as f:
                    d = pickle.load(f)
            ### there are some items not being used for train / test        
            except:
                sequence_data = torch.cat([sequence_data, torch.zeros(4096).unsqueeze(0)],0)
                check_pad.append(item_id)
                continue
            else:
                if item_id == 0:
                    sequence_data = torch.FloatTensor(d).unsqueeze(0)
                else:
                    sequence_data = torch.cat((sequence_data, torch.FloatTensor(d).unsqueeze(0)),0)
                f.close()
        # padding = torch.zeros(4096).unsqueeze(0)
        # embed_weight = torch.cat((padding,sequence_data), 0)
        # print(embed_weight)
        with open(f'./pretrained/{self.dataset}/embed_weight.pkl', 'wb') as f:
            pickle.dump(sequence_data, f)
        with open(f'./pretrained/{self.dataset}/check_pad.pkl', 'wb') as f:
            pickle.dump(check_pad, f)
        return sequence_data

    def forward(self, x):
        ### only for joint training
        if isinstance(x, PackedSequence):
            bs = x.batch_sizes
            x = self.embedding(x.data)
            x = self.activations[self.activation](self.input_linear(x))
            x = PackedSequence(data=x, batch_sizes=bs)
        else:
            x = self.embedding(x)
            x = self.activations[self.activation](self.input_linear(x))
        _, (h, _) = self.lstm(x)
        if self.bidirectional:
            h = h[0] + h[1] / 2
        h = h.squeeze()
        result = self.output_linear(h)
        return F.softmax(result, dim=-1), h


class RNNEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layer, dropout_p, bidirectional, class_num, dataset, condition_size):
        super(RNNEncoder, self).__init__()
        self.dataset = dataset
        if self.dataset == 'ml-1m':
            if os.path.exists(f'./pretrained/{self.dataset}/embed_weight.pkl'):
                with open(f'./pretrained/{self.dataset}/embed_weight.pkl', 'rb') as f:
                    weight = pickle.load(f)
                    print("load ml-1m embedding")
            else:
                print("start to make embedding!")        
                weight = self._load_weight()

        else:
            if os.path.exists(f'./pretrained/{self.dataset}/embed_weight.pkl'):
                with open(f'./pretrained/{self.dataset}/embed_weight.pkl', 'rb') as f:
                    weight = pickle.load(f)
            else:        
                weight = self._load_weight()

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.input_linear = nn.Linear(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layer, \
                                    dropout=dropout_p, bidirectional=bidirectional, batch_first=True)
        self.bidirectional = bidirectional
        # if bidirectional:
            # hidden_size = hidden_size * 2
        self.output_linear = nn.Linear(hidden_size, class_num)
        self.output_linear2 = nn.Linear(hidden_size, condition_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def _load_weight(self):
        for movie_id in range(1,3356):
            if movie_id % 3000 == 0:
                print('embedding is being made :', movie_id)
            with open(f'./data/{self.dataset}/genome/{movie_id - 1}.pkl', 'rb') as f:
                d = pickle.load(f)
                if movie_id-1==0:
                    sequence_data = torch.FloatTensor(d).unsqueeze(0)
                else:
                    ## 잘 못 되있음
                    # sequence_data = torch.cat([torch.FloatTensor(d).unsqueeze(0), sequence_data],0)
                    sequence_data = torch.cat((sequence_data, torch.FloatTensor(d).unsqueeze(0)),0)
                f.close()
        padding = torch.zeros(1128).unsqueeze(0)
        embed_weight = torch.cat((padding,sequence_data), 0)
        # print(embed_weight)
        with open(f'./pretrained/{self.dataset}/embed_weight.pkl', 'wb') as f:
            pickle.dump(embed_weight, f)
        return embed_weight

    # def forward(self, x):
    #     ### only for joint training
    #     if not self.pretrain:
    #         x, l = pad_packed_sequence(x) 
    #         x = x.transpose(0,1) 
    #     ### added
    #     x = self.embedding(x)
    #     # x = self.tanh(self.input_linear(x)) #[batch, seq, embedding_size]
    #     x = pack_padded_sequence(x, l)
    #     _, (h, _) = self.lstm(x)
        
    #     if self.bidirectional:
    #         h = h[0] + h[1] / 2
    #     h = h.squeeze()
    #     result = self.output_linear(h)
        
    #     ### only for joint training
    #     # result = self.sigmoid(result)
    #     return result, h
    
    def forward(self, x):
        ### only for joint training
        if isinstance(x, PackedSequence):
            bs = x.batch_sizes
            x = self.embedding(x.data)
            x = self.tanh(self.input_linear(x))
            x = PackedSequence(data=x, batch_sizes=bs)
        else:
            x = self.embedding(x)
            x = self.tanh(self.input_linear(x))
        # x = self.tanh(self.input_linear(x)) #[batch, seq, embedding_size]
        # x = pack_padded_sequence(x, l)
        _, (h, _) = self.lstm(x)
        # print(h.shape)
        if self.bidirectional:
            h = h[0] + h[1] / 2
        h = h.squeeze()
        result = self.output_linear(h)
        h = self.output_linear2(h)
        ### only for joint training
        # result = self.sigmoid(result)
        # print(result)
        return F.softmax(result, dim=-1), h


class VAEEncoder(nn.Module):
    # def __init__(self, input_size, hidden_size, latent_size, condition, condition_size):
    def __init__(self, dims, condition, condition_size, activation='tanh'):
        super(VAEEncoder, self).__init__()
        self.condition = condition
        d = dims.copy() 
        if self.condition:
            d[0] = d[0] + condition_size
        d[-1] = d[-1] * 2
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(d[:-1], d[1:])])

        self.activation = activation
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        # self.layer_1 = nn.Linear(input_size, hidden_size)
        # self.layer_2 = nn.Linear(hidden_size, latent_size*2)
        # self.output_layer = nn.Linear(hidden_size_2, latent_size*2)

        # self.attn_layer_h = nn.Linear(condition_size, condition_size)
        # self.attn_layer = nn.Linear(input_size-condition_size, condition_size)

        ### last
        # self.attn_layer_h = nn.Linear(condition_size, input_size - condition_size)
        # self.attn_layer = nn.Linear(input_size-condition_size, input_size - condition_size)

        self.latent_size = dims[-1]
        self.drop = nn.Dropout(0.5)
        ### dropout removed
        # self.drop_2 = nn.Dropout(0.1)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        # self.condition_mlp = nn.Linear(condition_size, 30)

    def forward(self, x, condition_h):
        x = F.normalize(x)
        x = self.drop(x)
        if self.condition:
            # condition_h = condition_h.squeeze()
            # print(condition_h.shape)
            x = torch.cat((x, condition_h), dim=1)
            # print(x.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1 :
                x  = self.activations[self.activation](x)
            else:
                gaussian_params = x
        # gaussian_params = self.layer_2(self.drop_2(self.tanh(self.layer_1(x))))
        mu = gaussian_params[:,:self.latent_size]
        logvar = gaussian_params[:, self.latent_size:]
        return mu, logvar

    # def forward(self, x, condition_h):
    #     x = F.normalize(x)
    #     x = self.drop(x)
    #     if self.condition:
    #         attn = self.softmax(self.elu(self.attn_layer_h(condition_h) + self.attn_layer(x)))
    #         condition_h = attn * condition_h
    #         condition_h = condition_h.squeeze()
    #         x = torch.cat((x, condition_h), dim=1)
    #     gaussian_params = self.layer_2(self.drop_2(self.tanh(self.layer_1(x))))
    #     # gaussian_params = self.output_layer(self.drop_2(self.elu(self.layer_2(self.drop_2(self.elu(self.layer_1(x)))))))
    #     mu = gaussian_params[:,:self.latent_size]
    #     # logvar = self.softplus(gaussian_params[:, self.latent_size:]) + 1e-6
    #     logvar = gaussian_params[:, self.latent_size:]
    #     return mu, logvar 
    #### last
    # def forward(self, x, condition_h):
    #     x = F.normalize(x)
    #     # x = self.drop(x)
    #     if self.condition:
    #         attn = self.softmax(self.elu(self.attn_layer_h(condition_h) + self.attn_layer(x)))
    #         x = attn * x
    #         # condition_h = self.elu(self.condition_mlp(condition_h))
    #         condition_h = condition_h.squeeze()
    #         x = torch.cat((x, condition_h), dim=1)
    #     gaussian_params = self.layer_2(self.drop_2(self.tanh(self.layer_1(x))))
    #     # gaussian_params = self.output_layer(self.drop_2(self.elu(self.layer_2(self.drop_2(self.elu(self.layer_1(x)))))))
    #     mu = gaussian_params[:,:self.latent_size]
    #     # logvar = self.softplus(gaussian_params[:, self.latent_size:]) + 1e-6
    #     logvar = gaussian_params[:, self.latent_size:]
    #     return mu, logvar 
    # def forward(self, x, condition_h):
    #     x = F.normalize(x)
    #     x = self.drop(x)
    #     if self.condition:
    #         condition_h = self.elu(self.condition_mlp(condition_h))
    #         condition_h = condition_h.squeeze()
    #         x = torch.cat((x, condition_h), dim=1)
    #     gaussian_params = self.layer_2(self.drop_2(self.tanh(self.layer_1(x))))
    #     # gaussian_params = self.output_layer(self.drop_2(self.elu(self.layer_2(self.drop_2(self.elu(self.layer_1(x)))))))
    #     mu = gaussian_params[:,:self.latent_size]
    #     # logvar = self.softplus(gaussian_params[:, self.latent_size:]) + 1e-6
    #     logvar = gaussian_params[:, self.latent_size:]
    #     return mu, logvar 
        
class VAEDecoder(nn.Module):
    def __init__(self, dims, condition, condition_size, activation):
        super(VAEDecoder, self).__init__()
        self.condition = condition
        self.activation = activation
        d = dims.copy()
        if self.condition:
            d[0] = d[0] + condition_size
        
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(d[:-1], d[1:])])
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])

    def forward(self, x, condition_h):
        if self.condition:
            condition_h = condition_h.squeeze()
            x = torch.cat((x, condition_h), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1 :
                x  = self.activations[self.activation](x)
            else:
                result = x
        # result = self.layer_2(self.drop_2(self.tanh(self.layer_1(latent))))
        return result

class VAEDecoder_attn(nn.Module):
    def __init__(self, dims, condition, condition_size, activation):
        super(VAEDecoder_attn, self).__init__()
        self.condition = condition
        self.activation = activation
        d = dims.copy()
        if self.condition:
            d[0] = d[0] + condition_size
        
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(d[:-1], d[1:])])
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])

        self.attn_layer_h = nn.Linear(condition_size, condition_size)
        self.attn_layer = nn.Linear(d[0] - condition_size, condition_size)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, condition_h):
        if self.condition:
            attn = self.softmax(self.elu(self.attn_layer_h(condition_h) + self.attn_layer(x)))
            attn_condition = attn * condition_h
            attn_condition = attn_condition.squeeze()
            x = torch.cat((x, attn_condition), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1 :
                x  = self.activations[self.activation](x)
            else:
                result = x
        return result

class VAE_RNN_rec(nn.Module):
    def __init__(self, dims, input_size_rnn, embedding_size, num_layer, dropout_p, bidirectional, \
                class_num,  hidden_size_rnn, condition, dataset, activation, freeze, attn, condition_size):
        super(VAE_RNN_rec, self).__init__()
        self.dims = dims
        # self.encoder = VAEEncoder(self.dims, condition, hidden_size_rnn, activation)
        self.encoder = VAEEncoder(self.dims, condition, condition_size, activation)
        # self.reparam = Reparametrize()
        if attn:
            # self.decoder = VAEDecoder_attn(self.dims[::-1], condition, hidden_sized_rnn, activation)  
            self.decoder = VAEDecoder_attn(self.dims[::-1], condition, condition_size, activation)  
        else:
            # self.decoder = VAEDecoder(self.dims[::-1], condition, hidden_size_rnn, activation)
            self.decoder = VAEDecoder(self.dims[::-1], condition, condition_size, activation)
        # self.decoder = VAEDecoder_attn(self.dims[::-1], condition, hidden_size_rnn, activation)  
        if dataset == 'amazon' or dataset == 'amazon_min20_woman' or dataset == 'amazon_min20_woman_fix' or dataset == 'amazon_min10_woman':
            self.RNNEncoder = RNNEncoder_amazon(input_size_rnn, embedding_size, hidden_size_rnn, num_layer, dropout_p,\
                                            bidirectional, class_num, dataset, self.dims[0], activation, freeze)
        else:
            self.RNNEncoder = RNNEncoder(input_size_rnn, embedding_size, hidden_size_rnn, num_layer, dropout_p, bidirectional, class_num, dataset, condition_size)

        self.condition = condition
        self.init_weights()

    def forward(self, model_input):
        if self.condition:
            x, h = model_input
        else:
            x = model_input
            h = None
        mu, logvar = self.encoder(x, h)
        z = self.reparam(mu, logvar)
        recon_x = self.decoder(z, h)
        return recon_x, mu, logvar

    def reparam(self, mu, logvar):
        if self.training:
            # z = torch.randn_like(sigma, dtype=torch.float32) * sigma + mu
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
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

# class VAE_rec(nn.Module):
#     def __init__(self, input_size_vae, input_size_rnn, embedding_size, num_layer, dropout_p, bidirectional, \
#                 class_num, hidden_size_vae, hidden_size_rnn, latent_size, condition, data_dir, pretrain):
#         super(VAE_rec, self).__init__()
#         self.encoder = VAEEncoder(input_size_vae, hidden_size_vae, latent_size, condition, hidden_size_rnn)
#         # self.reparam = Reparametrize()
#         self.decoder = VAEDecoder(input_size_vae, hidden_size_vae, latent_size, condition, hidden_size_rnn)
#         # self.RNNEncoder = RNNEncoder(input_size_rnn, embedding_size, hidden_size_rnn, num_layer, dropout_p, bidirectional, class_num, data_dir, pretrain)
#         self.condition = condition
#         self.init_weights()

#     def forward(self, model_input):
#         if self.condition:
#             x, h = model_input
#         else:
#             x = model_input
#             h = None
#         mu, logvar = self.encoder(x, h)
#         z = self.reparam(mu, logvar)
#         recon_x = self.decoder(z, h)
#         return recon_x, mu, logvar

#     def reparam(self, mu, logvar):
#         if self.training:
#             # z = torch.randn_like(sigma, dtype=torch.float32) * sigma + mu
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def init_weights(self):
#         for layer in [self.encoder.layer_1,self.encoder.layer_2]:
#             # Xavier Initialization for weights
#             size = layer.weight.size()
#             fan_out = size[0]
#             fan_in = size[1]
#             std = np.sqrt(2.0/(fan_in + fan_out))
#             layer.weight.data.normal_(0.0, std)

#             # Normal Initialization for Biases
#             layer.bias.data.normal_(0.0, 0.001)
        
#         for layer in [self.decoder.layer_1, self.decoder.layer_2]:
#             # Xavier Initialization for weights
#             size = layer.weight.size()
#             fan_out = size[0]
#             fan_in = size[1]
#             std = np.sqrt(2.0/(fan_in + fan_out))
#             layer.weight.data.normal_(0.0, std)

#             # Normal Initialization for Biases
#             layer.bias.data.normal_(0.0, 0.001)

if __name__ == '__main__':
    loader = RatingDataLoader('./data/ml-20m', 2, True, 0.3)
    model = VAE_rec(9731, 100, 10, True, 20)
    # condition = torch.Tensor(20)
    for data in loader:
        print("condition shape", data["condition"].shape)
        print(data['rating'].size())
        output = model(data['rating'], data['condition'])
        break



