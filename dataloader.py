from torch.utils.data import DataLoader, Subset
from dataset_total import *
import numpy as np
import sys, os
import torch
from torch.utils.data import random_split
import random 
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_sequence

class ItemFeatureDataLoader_amazon(DataLoader):
    def __init__(self, data_dir, seq_len, batch_size, train, test_ratio):
        self.dataset = ItemDataset_amazon(data_dir, seq_len)
        self.train = train
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.data_dir = data_dir
       
        split_dataset = self.TrainTestSplit()
        super(ItemFeatureDataLoader_amazon, self).__init__(split_dataset, self.batch_size, shuffle=True, num_workers=20)
    
    def TrainTestSplit(self):
        master_dataset_num = len(self.dataset)
        n_test = int(master_dataset_num * self.test_ratio)
        n_train = master_dataset_num - n_test
        train_set, test_set = random_split(self.dataset, (n_train, n_test))
        if self.train:
            return train_set
        else:
            return test_set

class ItemFeatureDataLoader_ml1m(DataLoader):
    def __init__(self, data_dir, seq_len, batch_size, train, test_ratio):
        self.dataset = ItemDataset(data_dir, seq_len)
        self.train = train
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.data_dir = data_dir
       
        split_dataset = self.TrainTestSplit()
        # class_weights = [1,0.8,1]
        # class_weights = [0.96, 0.023, 0.017]
        # for weighted sampling - unbalanced dataset
     
        # print("set class weight")
        # if self.train:
        #     train_targets = [sample["label"] for sample in split_dataset]
        #     # print("train_targets", train_targets)
        #     samples_weight = [class_weights[class_id] for class_id in train_targets]
        # else:
        #     test_targets = [sample["label"] for sample in split_dataset]
        #     samples_weight = [class_weights[class_id] for class_id in test_targets]

        # weight_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(split_dataset))
        # super(ItemFeatureDataLoader_ml1m, self).__init__(split_dataset, self.batch_size, sampler=weight_sampler, num_workers=20)
        super(ItemFeatureDataLoader_ml1m, self).__init__(split_dataset, self.batch_size, shuffle=True, num_workers=20)
    
    def TrainTestSplit(self):
        master_dataset_num = len(self.dataset)
        n_test = int(master_dataset_num * self.test_ratio)
        n_train = master_dataset_num - n_test
        train_set, test_set = random_split(self.dataset, (n_train, n_test))
        if self.train:
            return train_set
        else:
            return test_set


# class ItemFeatureDataLoader(DataLoader):
#     def __init__(self, data_dir, seq_len, batch_size, train, test_ratio):
#         self.dataset = ItemDataset(data_dir, seq_len)
#         self.train = train
#         self.batch_size = batch_size
#         self.test_ratio = test_ratio
#         self.data_dir = data_dir
#         if os.path.exists('./pretrained/dataset/trainset.pkl') and os.path.exists('./pretrained/dataset/testset.pkl'):
#             print("start loading train, test dataset")
#             if self.train:
#                 with open('./pretrained/dataset/trainset.pkl','rb') as f:
#                     split_dataset = pickle.load(f)
#             else:
#                 with open('./pretrained/dataset/testset.pkl','rb') as f:
#                     split_dataset = pickle.load(f)
#         else: 
#             print("start split train test")
#             split_dataset = self.TrainTestSplit()
#         class_weights = [0.96, 0.023, 0.017]
#         # for weighted sampling - unbalanced dataset
#         if os.path.exists('./pretrained/dataset/trainset_sampleweight.pkl') and os.path.exists('./pretrained/dataset/testset_sampleweight.pkl'):
#             if self.train:
#                 with open('./pretrained/dataset/trainset_sampleweight.pkl','rb') as f:
#                     samples_weight = pickle.load(f)
#             else:
#                 with open('./pretrained/dataset/testset_sampleweight.pkl','rb') as f:
#                     samples_weight = pickle.load(f)

#         else:
#             print("set class weight")
#             if self.train:
#                 train_targets = [sample["label"] for sample in split_dataset]
#                 samples_weight = [class_weights[class_id] for class_id in train_targets]
#                 with open('./pretrained/dataset/trainset_sampleweight.pkl','wb') as f:
#                     pickle.dump(samples_weight, f)
#                     f.close()
#             else:
#                 test_targets = [sample["label"] for sample in split_dataset]
#                 samples_weight = [class_weights[class_id] for class_id in test_targets]
#                 with open('./pretrained/dataset/testset_sampleweight.pkl','wb') as f:
#                     pickle.dump(samples_weight, f)
#                     f.close()

#         weight_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(split_dataset))
#         super(ItemFeatureDataLoader, self).__init__(split_dataset, self.batch_size, sampler=weight_sampler, num_workers=20)
    
#     def TrainTestSplit(self):
#         master_dataset_num = len(self.dataset)
#         n_test = int(master_dataset_num * self.test_ratio)
#         n_train = master_dataset_num - n_test
#         train_set, test_set = random_split(self.dataset, (n_train, n_test))
        
#         if self.train:
#             return train_set
#         else:
#             return test_set

   # def timefn(fn):
#     def wrap(*args):
#         t1 = time.time()
#         result = fn(*args)
#         t2 = time.time()
#         print("@timefn:{} took {} seconds".format(fn.__name__, t2-t1))
#         return result
#     return wrap


# def collate_fn(item_list):
#     order = np.argsort([item["item_feature"].shape[0] for item in item_list])
#     rating_sorted = [item_list[i]["rating"] for i in order[::-1]]
#     item_sorted = [item_list[i]["item_feature"] for i in order[::-1]]
#     label_sorted= [item_list[i]["label"].unsqueeze(0) for i in order[::-1]]
#     rating = torch.cat(rating_sorted, 0)
#     label = torch.cat(label_sorted, 0)
#     item = pack_sequence(item_sorted)
#     sample = {"rating" : rating, "item_feature" : item, "label":label}
#     return sample

# class JointDataLoader(DataLoader):
#     def __init__(self, data_dir, batch_size, train, test_ratio, eval_ratio, n_sample, num_workers):
#         self.n_sample = n_sample
#         self.dataset = ItemRatingDataset(data_dir, n_sample)
#         # boolean
#         self.data_dir = data_dir
#         self.train = train
#         self.eval_ratio = eval_ratio
#         self.test_ratio = test_ratio
#         split_dataset = self.TrainTestSplit()
#         if self.train:
#             super(JointDataLoader, self).__init__(split_dataset, batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
#         else:
#             super(JointDataLoader, self).__init__(split_dataset, batch_size, shuffle=True, num_workers=num_workers)

#     def TrainTestSplit(self):
#         master_dataset_num = len(self.dataset)
#         n_test = int(master_dataset_num * self.test_ratio)
#         n_train = master_dataset_num - n_test
#         idx = list(range(master_dataset_num))
#         random.shuffle(idx)
#         train_idx = idx[:n_train]
#         test_idx = idx[(n_train):]
#         train_set = Subset(self.dataset, train_idx)
#         test_set = ItemRatingSubset(self.dataset, test_idx, self.data_dir, self.eval_ratio, self.n_sample)
#         if self.train:
#             return train_set
#         else:
#             return test_set

# class ItemRatingSubset(ItemRatingDataset):
#     def __init__(self, dataset, indices, data_dir, eval_ratio, n_sample):
#         super(ItemRatingSubset, self).__init__(data_dir, n_sample)
#         self.dataset = dataset
#         self.user_indices = indices
#         self.data = dataset.data_rating
#         self.user_1 = dataset.user_per_dict["user_1"]
#         self.user_3 = dataset.user_per_dict["user_3"]
#         self.user_4 = dataset.user_per_dict["user_4"]
#         self.user_2 = dataset.user_per_dict["user_2"]
#         # self.pretrained = dataset.pretrained
#         self.data_read(eval_ratio)

#     def data_read(self, eval_ratio):
#         df = self.tp.loc[self.tp["uid"].isin(self.user_indices)]
#         #split test and eval dataset
#         test_data, eval_data = self.split_eval_test(df, eval_ratio)

#         test_user_uni = test_data['uid'].unique()
#         user2id = dict((uid,i) for (i, uid) in enumerate(test_user_uni))
#         self.total_test_user = len(test_user_uni)
#         # start_idx = min(test_data['uid'].min(), eval_data['uid'].min())
#         # end_idx = max(test_data['uid'].max(), eval_data['uid'].max())

#         t_rows, t_cols = test_data['uid'].map(lambda x:user2id[x]), test_data['sid']
#         e_rows, e_cols = eval_data['uid'].map(lambda x:user2id[x]), eval_data['sid']
        
#         self.splited_data_test = sparse.csr_matrix((np.ones_like(t_rows),
#                                 (t_rows, t_cols)), dtype='float64',
#                                 shape=(self.total_test_user, 10350))

#         self.splited_data_eval = sparse.csr_matrix((np.ones_like(e_rows),
#                                 (e_rows, e_cols)), dtype='float64',
#                                 shape=(self.total_test_user, 10350))

#     def split_eval_test(self, data, eval_ratio):
#         data_grouped_by_user = data.groupby('uid')
#         tr_list, te_list = list(), list()
#         np.random.seed(777)
#         for i, (_, group) in enumerate(data_grouped_by_user):
#             n_items_u = len(group)
#             if n_items_u >= 5:
#                 idx = np.zeros(n_items_u, dtype='bool')
#                 idx[np.random.choice(n_items_u, size=int(eval_ratio * n_items_u), replace=False).astype('int64')] = True
#                 tr_list.append(group[np.logical_not(idx)])
#                 te_list.append(group[idx])
#             else:
#                 tr_list.append(group)
#         data_tr = pd.concat(tr_list)
#         data_te = pd.concat(te_list)
#         return data_tr, data_te

#     def __getitem__(self, idx):
#         if idx in self.user_1:
#             per = 1
#         elif idx in self.user_2:
#             per = 2
#         elif idx in self.user_3:
#             per = 3
#         else:
#             per = 4
#         test_te = self.splited_data_test[idx].toarray()
#         test_ev = self.splited_data_eval[idx].toarray()
#         sample = {'test_te':torch.FloatTensor(test_te).squeeze(), 'test_eval':torch.FloatTensor(test_ev).squeeze(), 'label':per}
#         return sample
#         # return self.dataset[self.indices[idx]]

#     def __len__(self):
#         return self.total_test_user




# class RatingDataLoader(DataLoader):
#     def __init__(self, data_dir, batch_size, train, test_ratio, eval_ratio):
#         self.dataset = RatingDataset(data_dir)
#         # boolean
#         self.data_dir = data_dir
#         self.train = train
#         self.eval_ratio = eval_ratio
#         self.test_ratio = test_ratio
#         split_dataset = self.TrainTestSplit()
#         super(RatingDataLoader, self).__init__(split_dataset, batch_size, shuffle=True)

#     def TrainTestSplit(self):
#         master_dataset_num = len(self.dataset)
#         n_test = int(master_dataset_num * self.test_ratio)
#         n_train = master_dataset_num - n_test
#         idx = list(range(master_dataset_num))
#         random.shuffle(idx)
#         train_idx = idx[:n_train]
#         test_idx = idx[(n_train):]
#         if self.train:
#             train_set = Subset(self.dataset, train_idx)
#             return train_set
#         else:
#             test_set = RatingSubset(self.dataset, test_idx, self.data_dir, self.eval_ratio)
#             return test_set

# class RatingSubset(RatingDataset):
#     def __init__(self, dataset, indices, data_dir, eval_ratio):
#         super(RatingSubset, self).__init__(data_dir)
#         self.dataset = dataset
#         self.user_indices = indices

#         self.data = dataset.data
#         self.user_1 = dataset.user_1
#         self.user_3 = dataset.user_3
#         self.user_4 = dataset.user_4
#         self.user_2 = dataset.user_2
#         self.pretrained = dataset.pretrained
        
#         self.data_read(eval_ratio)

#     def data_read(self, eval_ratio):
#         df = self.tp.loc[self.tp["uid"].isin(self.user_indices)]
#         #split test and eval dataset
#         test_data, eval_data = self.split_eval_test(df, eval_ratio)

#         test_user_uni = test_data['uid'].unique()
        
#         self.user2id = dict((uid,i) for (i, uid) in enumerate(test_user_uni))
#         # with open('./user2id.pkl', 'wb') as f:
#         #     pickle.dump(self.user2id, f)
#         self.total_test_user = len(test_user_uni)
        
        
#         # start_idx = min(test_data['uid'].min(), eval_data['uid'].min())
#         # end_idx = max(test_data['uid'].max(), eval_data['uid'].max())

#         t_rows, t_cols = test_data['uid'].map(lambda x:self.user2id[x]), test_data['sid']
#         e_rows, e_cols = eval_data['uid'].map(lambda x:self.user2id[x]), eval_data['sid']
        
#         self.splited_data_test = sparse.csr_matrix((np.ones_like(t_rows),
#                                 (t_rows, t_cols)), dtype='float64',
#                                 shape=(self.total_test_user,10350))
#         # with open('./splited_data_test.pkl', 'wb') as f:
#         #     pickle.dump(self.splited_data_test, f)
#         self.splited_data_eval = sparse.csr_matrix((np.ones_like(e_rows),
#                                 (e_rows, e_cols)), dtype='float64',
#                                 shape=(self.total_test_user,10350))
#         # with open('./splited_data_eval.pkl', 'wb') as f:
#         #     pickle.dump(self.splited_data_eval, f)
        
#     def split_eval_test(self, data, eval_ratio):
#         data_grouped_by_user = data.groupby('uid')
#         tr_list, te_list = list(), list()
#         np.random.seed(98765)
#         for i, (_, group) in enumerate(data_grouped_by_user):
#             n_items_u = len(group)
#             if n_items_u >= 5:
#                 idx = np.zeros(n_items_u, dtype='bool')
#                 idx[np.random.choice(n_items_u, size=int(eval_ratio * n_items_u), replace=False).astype('int64')] = True
#                 tr_list.append(group[np.logical_not(idx)])
#                 te_list.append(group[idx])
#             else:
#                 tr_list.append(group)
#         data_tr = pd.concat(tr_list)
#         data_te = pd.concat(te_list)
#         return data_tr, data_te

#     def __getitem__(self, idx):
#         # if idx in self.user_1:
#         #     h = self.pretrained[0]
#         # elif idx in self.user_2:
#         #     h = self.pretrained[1]
#         # elif idx in self.user_3:
#         #     h = self.pretrained[2]
#         # else:
#         #     h = self.pretrained[3]
#         h = torch.tensor(1)
#         test_te = self.splited_data_test[idx].toarray()
#         test_ev = self.splited_data_eval[idx].toarray()
#         # if test_ev.sum() < 1:
#         #     print(idx)
#         print(test_te)
#         sample = {'test_te':test_te, 'test_eval':test_ev,'condition':h}
#         # sample = {'test_te':torch.FloatTensor(test_te).squeeze(), 'test_eval':torch.FloatTensor(test_ev).squeeze(), 'condition':h}
#         return sample
#         # return self.dataset[self.indices[idx]]

#     def __len__(self):
#         return self.total_test_user

# if __name__ == '__main__':
#     # dataset = RatingDataset('./data/ml-20m')
#     loader = JointDataLoader('./data/ml-20m', 5, False, 0.2,0.2, 50,0)
#     # loader = RatingDataLoader('./data/ml-20m', 5, False, 0.2)
#     # for data in loader:
#     #     print(data)
#     #     break

#     # loader = ItemFeatureDataLoader('./data/ml-20m', 20, 3, True, 0.3)
#     for data in loader:
#         print(data['test_te'])
#         break 