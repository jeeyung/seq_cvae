import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import time
from scipy import sparse
import random
import sys
from torch.nn.utils.rnn import pad_sequence, pack_sequence
import torch.nn.functional as F
''' it is for amazond data'''

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def timefn(fn):
    def wrap(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print("@timefn:{} took {} seconds".format(fn.__name__, t2-t1))
        return result
    return wrap

class AmazonRatingLoader():
    def __init__(self, data_dir, item_num):
        self.item_num = item_num
        self.data_dir = data_dir
        self.tp = pd.read_csv(os.path.join(self.data_dir,'preprocessed_data.csv'))

    def dataread(self):
        tr_users, vd_users, te_users = [],[],[]
        np.random.seed(98765)
        t_users = []
        for i in range(1,6):
            per_user = self.tp[self.tp.user_per == i].user.unique()
            idx_perm = np.random.permutation(len(per_user))
            per_user = per_user[idx_perm]
            per_sample = int(len(per_user) * 0.15)
            tr_users.extend(per_user[:(len(per_user)-per_sample*2)])
            vd_users.extend(per_user[(len(per_user)-per_sample *2) : (len(per_user) - per_sample)])
            te_users.extend(per_user[(len(per_user)-per_sample):])
            t_users.extend(per_user)
        
        train_plays = self.tp.loc[self.tp['user'].isin(tr_users)]
        self.unique_sid = pd.unique(train_plays['item'])

        self.show2id = dict((sid, i) for (i, sid) in enumerate(self.unique_sid))
        self.profile2id = dict((pid, i) for (i, pid) in enumerate(t_users))

        with open(os.path.join(self.data_dir, 'show2id.pkl'), 'wb') as f:
            pickle.dump(self.show2id, f)
        with open(os.path.join(self.data_dir, 'profile2id.pkl'), 'wb') as f:
            pickle.dump(self.profile2id, f)

        with open(os.path.join(self.data_dir, 'tr_users.pkl'), 'wb') as f:
            pickle.dump(tr_users, f)
        with open(os.path.join(self.data_dir, 'vd_users.pkl'), 'wb') as f:
            pickle.dump(vd_users, f)
        with open(os.path.join(self.data_dir, 'te_users.pkl'), 'wb') as f:
            pickle.dump(te_users, f)

        # print(len(tr_users), len(vd_users), len(te_users))

        vad_tr, vad_te = self.preprocess(vd_users)
        test_tr, test_te = self.preprocess(te_users)
        
        train_plays, vad_tr, vad_te, test_tr, test_te = self.numerize(train_plays), self.numerize(vad_tr), self.numerize(vad_te),\
                                                        self.numerize(test_tr), self.numerize(test_te)
        return train_plays, vad_tr, vad_te, test_tr, test_te

    def numerize(self, df):
        uid = df['user'].map(lambda x:self.profile2id[x])
        sid = df['item'].map(lambda x:self.show2id[x])
        return pd.DataFrame(data={'uid': uid, 'sid': sid, 'user' : df['user']}, columns=['uid', 'sid', 'user'])

    def UserItemDict(self):
        self.user_per_dict = {}
        self.item_per_dict = {}
        for i in range(1,6):
            variable_name = "user_{}".format(i)
            #user_period 마다 user list
            self.user_per_dict[variable_name] = self.tp[self.tp["user_per"]==i].user.values.tolist()
            variable_name = "item_{}".format(i)
            #user_period 마다 item list
            self.item_per_dict[variable_name] = self.tp[self.tp["user_per"]==i][self.tp["only_per_user"]==True].item.values.tolist()
            # self.item_per_dict[variable_name] = tp[tp["user_per"]==i].sid.values.tolist()

    def preprocess(self, users):
        df = self.tp.loc[self.tp['user'].isin(users)]
        df = df.loc[df['item'].isin(self.unique_sid)]
        data_tr, data_te = self.split_train_test_proportion(df)
        return data_tr, data_te

    def split_train_test_proportion(self, data, test_prop=0.2):
        print("start train eval split!")
        np.random.seed(98765)
        data_grouped_by_user = data.groupby('user')
        tr_list, te_list = list(), list()
        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)
            if n_items_u >= 5:
                if len(group[group["only_per_user"]==True]) >0:
                    group = group.reset_index()
                    n_items_u = len(group)
                    if len(group[group["only_per_user"]==True]) < int(test_prop * n_items_u):
                        te_list.append(group[group["only_per_user"]==True])
                        selected_index = group[group["only_per_user"]==True].index
                        idx = np.zeros(n_items_u, dtype='bool')
                        choice = np.random.choice(n_items_u, size=int(test_prop * n_items_u),\
                                            replace=False).astype('int64')
                        rest = list(set(choice) - set(selected_index))
                        idx[rest] = True
                        tr_list.append(group[np.logical_not(idx)])
                        te_list.append(group[idx])
                    else:
                        selected_index = group[group["only_per_user"]==True].index
                        choice = np.random.choice(selected_index, size=int(test_prop * n_items_u),\
                                            replace=False).astype('int64')
                        idx = np.zeros(n_items_u, dtype='bool')
                        idx[choice] = True
                        tr_list.append(group[np.logical_not(idx)])
                        te_list.append(group[idx])
                else:
                    idx = np.zeros(n_items_u, dtype='bool')
                    idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
                    tr_list.append(group[np.logical_not(idx)])
                    te_list.append(group[idx])
            else:
                tr_list.append(group)
            if i % 1000 == 0:
                print("%d users sampled" % i)
                sys.stdout.flush()
        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        return data_tr, data_te

    def get_seq_data(self, df_ori):
        self.UserItemDict()
        print("start pre-processing item feature!")
        users = df_ori.user.unique()
        df = self.tp.loc[self.tp["user"].isin(users)]
        grouped = df.groupby('user')
        for i, (g_user, group) in enumerate(grouped):
            if i % 1000 == 0:
                print('order : ', i)
            if g_user in self.user_per_dict['user_1']:
                per = 0
            elif g_user in self.user_per_dict['user_2']:
                per = 1
            elif g_user in self.user_per_dict['user_3']:
                per = 2
            elif g_user in self.user_per_dict['user_4']:
                per = 3
            else:
                per = 4
            g_uid = self.profile2id[g_user]
            if os.path.exists(self.data_dir + f'/amazon_feature_per_user/period_{per+1}/user_{g_uid}.pkl'):
                continue
            # try:
            #     with open(self.data_dir + f'/item_genomes_ver8/period_{per+1}/user_{g_uid}.pkl','rb') as f:
            #         sequence_data = pickle.load(f)
            #         f.close()
            # except:
            else:
                os.mkdir(os.path.join(self.data_dir, f'/amazon_feature_per_user/period_{per+1}'))
                for i in range(5):
                    if per == i:
                        item_v_name = "item_{}".format(i+1)
                        items_unique_period = set(self.item_per_dict[item_v_name]).intersection(set(group.item.tolist()))
                        items_unique_period_sorted = group[group['item'].isin(list(items_unique_period))].sort_values(by=['timestamp']).item.tolist()
                        if len(items_unique_period_sorted) < 5:
                            _, _, files = next(os.walk(os.path.join(self.data_dir, f"item_features_seq20/period_{per+1}")))
                            period_num = len(files)
                            # print("file nums : ", period_num)
                            ran_num = random.choice(list(range(period_num)))
                            with open(self.data_dir + f'/item_features_seq20/period_{per+1}/chunk_{ran_num}.pkl','rb') as f:
                                sequence_data = pickle.load(f)
                                f.close()
                        else:
                            sequence_data=[]
                            for i, item in enumerate(items_unique_period_sorted):
                                try:
                                    item_sid = self.show2id[item]
                                except:
                                    continue
                                else:
                                    sequence_data.append(item_sid)
                        break
                    else:
                        continue
                with open(self.data_dir + f'/amazon_feature_per_user/period_{per+1}/user_{g_uid}.pkl','wb') as f:
                    pickle.dump(sequence_data, f)
                    f.close()

    def load_train_data(self, csv_file):
        tp = pd.read_csv(csv_file)
        train_user_uni = tp['uid'].unique()
        ### sorting??
        train_user_uni.sort()
        user2id = dict((uid,i) for (i, uid) in enumerate(train_user_uni))
        n_users = len(train_user_uni)

        rows = tp['uid'].map(lambda x:user2id[x])
        cols = tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                (rows, cols)), dtype='float64',
                                shape=(n_users, self.item_num))
        return data

    # @staticmethod
    def load_tr_te_data(self, csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        vad_te_user_uni = tp_te['uid'].unique()
        ## test에는 없고 train에만 있는 user이 존재했었음, 이제는 없을듯?
        # with open('./data/ml-1m/vd_users_uid_fix_2.pkl', 'wb') as f:
            # pickle.dump(vad_te_user_uni, f)
        ## test에 있는 user만 뽑음
        tp_tr = tp_tr[tp_tr.uid.isin(vad_te_user_uni)]
        # vad_tr_user_uni = tp_tr['uid'].unique()
        # tp_tr = tp_tr[tp_tr.uid.isin(vad_te_user_uni)]
        ## sorting???
        vad_te_user_uni.sort()
        vad_user2id = dict((uid,i) for (i, uid) in enumerate(vad_te_user_uni))
        n_users = len(vad_te_user_uni)
        
        rows_tr = tp_tr['uid'].map(lambda x:vad_user2id[x])
        rows_te = tp_te['uid'].map(lambda x:vad_user2id[x])
        cols_tr = tp_tr['sid']
        cols_te = tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                (rows_tr, cols_tr)), dtype='float64', shape=(n_users, self.item_num))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                (rows_te, cols_te)), dtype='float64', shape=(n_users, self.item_num))
        return data_tr, data_te
    
    def load_pretrained(self):
        for i in range(5):
            if i == 0:
                self.pretrained = torch.load(f'./pretrained/amazon/h_{i+1}.pt').unsqueeze(0)
            else:
                self.pretrained = torch.cat((self.pretrained, torch.load(f'./pretrained/amazon/h_{i+1}.pt').unsqueeze(0)), 0)

    def fixed_hidden(self, total_batch, tr_f, batch_size, idx_pe):
        self.load_pretrained()
        with open(os.path.join(self.data_dir, 'profile2id.pkl'), 'rb') as f:
            profile2id = pickle.load(f)

        per1_users = self.tp.loc[self.tp.user_per==1].user.map(lambda x:profile2id[x]).unique()
        per2_users = self.tp.loc[self.tp.user_per==2].user.map(lambda x:profile2id[x]).unique()
        per3_users = self.tp.loc[self.tp.user_per==3].user.map(lambda x:profile2id[x]).unique()
        per4_users = self.tp.loc[self.tp.user_per==4].user.map(lambda x:profile2id[x]).unique()

        if tr_f == 'train':
            with open(f'{self.data_dir}/tr_users.pkl', 'rb') as f:
                users = pickle.load(f)
        elif tr_f == 'valid':
            with open(f'{self.data_dir}/vd_users.pkl', 'rb') as f:
                users = pickle.load(f)
                f.close()
            # with open('./data/ml-1m/vd_users_uid_fix_2.pkl', 'rb') as f:
                # vd_uid = pickle.load(f)
        else:
            with open(f'{self.data_dir}/te_users.pkl', 'rb') as f:
                users = pickle.load(f)
        uid=[]
        # if tr_f == 'valid':
            # for us in users:
                # uss = profile2id[us]
                # if uss in vd_uid:
                    # uid.append(uss)
        # else:
        for us in users:
            uid.append(profile2id[us])
        ### sorting???
        uid.sort()
        uid = np.array(uid)
        uid = uid[idx_pe]

        for i in range(total_batch):
            h_list = []
            user_ba = uid[i*batch_size:(i+1)*batch_size]
            for u in user_ba:
                if u in per1_users:
                    h = self.pretrained[0]
                elif u in per2_users:
                    h = self.pretrained[1]
                elif u in per3_users:
                    h = self.pretrained[2]
                elif u in per4_users:
                    h = self.pretrained[3]
                else:
                    h = self.pretrained[4]
                # h = F.sigmoid(h)
                h_list.append(torch.tensor(h.unsqueeze(0)))
            yield torch.cat(h_list, dim=0)

    def load_sequence_data_generator(self, total_batch, tr_f, batch_size, idx_pe):
        with open(os.path.join(self.data_dir, 'profile2id.pkl'), 'rb') as f:
            profile2id = pickle.load(f)

        per1_users = self.tp.loc[self.tp.user_per==1].user.map(lambda x:profile2id[x]).unique()
        per2_users = self.tp.loc[self.tp.user_per==2].user.map(lambda x:profile2id[x]).unique()
        per3_users = self.tp.loc[self.tp.user_per==3].user.map(lambda x:profile2id[x]).unique()
        per4_users = self.tp.loc[self.tp.user_per==4].user.map(lambda x:profile2id[x]).unique()
        # per3_users = self.tp.loc[self.tp.user_per==3].userId.unique()
        if tr_f == 'train':
            with open(f'{self.data_dir}/tr_users.pkl', 'rb') as f:
                users = pickle.load(f)
        elif tr_f == 'valid':
            with open(f'{self.data_dir}/vd_users.pkl', 'rb') as f:
                users = pickle.load(f)
                f.close()
            # with open('./data/ml-1m/vd_users_uid_fix_2.pkl', 'rb') as f:
                # vd_uid = pickle.load(f)
        else:
            with open(f'{self.data_dir}/te_users.pkl', 'rb') as f:
                users = pickle.load(f)
        uid=[]
        # if tr_f == 'valid':
        #     for us in users:
        #         uss = profile2id[us]
        #         if uss in vd_uid:
        #             uid.append(uss)
        # else:
        for us in users:
            uid.append(profile2id[us])
        uid.sort()
        ### add
        uid = np.array(uid)
        uid = uid[idx_pe]
        for i in range(total_batch):
            sequence_data_list = []
            per_list = []
            user_ba = uid[i*batch_size:(i+1)*batch_size]
            for u in user_ba:
                # print(u)
                if u in per1_users:
                    per = 0
                elif u in per2_users:
                    per = 1
                elif u in per3_users:
                    per = 2
                elif u in per4_users:
                    per = 3
                else:
                    per = 4
                # uid = profile2id[u]
                with open(self.data_dir + f'/amazon_feature_per_user/period_{per+1}/user_{u}.pkl','rb') as f:
                    sequence_data = pickle.load(f)
                    f.close() 
                    sequence_data_list.append(sequence_data)
                    per_list.append(per)

            ## order 주의할것!!!의
            order = np.argsort([len(item) for item in sequence_data_list])
            item_sorted = [torch.LongTensor(sequence_data_list[i]) for i in order[::-1]]
            label_sorted = [per_list[i] for i in order[::-1]]
            item = pack_sequence(item_sorted)
            yield (order[::-1], item, torch.tensor(label_sorted, dtype=torch.long))

class ItemDataset_amazon(Dataset):
    def __init__(self, data_dir, seq_len):
        self.data_dir = data_dir
        self.dataread(data_dir) 
        self.seq_len = seq_len

        with open(f'{self.data_dir}/show2id.pkl', 'rb') as f:
            self.show2id = pickle.load(f)
        with open(f'{self.data_dir}/profile2id.pkl', 'rb') as f:
            self.profile2id = pickle.load(f)

        if not os.path.exists(os.path.join(data_dir, f'item_features_seq{self.seq_len}')):
            os.mkdir(os.path.join(data_dir, f'item_features_seq{self.seq_len}'))
        dir_list = [f for f in os.listdir(os.path.join(data_dir,f'item_features_seq{self.seq_len}')) if not f.startswith('.')]
            
        if len(dir_list) == 5:
            print('loading saved files ...')
            # self.MakeDataset()
        else:
            print('making files...')
            os.mkdir(os.path.join(data_dir, f'item_features_seq{self.seq_len}'))
            os.mkdir(os.path.join(data_dir, f'item_features_seq{self.seq_len}', "period_1"))
            os.mkdir(os.path.join(data_dir, f'item_features_seq{self.seq_len}', "period_2"))
            os.mkdir(os.path.join(data_dir, f'item_features_seq{self.seq_len}', "period_3"))
            os.mkdir(os.path.join(data_dir, f'item_features_seq{self.seq_len}', "period_4"))
            os.mkdir(os.path.join(data_dir, f'item_features_seq{self.seq_len}', "period_5"))
            self.MakeDataset()

    def dataread(self, data_dir):
        self.data = pd.read_csv(os.path.join(data_dir,'./preprocessed_data.csv'))
    
    @timefn
    def MakeDataset(self):
        ''' it is possible to be changed to way where making sequence per user'''
        for per in range(1,6):
            print("start ", per)
            df = self.data[self.data["user_per"]==per][self.data["only_per_user"]==True]
            period_list = df.loc[df.item.isin(self.show2id.keys())].item.values.tolist()
            # period_list = self.data[self.data["item_per"]==per][self.data["only_per"]==True].sid.values.tolist()
            
            for k in range(int(len(period_list)/self.seq_len)):
                sequence_data = []
                chunk = period_list[k*self.seq_len:(k+1)*self.seq_len]
                for item in chunk:
                    # embedding doesn't need padding !
                    i_sid = self.show2id[item] 
                    sequence_data.append(i_sid)
                with open(self.data_dir + f'/item_features_seq{self.seq_len}/period_{per}/chunk_{k}.pkl','wb') as f:
                    pickle.dump(sequence_data, f)
                    f.close()
                    del sequence_data

    def __getitem__(self, idx):
        files_1_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_1"))[2])
        files_2_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_2"))[2])+files_1_num
        files_3_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_3"))[2])+files_2_num
        files_4_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_4"))[2])+files_3_num
        if idx < files_1_num:
            with open(self.data_dir + f'/item_features_seq{self.seq_len}/period_1/chunk_{idx}.pkl', 'rb') as f:
                seq_data = pickle.load(f)
            per = 0
        elif idx < files_2_num:
            idx = idx - files_1_num
            with open(self.data_dir + f'/item_features_seq{self.seq_len}/period_2/chunk_{idx}.pkl', 'rb') as f:
                seq_data = pickle.load(f)
            per = 1
        elif idx < files_3_num:
            idx = idx - files_2_num
            with open(self.data_dir + f'/item_features_seq{self.seq_len}/period_3/chunk_{idx}.pkl', 'rb') as f:
                seq_data = pickle.load(f)
            per = 2
        elif idx < files_4_num:
            idx = idx - files_3_num
            with open(self.data_dir + f'/item_features_seq{self.seq_len}/period_4/chunk_{idx}.pkl', 'rb') as f:
                seq_data = pickle.load(f)
            per = 3
        else:
            idx = idx - files_4_num
            with open(self.data_dir + f'/item_features_seq{self.seq_len}/period_5/chunk_{idx}.pkl', 'rb') as f:
                seq_data = pickle.load(f)
            per = 4
        assert torch.FloatTensor(seq_data).shape[0] == self.seq_len
        # if torch.FloatTensor(seq_data).shape[0] != self.seq_len:
            # seq_data = np.append(seq_data, [[0]*1128], axis=0)
        sample = {'feature':torch.LongTensor(seq_data), 'label' :per}
        return sample
        
    def __len__(self):
        files_1_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_1"))[2])
        files_2_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_2"))[2])
        files_3_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_3"))[2])
        files_4_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_4"))[2])
        files_5_num = len(next(os.walk(self.data_dir + f"/item_features_seq{self.seq_len}/period_5"))[2])
        # print("total", files_1_num + files_2_num + files_3_num)
        return files_1_num + files_2_num + files_3_num + files_4_num + files_5_num


class ItemRatingLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tp = pd.read_csv(os.path.join(self.data_dir,'train_only_user_per.csv'))

    def dataread(self):
        tr_users, vd_users, te_users = [],[],[]
        np.random.seed(98765)
        t_users = []
        for i in range(1,4):
            per_user = self.tp[self.tp.user_per == i].userId.unique()
            idx_perm = np.random.permutation(len(per_user))
            per_user = per_user[idx_perm]
            per_sample = 300
            tr_users.extend(per_user[:(len(per_user)-per_sample*2)])
            vd_users.extend(per_user[(len(per_user)-per_sample *2) : (len(per_user) - per_sample)])
            te_users.extend(per_user[(len(per_user)-per_sample):])
            t_users.extend(per_user)
        
        train_plays = self.tp.loc[self.tp['userId'].isin(tr_users)]
        self.unique_sid = pd.unique(train_plays['movieId'])

        self.show2id = dict((sid, i) for (i, sid) in enumerate(self.unique_sid))
        self.profile2id = dict((pid, i) for (i, pid) in enumerate(t_users))

        with open(os.path.join(self.data_dir, 'show2id.pkl'), 'wb') as f:
            pickle.dump(self.show2id, f)
        with open(os.path.join(self.data_dir, 'profile2id.pkl'), 'wb') as f:
            pickle.dump(self.profile2id, f)

        with open(os.path.join(self.data_dir, 'tr_users.pkl'), 'wb') as f:
            pickle.dump(tr_users, f)
        with open(os.path.join(self.data_dir, 'vd_users.pkl'), 'wb') as f:
            pickle.dump(vd_users, f)
        with open(os.path.join(self.data_dir, 'te_users.pkl'), 'wb') as f:
            pickle.dump(te_users, f)

        # print(len(tr_users), len(vd_users), len(te_users))

        vad_tr, vad_te = self.preprocess(vd_users)
        test_tr, test_te = self.preprocess(te_users)
        
        train_plays, vad_tr, vad_te, test_tr, test_te = self.numerize(train_plays), self.numerize(vad_tr), self.numerize(vad_te),\
                                                        self.numerize(test_tr), self.numerize(test_te)
        return train_plays, vad_tr, vad_te, test_tr, test_te

    def numerize(self, df):
        uid = df['userId'].map(lambda x:self.profile2id[x])
        sid = df['movieId'].map(lambda x:self.show2id[x])
        return pd.DataFrame(data={'uid': uid, 'sid': sid, 'userId' : df['userId']}, columns=['uid', 'sid', 'userId'])

    def UserItemDict(self):
        self.user_per_dict = {}
        self.item_per_dict = {}
        for i in range(1,4):
            variable_name = "user_{}".format(i)
            #user_period 마다 user list
            self.user_per_dict[variable_name] = self.tp[self.tp["user_per"]==i].userId.values.tolist()
            variable_name = "item_{}".format(i)
            #user_period 마다 item list
            self.item_per_dict[variable_name] = self.tp[self.tp["user_per"]==i][self.tp["only_per_user"]==True].movieId.values.tolist()
            # self.item_per_dict[variable_name] = tp[tp["user_per"]==i].sid.values.tolist()

    def preprocess(self, users):
        df = self.tp.loc[self.tp['userId'].isin(users)]
        df = df.loc[df['movieId'].isin(self.unique_sid)]
        data_tr, data_te = self.split_train_test_proportion(df)
        return data_tr, data_te

    def split_train_test_proportion(self, data, test_prop=0.2):
        print("start train eval split!")
        np.random.seed(98765)
        data_grouped_by_user = data.groupby('userId')
        tr_list, te_list = list(), list()
        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)
            if n_items_u >= 5:
                if len(group[group["only_per_user"]==True]) >0:
                    group = group.reset_index()
                    n_items_u = len(group)
                    te_list.append(group[group["only_per_user"]==True])
                    selected_index = group[group["only_per_user"]==True].index
                    idx = np.zeros(n_items_u, dtype='bool')
                    choice = np.random.choice(n_items_u, size=int(0.2* n_items_u),\
                                        replace=False).astype('int64')
                    rest = list(set(choice) - set(selected_index))
                    idx[rest] = True
                    tr_list.append(group[np.logical_not(idx)])
                    te_list.append(group[idx])
                else:
                    idx = np.zeros(n_items_u, dtype='bool')
                    idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
                    tr_list.append(group[np.logical_not(idx)])
                    te_list.append(group[idx])
            else:
                tr_list.append(group)
            if i % 1000 == 0:
                print("%d users sampled" % i)
                sys.stdout.flush()
        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        return data_tr, data_te

    def get_seq_data(self, df_ori):
        self.UserItemDict()
        print("start pre-processing item feature!")
        users = df_ori.userId.unique()
        df = self.tp.loc[self.tp["userId"].isin(users)]
        # df = df.sort_values(by=['uid'])
        grouped = df.groupby('userId')
        # sequence_data_list = []
        # per_list = []
        # sequence_per_user = {}
        for i, (g_user, group) in enumerate(grouped):
            if i % 1000 == 0:
                print('order : ', i)
            if g_user in self.user_per_dict['user_1']:
                per = 0
            elif g_user in self.user_per_dict['user_2']:
                per = 1
            else:
                per = 2
            # n_items_u = len(group)
            g_uid = self.profile2id[g_user]
            if os.path.exists(self.data_dir + f'/item_genomes_v2/period_{per+1}/user_{g_uid}.pkl'):
                continue
            # try:
            #     with open(self.data_dir + f'/item_genomes_ver8/period_{per+1}/user_{g_uid}.pkl','rb') as f:
            #         sequence_data = pickle.load(f)
            #         f.close()
            # except:
            else:
                for i in range(3):
                    if per == i:
                        item_v_name = "item_{}".format(i+1)
                        items_unique_period = set(self.item_per_dict[item_v_name]).intersection(set(group.movieId.tolist()))
                        items_unique_period_sorted = group[group['movieId'].isin(list(items_unique_period))].sort_values(by=['timestamp']).movieId.tolist()
                        if len(items_unique_period_sorted) < 5:
                            _, _, files = next(os.walk(os.path.join(self.data_dir, f"item_genomes_seq20/period_{per+1}")))
                            period_num = len(files)
                            # print("file nums : ", period_num)
                            ran_num = random.choice(list(range(period_num)))
                            with open(self.data_dir + f'/item_genomes_seq20/period_{per+1}/chunk_{ran_num}.pkl','rb') as f:
                                sequence_data = pickle.load(f)
                                f.close()
                        else:
                            sequence_data=[]
                            for i, item in enumerate(items_unique_period_sorted):
                                try:
                                    item_sid = self.show2id[item]+1
                                except:
                                    continue
                                else:
                                    sequence_data.append(item_sid)
                        break
                    else:
                        continue
                with open(self.data_dir + f'/item_genomes_v2/period_{per+1}/user_{g_uid}.pkl','wb') as f:
                    pickle.dump(sequence_data, f)
                    f.close()
        #     sequence_data_list.append(sequence_data)
        #     per_list.append(per)
        # return sequence_data_list, per_list
    
    @staticmethod
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        # n_users = tp['uid'].max() + 1

        train_user_uni = tp['uid'].unique()
        train_user_uni.sort()
        user2id = dict((uid,i) for (i, uid) in enumerate(train_user_uni))
        n_users = len(train_user_uni)

        rows = tp['uid'].map(lambda x:user2id[x])
        cols = tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                (rows, cols)), dtype='float64',
                                shape=(n_users, 3355))
        return data

    @staticmethod
    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        vad_te_user_uni = tp_te['uid'].unique()
        with open('./data/ml-1m/vd_users_uid_fix_2.pkl', 'wb') as f:
            pickle.dump(vad_te_user_uni, f)
        
        tp_tr = tp_tr[tp_tr.uid.isin(vad_te_user_uni)]
        
        # vad_tr_user_uni = tp_tr['uid'].unique()
        # tp_tr = tp_tr[tp_tr.uid.isin(vad_te_user_uni)]
        vad_te_user_uni.sort()
        vad_user2id = dict((uid,i) for (i, uid) in enumerate(vad_te_user_uni))
        n_users = len(vad_te_user_uni)
        
        rows_tr = tp_tr['uid'].map(lambda x:vad_user2id[x])
        rows_te = tp_te['uid'].map(lambda x:vad_user2id[x])
        cols_tr = tp_tr['sid']
        cols_te = tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                (rows_tr, cols_tr)), dtype='float64', shape=(n_users, 3355))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                (rows_te, cols_te)), dtype='float64', shape=(n_users, 3355))
        return data_tr, data_te
    
    def load_pretrained(self):
        for i in range(3):
            if i == 0:
                self.pretrained = torch.load(f'./pretrained/ml-1m/h_{i+1}.pt').unsqueeze(0)
            else:
                self.pretrained = torch.cat((self.pretrained, torch.load(f'./pretrained/ml-1m/h_{i+1}.pt').unsqueeze(0)), 0)

    def fixed_hidden(self, total_batch, tr_f, batch_size, idx_pe):
        self.load_pretrained()
        with open(os.path.join(self.data_dir, 'profile2id.pkl'), 'rb') as f:
            profile2id = pickle.load(f)

        per1_users = self.tp.loc[self.tp.user_per==1].userId.map(lambda x:profile2id[x]).unique()
        per2_users = self.tp.loc[self.tp.user_per==2].userId.map(lambda x:profile2id[x]).unique()

        if tr_f == 'train':
            with open(f'./data/ml-1m/tr_users.pkl', 'rb') as f:
                users = pickle.load(f)
        elif tr_f == 'valid':
            with open(f'./data/ml-1m/vd_users.pkl', 'rb') as f:
                users = pickle.load(f)
                f.close()
            with open('./data/ml-1m/vd_users_uid_fix_2.pkl', 'rb') as f:
                vd_uid = pickle.load(f)
        else:
            with open(f'./data/ml-1m/te_users.pkl', 'rb') as f:
                users = pickle.load(f)
        uid=[]
        if tr_f == 'valid':
            for us in users:
                uss = profile2id[us]
                if uss in vd_uid:
                    uid.append(uss)
        else:
            for us in users:
                uid.append(profile2id[us])
        uid.sort()
        uid = np.array(uid)
        uid = uid[idx_pe]

        for i in range(total_batch):
            h_list = []
            user_ba = uid[i*batch_size:(i+1)*batch_size]
            for u in user_ba:
                if u in per1_users:
                    h = self.pretrained[0]
                elif u in per2_users:
                    h = self.pretrained[1]
                else:
                    h = self.pretrained[2]
                # h = F.sigmoid(h)
                h_list.append(torch.tensor(h.unsqueeze(0)))
            yield torch.cat(h_list, dim=0)

    def load_sequence_data_generator(self, total_batch, tr_f, batch_size, idx_pe):
        with open(os.path.join(self.data_dir, 'profile2id.pkl'), 'rb') as f:
            profile2id = pickle.load(f)

        per1_users = self.tp.loc[self.tp.user_per==1].userId.map(lambda x:profile2id[x]).unique()
        per2_users = self.tp.loc[self.tp.user_per==2].userId.map(lambda x:profile2id[x]).unique()
        # print(per2_users)
        # per3_users = self.tp.loc[self.tp.user_per==3].userId.unique()
        if tr_f == 'train':
            with open(f'./data/ml-1m/tr_users.pkl', 'rb') as f:
                users = pickle.load(f)
        elif tr_f == 'valid':
            with open(f'./data/ml-1m/vd_users.pkl', 'rb') as f:
                users = pickle.load(f)
                f.close()
            with open('./data/ml-1m/vd_users_uid_fix_2.pkl', 'rb') as f:
                vd_uid = pickle.load(f)
        else:
            with open(f'./data/ml-1m/te_users.pkl', 'rb') as f:
                users = pickle.load(f)
        uid=[]
        if tr_f == 'valid':
            for us in users:
                uss = profile2id[us]
                if uss in vd_uid:
                    uid.append(uss)
        else:
            for us in users:
                uid.append(profile2id[us])
        uid.sort()
        
        ### add
        uid = np.array(uid)
        uid = uid[idx_pe]
        for i in range(total_batch):
            sequence_data_list = []
            per_list = []
            user_ba = uid[i*batch_size:(i+1)*batch_size]
            for u in user_ba:
                # print(u)
                if u in per1_users:
                    per = 0
                elif u in per2_users:
                    per = 1
                else:
                    per = 2
                # uid = profile2id[u]
                with open(self.data_dir + f'/item_genomes_v2/period_{per+1}/user_{u}.pkl','rb') as f:
                    sequence_data = pickle.load(f)
                    f.close() 
                    sequence_data_list.append(sequence_data)
                    per_list.append(per)

            ## order 완전 잘못 미친
            order = np.argsort([len(item) for item in sequence_data_list])
            item_sorted = [torch.LongTensor(sequence_data_list[i]) for i in order[::-1]]
            label_sorted = [per_list[i] for i in order[::-1]]
            item = pack_sequence(item_sorted)
            yield (order[::-1], item, torch.tensor(label_sorted, dtype=torch.long))
    

class ItemDataset(Dataset):
    def __init__(self, data_dir, seq_len):
        # print(111)
        self.data_dir = data_dir
        self.dataread(data_dir) 
        self.seq_len = seq_len

        with open('./data/ml-1m/show2id.pkl', 'rb') as f:
            self.show2id = pickle.load(f)
        with open('./data/ml-1m/profile2id.pkl', 'rb') as f:
            self.profile2id = pickle.load(f)

        if not os.path.exists(os.path.join(data_dir, f'item_genomes_seq{self.seq_len}')):
            os.mkdir(os.path.join(data_dir, f'item_genomes_seq{self.seq_len}'))
        dir_list = [f for f in os.listdir(os.path.join(data_dir,f'item_genomes_seq{self.seq_len}')) if not f.startswith('.')]
            
        if len(dir_list) == 3:
            print('loading saved files ...')
            # self.MakeDataset()
        else:
            print('making files...')
            os.mkdir(os.path.join(data_dir, f'item_genomes_seq{self.seq_len}', "period_1"))
            os.mkdir(os.path.join(data_dir, f'item_genomes_seq{self.seq_len}', "period_2"))
            os.mkdir(os.path.join(data_dir, f'item_genomes_seq{self.seq_len}', "period_3"))
            self.MakeDataset()

    def dataread(self, data_dir):
        self.data = pd.read_csv(os.path.join(data_dir,'./train_only_user_per.csv'))
    
    @timefn
    def MakeDataset(self):
        for per in range(1,4):
            print("start ", per)
            df = self.data[self.data["user_per"]==per][self.data["only_per_user"]==True]
            period_list = df.loc[df.movieId.isin(self.show2id.keys())].movieId.values.tolist()
            
            for k in range(int(len(period_list)/self.seq_len)):
                sequence_data = []
                chunk = period_list[k*self.seq_len:(k+1)*self.seq_len]
                for item in chunk:
                    i_sid = self.show2id[item] + 1
                    sequence_data.append(i_sid)
                with open(self.data_dir + f'/item_genomes_seq{self.seq_len}/period_{per}/chunk_{k}.pkl','wb') as f:
                    pickle.dump(sequence_data, f)
                    f.close()
                    del sequence_data

    def __getitem__(self, idx):
        files_1_num = len(next(os.walk(self.data_dir + f"/item_genomes_seq{self.seq_len}/period_1"))[2])
        files_2_num = len(next(os.walk(self.data_dir + f"/item_genomes_seq{self.seq_len}/period_2"))[2])+files_1_num
        if idx < files_1_num:
            with open(self.data_dir + f'/item_genomes_seq{self.seq_len}/period_1/chunk_{idx}.pkl', 'rb') as f:
                seq_data = pickle.load(f)
            per = 0
        elif idx < files_2_num:
            idx = idx - files_1_num
            with open(self.data_dir + f'/item_genomes_seq{self.seq_len}/period_2/chunk_{idx}.pkl', 'rb') as f:
                seq_data = pickle.load(f)
            per = 1
        else:
            idx = idx - files_2_num
            with open(self.data_dir + f'/item_genomes_seq{self.seq_len}/period_3/chunk_{idx}.pkl', 'rb') as f:
                seq_data = pickle.load(f)
            per = 2
        assert torch.FloatTensor(seq_data).shape[0] == self.seq_len
        sample = {'feature':torch.LongTensor(seq_data), 'label' :per}
        return sample
        
    def __len__(self):
        files_1_num = len(next(os.walk(self.data_dir + f"/item_genomes_seq{self.seq_len}/period_1"))[2])
        files_2_num = len(next(os.walk(self.data_dir + f"/item_genomes_seq{self.seq_len}/period_2"))[2])
        files_3_num = len(next(os.walk(self.data_dir + f"/item_genomes_seq{self.seq_len}/period_3"))[2])
        return files_1_num + files_2_num + files_3_num 


# if __name__ == "__main__":
#     dataload = AmazonRatingLoader('./data/amazon', 14984)
# #     #rating data
#     train_plays, vad_tr, vad_te, test_tr, test_te = dataload.dataread()
#     train_plays.to_csv(os.path.join(dataload.data_dir,'train.csv'), index=False)
#     vad_tr.to_csv(os.path.join(dataload.data_dir,'valid_tr.csv'), index=False)
#     vad_te.to_csv(os.path.join(dataload.data_dir,'valid_te.csv'), index=False)
#     test_tr.to_csv(os.path.join(dataload.data_dir,'test_tr.csv'), index=False)
#     test_te.to_csv(os.path.join(dataload.data_dir,'test_te.csv'), index=False)
#     print("save all data!")
    
#     # dataload.get_seq_data(vad_tr)
#     # dataload.get_seq_data(train_plays)
#     # dataload.get_seq_data(test_tr)

if __name__ == "__main__":
    dataset = ItemDataset_amazon('./data/amazon', 20)
    loader = DataLoader(dataset, batch_size=3)
    for i, data in enumerate(loader):
        print(data['feature'].shape)
        break