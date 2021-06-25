# -*- coding:utf-8 -*-
"""

@Author: LMC_ZC

"""

import torch
import pickle
import numpy as np
import pandas as pd
from utils.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names


class Dataset(object):
    def __init__(self, data_path1, data_path2):
        self.data = pd.read_csv(data_path1)
        with open(data_path2, 'rb') as f:
            self.uid_vocabulary, self.fid_vocabulary, self.aid_vocabulary, \
            self.bsong_vocabulary, self.bsinger_vocabulary, self.device_vocabulary = pickle.load(f)

        self.max_uid_length = self._get_length(self.data, 'userid', 'feedid').max()

    def _get_length(self, df, src, des):

        data_grouped = df.groupby(src)
        return data_grouped[des].count()

    def _get_history(self, df, src, des):

        data_grouped = df.groupby(src)
        sub_data = list(data_grouped[des])
        history = [[d[1].to_list() + [0] * (self.max_uid_length - len(d[1]))] for d in sub_data]
        history = np.array(history)

        return history.squeeze()

    def _get_feed_embedding(self, df, feed_path='../data/processed/feed_embeddings.npy'):
        feed_emb = list()
        d = np.load(feed_path)
        for value in df['feedid'].values:
            feed_emb.append(d[value - 1])
        #feed_emb = d[df['feedid'].values]
        print(len(feed_emb))
        feed_emb = np.array(feed_emb)
        return feed_emb

    def get_feature_columns(self):
        feature_columns = [SparseFeat('user', self.uid_vocabulary, embedding_dim=64),
                           SparseFeat('feed', self.fid_vocabulary + 1, embedding_dim=64),
                           SparseFeat('author', self.aid_vocabulary + 1, embedding_dim=16),
                           SparseFeat('song', self.bsong_vocabulary + 1, embedding_dim=16),
                           SparseFeat('singer', self.bsinger_vocabulary + 1, embedding_dim=16),
                           SparseFeat('device', self.device_vocabulary + 1, embedding_dim=2),
                           DenseFeat('feed_embedding', 64),
                           DenseFeat('videoplayseconds', 1)]

        feature_columns += [VarLenSparseFeat(SparseFeat('hist_feed', self.fid_vocabulary + 1, embedding_dim=64),
                                             self.max_uid_length, length_name="seq_length"),
                            VarLenSparseFeat(SparseFeat('hist_author', self.aid_vocabulary + 1, embedding_dim=16),
                                             self.max_uid_length, length_name="seq_length"),
                            VarLenSparseFeat(SparseFeat('hist_song', self.bsong_vocabulary + 1, embedding_dim=16),
                                             self.max_uid_length, length_name="seq_length"),
                            VarLenSparseFeat(SparseFeat('hist_singer', self.bsinger_vocabulary + 1, embedding_dim=16),
                                             self.max_uid_length, length_name="seq_length")]

        # feature_columns += [VarLenSparseFeat(SparseFeat('hist_feed', self.fid_vocabulary+1, embedding_dim=16),
        #                                     self.max_uid_length, length_name="seq_length"),
        #                    VarLenSparseFeat(SparseFeat('hist_author', self.aid_vocabulary+1, embedding_dim=16),
        #                                     self.max_uid_length, length_name="seq_length")]

        behavior_feature_list = ["feed", "author", "song", "singer"]
        # behavior_feature_list = ["feed", "author"]
        seq_length_list = ['seq_length']

        return feature_columns, behavior_feature_list, seq_length_list

    def get_xy(self, df, feature_columns, feed_path, has_y=True):

        uid = df['userid'].to_numpy(dtype=np.float32)
        fid = df['feedid'].to_numpy(dtype=np.float32)
        aid = df['authorid'].to_numpy(dtype=np.float32)
        bid1 = df['bgm_song_id'].to_numpy(dtype=np.float32)
        bid2 = df['bgm_singer_id'].to_numpy(dtype=np.float32)

        device = df['device'].to_numpy(dtype=np.float32)
        videoplay = df['videoplayseconds'].to_numpy(dtype=np.float32)
        feed_embedding = self._get_feed_embedding(df, feed_path)

        his_fid = self._get_history(df, "userid", "feedid")
        his_aid = self._get_history(df, "userid", "authorid")
        his_bid1 = self._get_history(df, "userid", "bgm_song_id")
        his_bid2 = self._get_history(df, "userid", "bgm_singer_id")

        uid_length = self._get_length(df, 'userid', 'feedid')

        feature_dict = {'user': uid, 'feed': fid, 'device': device, 'author': aid, 'song': bid1, 'singer': bid2,
                        'videoplayseconds': videoplay, 'feed_embedding': feed_embedding,
                        'hist_feed': his_fid, 'hist_author': his_aid, 'hist_song': his_bid1, 'hist_singer': his_bid2,
                        'seq_length': uid_length.to_numpy(dtype=np.int32)}

        # feature_dict = {'user': uid, 'feed': fid, 'device': device, 'author': aid,
        #                'videoplayseconds': videoplay,
        #                'hist_feed': his_fid, 'hist_author': his_aid,
        #                'seq_length': uid_length.to_numpy(dtype=np.int32)}

        x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}

        if has_y:
            y1 = df['read_comment'].to_numpy(dtype=np.float32)
            y2 = df['like'].to_numpy(dtype=np.float32)
            y3 = df['click_avatar'].to_numpy(dtype=np.float32)
            y4 = df['forward'].to_numpy(dtype=np.float32)
            y = {'read_comment': y1, 'like': y2, 'click_avatar': y3, 'forward': y4}

            return x, y
        else:
            return x

    def get_validation(self, train_file, val_file, hasfile=False, split_ratio=0.1):

        # 这个版本的对validation的划分主要满足了：分层采样(label=1, label=0), 然后对于每个user, 均在train 和 validation 出现
        # 但是这样采样会导致线下的分数非常高, 因为是对用户的历史交互数据随机采样, 导致出现了特征穿越
        """
        groupby_uid = self.raw.groupby('userid')
        train, val = [], []
        for d in groupby_uid:
            uid, uid_df = d[0], d[1]
            u_d = list(uid_df.groupby(y_colname))
            if len(u_d) < 2:
                df = u_d[0][1]
                train_df_mask = (np.random.rand(df.shape[0], 1) >= split_ratio)
                if np.mean(train_df_mask) == 1.0:
                    train_df_mask[-1] = ~train_df_mask[-1]
                train_df = df[train_df_mask]
                val_df = df[~train_df_mask]
            else:
                neg_df, pos_df = u_d[0][1], u_d[1][1]
                train_neg_df_mask = (np.random.rand(neg_df.shape[0], 1) >= split_ratio)
                train_pos_df_mask = (np.random.rand(pos_df.shape[0], 1) >= split_ratio)
                if np.mean(train_neg_df_mask) == 1.0 and np.mean(train_pos_df_mask) == 1.0:
                    train_neg_df_mask[-1] = ~train_neg_df_mask[-1]
                train_df = pd.concat([neg_df[train_neg_df_mask], pos_df[train_pos_df_mask]], axis=0)
                val_df = pd.concat([neg_df[~train_neg_df_mask], pos_df[~train_pos_df_mask]], axis=0)
            train.append(train_df)
            val.append(val_df)
        train = pd.concat(train, axis=0)
        val = pd.concat(val, axis=0)
        return train, val
        """

        if hasfile:
            train = pd.read_csv(train_file)
            test = pd.read_csv(val_file)

        else:
            train, test = [], []
            groupby_uid = self.data.groupby('userid')
            for d in groupby_uid:
                uid, u_df = d[0], d[1]

                date_ = u_df['date_'].unique()
                max_date_ = date_.max()

                if date_.shape[0] <= 1:
                    train_df_mask = (np.random.rand(u_df.shape[0], 1) >= split_ratio)
                    if np.mean(train_df_mask) == 1.0 or np.mean(train_df_mask) == 0.0:
                        train_df_mask[-1] = ~train_df_mask[-1]

                    test_df_mask = ~train_df_mask

                else:
                    mask = (u_df['date_'] == max_date_)
                    test_df_mask = mask
                    train_df_mask = ~mask

                train_df = u_df[train_df_mask]
                test_df = u_df[test_df_mask]

                train.append(train_df)
                test.append(test_df)

            train = pd.concat(train, axis=0)
            test = pd.concat(test, axis=0)

            train.to_csv(train_file, index=False)
            test.to_csv(val_file, index=False)

        return train, test


class CustomerTensorDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, table):

        self.x = x
        self.y = y
        self.table = table

    def __getitem__(self, index):

        # x[0] denote the userid
        x = self.x[index]

        ids = x[0].squeeze().type(torch.long).tolist()
        history_feat = [torch.tensor(t[ids]).type(torch.float32) for t in self.table]
        history_feat = torch.cat(history_feat, dim=-1)
        x = torch.cat((x, history_feat), dim=-1)

        if self.y is None:
            return x
        else:
            return x, self.y[index]

    def __len__(self):

        return self.x.shape[0]


if __name__ == '__main__':
    d = Dataset('../data/processed/data.csv', '../data/processed/statistical.pkl')
    feature_columns, behavior_feature_list, seq_length_list = d.get_feature_columns()
    x, y = d.get_xy(d.data, feature_columns, '../data/processed/feed_embeddings.npy')
    print('ojbk')
