# -*- coding:utf-8 -*-
"""

@Author: LMC_ZC

# test_a : userid, feedid, device
# date_ : 1 ~ 14
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class Process(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.user_action, self.feed_embeddings, self.feed_info, self.test_a = self._load(data_path)

        use_colnames = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id',
                        'videoplayseconds', 'device', 'date_',
                        'read_comment', 'like', 'comment', 'click_avatar', 'forward', 'follow', 'favorite']

        self.feature_engineering()
        self.process_df = self._merge()
        self.process_df = self.process_df[use_colnames]
        self.test_Completion()
        self._save()

    def _load(self, data_path):
        user_action = pd.read_csv(data_path + '/raw/user_action.csv')
        feed_embeddings = pd.read_csv(data_path + '/raw/feed_embeddings.csv')
        feed_info = pd.read_csv(data_path + '/raw/feed_info.csv')
        test_a = pd.read_csv(data_path + '/raw/test_a.csv')

        return user_action, feed_embeddings, feed_info, test_a

    def _merge(self):

        return pd.merge(self.user_action, self.feed_info, on='feedid')

    # 特征工程
    # 这里只是对 feed embedding 和 videoplayseconds 做了处理
    def feature_engineering(self):
        self._process_feed_embeddings()
        self._process_videoplayseconds()

        # 处理 feed embedding

    # 返回根据 feedid 排序后的(可设置降维) embedding, 这里为了处理快, 还是用numpy 存储好点
    def _process_feed_embeddings(self, reduce_dim=64):

        # sort
        # self.feed_embeddings = self.feed_embeddings.sort_values(by=['feedid'])

        all_feat_embeddings = [np.array(list(map(lambda x: eval(x), d.strip(' ').split(' '))), dtype=np.float32)
                               for d in self.feed_embeddings['feed_embedding']]
        all_feat_embeddings = [np.expand_dims(feat, axis=0) for feat in all_feat_embeddings]
        all_feat_embeddings = np.concatenate(all_feat_embeddings, axis=0)
        # all_feat_embeddings = all_feat_embeddings.T

        feed_embeddings_index = self.feed_embeddings['feedid'].values

        if reduce_dim is None:
            result = all_feat_embeddings
        else:
            pca = PCA(n_components=reduce_dim)
            pca.fit(all_feat_embeddings)
            result = pca.transform(all_feat_embeddings)

        self.feed_embeddings = result
        self.feed_embeddings_index = feed_embeddings_index

    # 用 log 做个平滑而已
    def _process_videoplayseconds(self):
        bias = 1.0
        eps = 1e-9

        val_vps = np.log2(self.feed_info['videoplayseconds'] + bias)
        val_vps = (val_vps - val_vps.min()) / (val_vps.max() - val_vps.min() + eps)
        self.feed_info['videoplayseconds'] = val_vps

    # 求 feedid 的特征
    def _generate_feedid2otherfeat(self, df, col_names):
        feedid2otherfeat = df[['feedid'] + [col_names]]
        feedid2otherfeat.index = feedid2otherfeat['feedid'].values
        feedid2otherfeat = feedid2otherfeat[col_names]

        otherfeat_pad = list(map(lambda x: feedid2otherfeat[x], self.test_a['feedid']))
        otherfeat_series = pd.Series(data=otherfeat_pad, name=col_names)

        return otherfeat_series

    def test_Completion(self):

        videoplayseconds_series = self._generate_feedid2otherfeat(self.feed_info, 'videoplayseconds')
        authorid_series = self._generate_feedid2otherfeat(self.feed_info, 'authorid')
        bsongid_series = self._generate_feedid2otherfeat(self.feed_info, 'bgm_song_id')
        bsingerid_series = self._generate_feedid2otherfeat(self.feed_info, 'bgm_singer_id')

        self.test_a = pd.concat([self.test_a, videoplayseconds_series, authorid_series,
                                 bsongid_series, bsingerid_series], axis=1)

    def _save(self):
        self.process_df.to_csv(self.data_path + '/processed/data.csv', index=None)
        self.test_a.to_csv(self.data_path + '/processed/test.csv', index=None)
        np.save(self.data_path + '/processed/feed_embeddings.npy', self.feed_embeddings)
        np.save(self.data_path + '/processed/feed_embeddings_index.npy', self.feed_embeddings_index)


if __name__ == '__main__':
    proj_path = '/kaggle/working/wechat_bigdata_din'
    D = Process(proj_path + '/data/wechat_algo_data1')
