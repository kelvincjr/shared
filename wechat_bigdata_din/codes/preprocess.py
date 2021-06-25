# -*- coding:utf-8 -*-
"""
@Author: LMC_ZC
# test_a : userid, feedid, device
"""

import pickle
import numpy as np
import pandas as pd


class Preprocess(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.user_action, self.feed_embeddings, self.feed_info, self.test_a = self._load(data_path)
        # self.raw_df = self._merge()

    def _load(self, data_path):
        user_action = pd.read_csv(data_path + '/user_action.csv')
        feed_embeddings = pd.read_csv(data_path + '/feed_embeddings.csv')
        feed_info = pd.read_csv(data_path + '/feed_info.csv')
        test_a = pd.read_csv(data_path + '/test_a.csv')

        return user_action, feed_embeddings, feed_info, test_a

    """
    def _merge(self):
        d = pd.merge(self.user_action, self.feed_info, on='feedid')
        d = pd.merge(d, self.feed_embeddings, on='feedid')
        return d
    """

    def _have_nan(self, df, col_name):
        return False if pd.isna(df[col_name]).mean() == 0.0 else True

    # 连续化: 从begin开始编号, 返回编号的总数和映射表
    def _serialize_id(self, mats, col_name, begin=0):
        mats = list(map(lambda x: x[col_name], mats))
        ids = np.concatenate(mats, axis=-1)
        ids = np.unique(ids)
        count_id = ids.shape[0]

        # mapping function
        map_ids = np.arange(begin, count_id + begin, dtype=np.int32)
        mapid_series = pd.Series(index=ids, data=map_ids)

        return count_id, mapid_series

    # 执行映射
    def _run_mapping(self, mat, mapping_series, col_name):
        mapped_id = mat[col_name].map(lambda x: mapping_series[x])
        mat[col_name] = mapped_id

        return mat

    def serialize(self):
        # 没有nan
        count_uid, uid_series = self._serialize_id(
            [self.user_action, self.test_a], 'userid', begin=0)
        count_fid, fid_series = self._serialize_id(
            [self.user_action, self.feed_info, self.feed_embeddings, self.test_a], 'feedid', begin=1)
        count_aid, aid_series = self._serialize_id([self.feed_info, ], 'authorid', begin=1)

        # bgm_song_id, bgm_singer_id 存在nan
        _, bsong_id_series = self._serialize_id([self.feed_info, ], 'bgm_song_id', begin=1)
        _, bsinger_id_series = self._serialize_id([self.feed_info, ], 'bgm_singer_id', begin=1)
        bsong_id_series = bsong_id_series[~bsong_id_series.index.duplicated(keep='first')]
        bsinger_id_series = bsinger_id_series[~bsinger_id_series.index.duplicated(keep='first')]
        count_bsong_id = bsong_id_series.values.shape[0]
        count_bsinger_id = bsinger_id_series.values.shape[0]

        self.user_action = self._run_mapping(self.user_action, uid_series, 'userid')
        self.user_action = self._run_mapping(self.user_action, fid_series, 'feedid')

        self.feed_info = self._run_mapping(self.feed_info, fid_series, 'feedid')
        self.feed_info = self._run_mapping(self.feed_info, bsong_id_series, 'bgm_song_id')
        self.feed_info = self._run_mapping(self.feed_info, bsinger_id_series, 'bgm_singer_id')

        self.test_a = self._run_mapping(self.test_a, uid_series, 'userid')
        self.test_a = self._run_mapping(self.test_a, fid_series, 'feedid')

        self.feed_embeddings = self._run_mapping(self.feed_embeddings, fid_series, 'feedid')

        return {'count_uid': count_uid, 'count_fid': count_fid, 'count_aid': count_aid,
                'count_bsong_id': count_bsong_id, 'count_bsinger_id': count_bsinger_id,
                'uid_series': uid_series, 'fid_series': fid_series, 'aid_series': aid_series,
                'bsong_id_series': bsong_id_series, 'bsinger_id_series': bsinger_id_series}

    def _save_series(self, df):
        index = pd.Series(df.index, name='src')
        val = pd.Series(df.values, name='des')

        return pd.concat([index, val], axis=1)

    def _save(self, dict_data):
        with open(self.data_path + '/processed/statistical.pkl', 'wb') as f:
            pickle.dump([dict_data['count_uid'],
                         dict_data['count_fid'],
                         dict_data['count_aid'],
                         dict_data['count_bsong_id'],
                         dict_data['count_bsinger_id'],
                         self.user_action['device'].unique().shape[0]], f, pickle.HIGHEST_PROTOCOL)

        self._save_series(dict_data['uid_series']).to_csv(self.data_path + '/raw/uid_series.csv', index=False)
        self._save_series(dict_data['fid_series']).to_csv(self.data_path + '/raw/fid_series.csv', index=False)
        self._save_series(dict_data['aid_series']).to_csv(self.data_path + '/raw/aid_series.csv', index=False)
        self._save_series(dict_data['bsong_id_series']).to_csv(self.data_path + '/raw/bsong_id_series.csv', index=False)
        self._save_series(dict_data['bsinger_id_series']).to_csv(self.data_path + '/raw/bsinger_id_series.csv',
                                                                 index=False)

        self.user_action.to_csv(self.data_path + '/raw/user_action.csv', index=False)
        self.feed_info.to_csv(self.data_path + '/raw/feed_info.csv', index=False)
        self.feed_embeddings.to_csv(self.data_path + '/raw/feed_embeddings.csv', index=False)
        self.test_a.to_csv(self.data_path + '/raw/test_a.csv', index=False)


if __name__ == '__main__':
    proj_path = '/kaggle/working/wechat_bigdata_din'
    D = Preprocess(proj_path + '/data/wechat_algo_data1')
    dict_data = D.serialize()
    D._save(dict_data)
