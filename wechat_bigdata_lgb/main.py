# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
# from tensorflow import feature_column as fc
from comm import ACTION_LIST, STAGE_END_DAY, FEA_COLUMN_LIST
from evaluation import uAUC, compute_weighted_score

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_checkpoint_dir', './data/model', 'model dir')
flags.DEFINE_string('root_path', './data/', 'data dir')
flags.DEFINE_integer('batch_size', 128, 'batch_size')
flags.DEFINE_integer('embed_dim', 10, 'embed_dim')
flags.DEFINE_float('learning_rate', 0.1, 'learning_rate')
flags.DEFINE_float('embed_l2', None, 'embedding l2 reg')

SEED = 2021


class WideAndDeep(object):

    def __init__(self, stage, action):
        """
        :param linear_feature_columns: List of tensorflow feature_column
        :param dnn_feature_columns: List of tensorflow feature_column
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        """
        super(WideAndDeep, self).__init__()
        self.num_epochs_dict = {"read_comment": 1, "like": 1, "click_avatar": 1, "favorite": 1, "forward": 1,
                                "comment": 1, "follow": 1}
        self.estimator = None
        self.stage = stage
        self.action = action
        self.dnn_feature_columns, self.linear_feature_columns = self.get_feature_columns()
        self.feature_names = get_feature_names(self.dnn_feature_columns + self.linear_feature_columns)
        tf.logging.set_verbosity(tf.logging.INFO)

    def build_estimator(self):
        if self.stage in ["evaluate", "offline_train"]:
            stage = "offline_train"
        else:
            stage = "online_train"
        model_checkpoint_stage_dir = os.path.join(FLAGS.model_checkpoint_dir, stage, self.action)
        if not os.path.exists(model_checkpoint_stage_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(model_checkpoint_stage_dir)
        elif self.stage in ["online_train", "offline_train"]:
            # 训练时如果模型目录已存在，则清空目录
            del_file(model_checkpoint_stage_dir)
        self.estimator = DeepFM(
            linear_feature_columns=self.linear_feature_columns + self.dnn_feature_columns,
            dnn_feature_columns=self.linear_feature_columns + self.dnn_feature_columns,
            dnn_hidden_units=[32, 8])
        self.estimator.compile('adagrad', 'binary_crossentropy', metrics=["binary_crossentropy", "auc"])

    def df_to_dataset(self, df, stage, action, shuffle=True, batch_size=128, num_epochs=1):
        '''
        把DataFrame转为tensorflow dataset
        :param df: pandas dataframe.
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        :param shuffle: Boolean.
        :param batch_size: Int. Size of each batch
        :param num_epochs: Int. Epochs num
        :return: tf.data.Dataset object.
        '''
        print(df.shape)
        print(df.columns)
        print("batch_size: ", batch_size)
        print("num_epochs: ", num_epochs)
        train, test = train_test_split(df, test_size=0.2)
        self.train_model_input = {name: train[name] for name in self.feature_names}
        self.test_model_input = {name: test[name] for name in self.feature_names}
        self.label = train[action]

    def get_feature_columns(self):
        '''
        获取特征列
        '''
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=self.action,
                                                                       day=STAGE_END_DAY[self.stage])
        stage_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        self.df = pd.read_csv(stage_dir)
        sparse_features = ["userid", "feedid", "authorid", "bgm_singer_id", "bgm_song_id"]
        self.df[sparse_features] = self.df[sparse_features].fillna('-1', )
        for feat in sparse_features:
            lbe = LabelEncoder()
            self.df[feat] = lbe.fit_transform(self.df[feat])
        # mms = MinMaxScaler(feature_range=(0, 1))
        # data[dense_features] = mms.fit_transform(data[dense_features])

        # df[dense_features] = df[dense_features].fillna(0, )
        linear_feature_columns = list()
        dnn_feature_columns = [SparseFeat(feat, self.df[feat].nunique(), FLAGS.embed_dim, dtype=str) for feat in sparse_features]

        video_seconds = DenseFeat(name='videoplayseconds')
        device = DenseFeat(name='device')
        linear_feature_columns.append(video_seconds)
        linear_feature_columns.append(device)
        # 行为统计特征
        for b in FEA_COLUMN_LIST:
            feed_b = DenseFeat(b + "sum")
            linear_feature_columns.append(feed_b)
            user_b = DenseFeat(b + "sum_user")
            linear_feature_columns.append(user_b)
        return dnn_feature_columns, linear_feature_columns

    def train(self):
        """
        训练单个行为的模型
        """

        self.df_to_dataset(df=self.df, stage=self.stage, action=self.action)
        self.estimator.fit(x=self.train_model_input, y=self.label.values, batch_size=128,
                           epochs=self.num_epochs_dict[self.action], verbose=2, validation_split=0.0)

    def evaluate(self):
        """
        评估单个行为的uAUC值
        """
        if self.stage in ["online_train", "offline_train"]:
            # 训练集，每个action一个文件
            action = self.action
        else:
            # 测试集，所有action在同一个文件
            action = "all"
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=action,
                                                                       day=STAGE_END_DAY[self.stage])
        evaluate_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(evaluate_dir)
        userid_list = df['userid'].astype(str).tolist()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        labels = df[self.action].values
        uauc = uAUC(labels, logits, userid_list)
        return df[["userid", "feedid"]], logits, uauc

    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action="all",
                                                                       day=STAGE_END_DAY[self.stage])
        submit_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(submit_dir)
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, self.stage, self.action)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time() - t) * 1000.0 / len(df) * 2000.0
        return df[["userid", "feedid"]], logits, ts


def del_file(path):
    '''
    删除path目录下的所有内容
    '''
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)


def main(argv):
    t = time.time()
    stage = argv[1] if len(argv) > 1 else 'offline_train'
    print('Stage: %s' % stage)
    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None
    for action in ACTION_LIST:
        print("Action:", action)
        model = WideAndDeep(stage, action)
        model.build_estimator()

        if stage in ["online_train", "offline_train"]:
            # 训练 并评估
            model.train()
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc

        if stage == "evaluate":
            # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc
            predict_dict[action] = logits

        if stage == "submit":
            # 预测线上测试集结果，保存预测结果
            ids, logits, ts = model.predict()
            predict_time_cost[action] = ts
            predict_dict[action] = logits

    if stage in ["evaluate", "offline_train", "online_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)

    if stage in ["evaluate", "submit"]:
        # 保存所有行为的预测结果，生成submit文件
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit_" + str(int(time.time())) + ".csv"
        submit_file = os.path.join(FLAGS.root_path, stage, file_name)
        print('Save to: %s' % submit_file)
        res.to_csv(submit_file, index=False)

    if stage == "submit":
        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s' % (time.time() - t))


if __name__ == "__main__":
    main(sys.argv)
