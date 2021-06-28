# coding: utf-8

import os, sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from comm import ACTION_LIST, STAGE_END_DAY, FEA_COLUMN_LIST
from evaluation import uAUC, compute_weighted_score

model_dir = './data/model'
root_path = './data/'

SEED = 2021


class CatboostTrainer(object):

    def __init__(self, cate_features, feature_columns, stage, action):
        """
        :param cate_features: List of tensorflow feature_column
        :param feature_columns: List of tensorflow feature_column
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param action: String. Including "read_comment"/"like"/"click_avatar"/"favorite"/"forward"/"comment"/"follow"
        """
        super(CatboostTrainer, self).__init__()
        self.estimator = None
        self.cate_features = cate_features
        self.feature_columns = feature_columns
        self.stage = stage
        self.action = action

    def build_estimator(self):
        if self.stage in ["evaluate", "offline_train"]:
            stage = "offline_train"
        else:
            stage = "online_train"
        model_stage_dir = os.path.join(model_dir, stage, self.action)
        if not os.path.exists(model_stage_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(model_stage_dir)

        self.estimator = CatBoostClassifier(iterations=1000, cat_features=self.cate_features,
                                            eval_metric='AUC', loss_function='CrossEntropy', logging_level='Verbose',
                                            learning_rate=0.05, depth=6, l2_leaf_reg=5)

    def train(self):
        """
        训练单个行为的模型
        """
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action=self.action,
                                                                       day=STAGE_END_DAY[self.stage])
        stage_dir = os.path.join(root_path, self.stage, file_name)
        df = pd.read_csv(stage_dir)

        for col in self.cate_features:
            df[col] = df[col].fillna(-1).astype("int")
        df = df[self.feature_columns + [self.action]].fillna(0)

        if self.stage == "offline_train":
            eval_stage = "evaluate"
            eval_action = "all"
            eval_file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=eval_stage, action=eval_action,
                                                                                day=STAGE_END_DAY[eval_stage])
            evaluate_dir = os.path.join(root_path, eval_stage, eval_file_name)
            eval_df = pd.read_csv(evaluate_dir)

            for col in self.cate_features:
                eval_df[col] = eval_df[col].fillna(-1).astype("int")
            eval_df = eval_df[self.feature_columns + [self.action]].fillna(0)

            self.estimator.fit(
                df[self.feature_columns], df[self.action],
                eval_set=(eval_df[self.feature_columns], eval_df[self.action]), plot=True
            )
        else:
            self.estimator.fit(
                df[self.feature_columns], df[self.action]
            )

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
        stage = self.stage
        if self.stage == 'offline_train':
            stage = "evaluate"
            action = "all"
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=stage, action=action,
                                                                       day=STAGE_END_DAY[stage])
        evaluate_dir = os.path.join(root_path, stage, file_name)
        df = pd.read_csv(evaluate_dir)
        userid_list = df['userid'].astype(str).tolist()
        logits = self.estimator.predict_proba(
            df[self.feature_columns]
        )[:, 1]
        labels = df[self.action].values
        uauc = uAUC(labels, logits, userid_list)
        return df[["userid", "feedid"]], logits, uauc

    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        stage = "submit"
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=stage, action="all",
                                                                       day=STAGE_END_DAY[stage])
        submit_dir = os.path.join(root_path, stage, file_name)
        df = pd.read_csv(submit_dir)
        t = time.time()
        logits = self.estimator.predict_proba(
            df[self.feature_columns]
        )[:, 1]
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


def get_feature_columns():
    '''
    获取特征列
    '''
    feature_columns = list()
    cate_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']

    feature_columns.append("videoplayseconds")
    feature_columns.append("device")
    # 行为统计特征
    for b in FEA_COLUMN_LIST:
        feature_columns.append(b + "sum")
        feature_columns.append(b + "sum_user")
        feature_columns.append(b + "sum_author")
        feature_columns.append(b + "sum_song")
        feature_columns.append(b + "sum_singer")
        feature_columns.append(b + "sum_user_author")
        feature_columns.append(b + "sum_user_song")
        feature_columns.append(b + "sum_user_singer")
    for col in cate_features:
        if col in feature_columns:
            feature_columns.remove(col)
    feature_columns += cate_features
    return feature_columns, cate_features


def main(argv):
    t = time.time()
    feature_columns, cate_features = get_feature_columns()
    stage = argv[1]
    print('Stage: %s' % stage)
    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None
    for action in ACTION_LIST:
        print("Action:", action)
        model = CatboostTrainer(cate_features, feature_columns, stage, action)
        model.build_estimator()

        if stage == "offline_train":
            # 训练 并评估
            model.train()
            ids, logits, action_uauc = model.evaluate()
            eval_dict[action] = action_uauc

        if stage == "online_train":
            model.train()
            # 预测线上测试集结果，保存预测结果
            ids, logits, ts = model.predict()
            predict_time_cost[action] = ts
            predict_dict[action] = logits

    if stage in ["offline_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)

    if stage in ["online_train"]:
        # 保存所有行为的预测结果，生成submit文件
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit_" + str(int(time.time())) + ".csv"
        submit_file = os.path.join(root_path, "submit", file_name)
        print('Save to: %s' % submit_file)
        res.to_csv(submit_file, index=False)

        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s' % (time.time() - t))


if __name__ == "__main__":
    main(sys.argv)
