import os
import pandas as pd
import numpy as np
import tensorflow as tf

from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from mmoe import MMOE
from evaluation import evaluate_deepctr

# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 设置GPU按需增长
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
'''
if tf.__version__.startswith('1.'):  # tensorflow 1
    config = tf.ConfigProto()  # allow_soft_placement=True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
else:  # tensorflow 2
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import os,signal
import psutil
def mem_info():
    pid = os.getpid()
    p = psutil.Process(pid)
    mem_info = p.memory_full_info()
    memory = mem_info.uss / 1024. / 1024.
    print('Memory used: {:.2f} MB'.format(memory))

def deep_mmoe(df):
    epochs = 1 #5
    batch_size = 256 #512
    embedding_dim = 128 #512
    #df = pd.read_csv("data/lgb.csv")
    feed_embeddings = pd.read_csv("data/wechat_algo_data1/feed_embeddings.csv")
    feed_embeddings['feed_embedding'] = feed_embeddings['feed_embedding'].apply(
        lambda x: list(map(float, x.strip().split())))
    #feed_embedding = np.array(feed_embeddings['feed_embedding'].values.tolist())
    #print('feed_embedding shape[0] {}'.format(feed_embedding.shape[0]))
    
    embeddings_index = {}
    for i, row in feed_embeddings.iterrows():
        feedid = row['feedid']
        feed_embedding = row['feed_embedding']
            # coefs = np.fromstring(row['feed_embedding'], "f", sep=" ")
        embeddings_index[feedid] = np.asarray(feed_embedding, dtype=float)
    num_tokens = feed_embeddings['feedid'].max() + 1
    feed_embedding_dim = len(feed_embeddings.feed_embedding[0])
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, feed_embedding_dim))
    for feedid, embedding in embeddings_index.items():
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[feedid] = embedding

    del feed_embeddings
    del embeddings_index
    print('check point after feed_embedding')
    mem_info()
    data = df[~df['read_comment'].isna()].reset_index(drop=True)
    test = df[df['read_comment'].isna()].reset_index(drop=True)
    del df
    print('check point after data, test')
    mem_info()

    play_cols = ['is_finish', 'play_times', 'play', 'stay']
    y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
    cols = [f for f in data.columns if f not in ['date_'] + play_cols + y_list]
    target = ["read_comment", "like", "click_avatar", "forward"]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    dense_features = [f for f in cols if f not in sparse_features]

    print('check point after dense_features')
    mem_info()
    # 1.fill nan dense_feature and do simple Transformation for dense features
    for feat in dense_features:
        data[feat] = data[feat].fillna(0, )
        test[feat] = test[feat].fillna(0, )

    for feat in dense_features:
        data[feat] = np.log(data[feat] + 1.0)
        test[feat] = np.log(test[feat] + 1.0)

    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())
    print('check point after fill nan')
    mem_info()

    train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]  # 第14天样本作为验证集
    pretrained_feed_embedding_initializer = tf.initializers.identity(feed_embedding)

    print('feedid max() + 1 {}'.format(data['feedid'].max() + 1))
    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat('feedid', vocabulary_size=data['feedid'].max() + 1, embedding_dim=512,
    #fixlen_feature_columns = [SparseFeat('feedid', vocabulary_size=feed_embedding.shape[0], embedding_dim=512,                                        
                                         embeddings_initializer=pretrained_feed_embedding_initializer)] + [
                                 SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                                 for feat in sparse_features if feat is not 'feedid'] + [DenseFeat(feat, 1) for feat in
                                                                                         dense_features]
    del data
    print('check point after train test')
    mem_info()

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # 3.generate input data for model
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]

    del train, val
    print('check point after train_model_input val_model_input')
    mem_info()

    # 4.Define Model,train,predict and evaluate
    train_model = MMOE(dnn_feature_columns, num_tasks=4, expert_dim=8, dnn_hidden_units=(128, 128),
                       tasks=['binary', 'binary', 'binary', 'binary'])
    train_model.compile("adagrad", loss='binary_crossentropy')
    print(train_model.summary())
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size) # * 4)
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)

    t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size) # * 20)
    t2 = time()
    print('4个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    print('4个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # 5.生成提交文件
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
    test[['userid', 'feedid'] + target].to_csv('result.csv', index=None, float_format='%.6f')
    print('to_csv ok')


if __name__ == "__main__":
    epochs = 5
    batch_size = 512
    embedding_dim = 512
    df = pd.read_csv("data/lgb.csv")
    feed_embeddings = pd.read_csv("data/wechat_algo_data1/feed_embedding.csv")
    feed_embeddings['feed_embedding'] = feed_embeddings['feed_embedding'].apply(
        lambda x: list(map(float, x.strip().split())))
    feed_embedding = np.array(feed_embeddings['feed_embedding'].values.tolist())
    data = df[~df['read_comment'].isna()].reset_index(drop=True)
    test = df[df['read_comment'].isna()].reset_index(drop=True)
    del df

    play_cols = ['is_finish', 'play_times', 'play', 'stay']
    y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
    cols = [f for f in data.columns if f not in ['date_'] + play_cols + y_list]
    target = ["read_comment", "like", "click_avatar", "forward"]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    dense_features = [f for f in cols if f not in sparse_features]

    # 1.fill nan dense_feature and do simple Transformation for dense features
    data[dense_features] = data[dense_features].fillna(0, )
    test[dense_features] = test[dense_features].fillna(0, )

    data[dense_features] = np.log(data[dense_features] + 1.0)
    test[dense_features] = np.log(test[dense_features] + 1.0)

    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())

    train = data[data['date_'] < 14]
    val = data[data['date_'] == 14]  # 第14天样本作为验证集
    pretrained_feed_embedding_initializer = tf.initializers.identity(feed_embedding)

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat('feedid', vocabulary_size=data['feedid'].max() + 1, embedding_dim=512,
                                         embeddings_initializer=pretrained_feed_embedding_initializer)] + [
                                 SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                                 for feat in sparse_features if feat is not 'feedid'] + [DenseFeat(feat, 1) for feat in
                                                                                         dense_features]
    del data
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # 3.generate input data for model
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]

    del train, val

    # 4.Define Model,train,predict and evaluate
    train_model = MMOE(dnn_feature_columns, num_tasks=4, expert_dim=8, dnn_hidden_units=(128, 128),
                       tasks=['binary', 'binary', 'binary', 'binary'])
    train_model.compile("adagrad", loss='binary_crossentropy')
    # print(train_model.summary())
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)

    t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    t2 = time()
    print('4个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    print('4个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # 5.生成提交文件
    for i, action in enumerate(target):
        test[action] = pred_ans[i]
    test[['userid', 'feedid'] + target].to_csv('result.csv', index=None, float_format='%.6f')
    print('to_csv ok')
