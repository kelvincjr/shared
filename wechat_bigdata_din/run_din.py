# -*- coding:utf-8 -*-
"""
@author : LMC_ZC
reference : https://github.com/shenweichen/DeepCTR-Torch/
"""

#path = '/opt/kelvin/python/knowledge_graph/e-commerce/wechat_bigdata/wechat_bigdata_din'
path = '/kaggle/working/wechat_bigdata_din'
import sys
sys.path.append(path) # for the fuck clouds


import torch
import numpy as np
from models.din import DeepInterestNetwork
from codes.dataset import Dataset
from torch.utils.data import DataLoader
from session.session import Session
from codes.utils import get_user_mask_df

# for validation
"""
if __name__ == '__main__':
    predict_label = ['read_comment', 'like', 'click_avatar', 'forward']
    all_num_epochs = [4, 4, 4, 4]
    
    D = Dataset(path+'/data/wechat_algo_data1/processed/data.csv', path+'/data/wechat_algo_data1/processed/statistical.pkl')
    feature_columns, behavior_feature_list, seq_length_list = D.get_feature_columns()
    
    print("split dataset ......")
    train_df, val_df = D.get_validation(path+'/data/wechat_algo_data1/processed/train_df.csv', path+'/data/wechat_algo_data1/processed/val_df.csv', hasfile=True, split_ratio=0.1)
    # print("split dataset ok")
    
    print("load dataset ......")
    train_x, train_y = D.get_xy(train_df, feature_columns, path+'/data/wechat_algo_data1/processed/feed_embeddings.npy', has_y=True)

    for i, epoch_and_label in enumerate(zip(all_num_epochs, predict_label)):  
        num_epochs, label = epoch_and_label[0], epoch_and_label[1]
        
        # process validation set for uauc
        val_df = get_user_mask_df(val_df, label) # filter for evaluation
        val_x, val_y = D.get_xy(val_df, feature_columns, path+'/data/wechat_algo_data1/processed/feed_embeddings.npy', has_y=True)
        
        din = DeepInterestNetwork(feature_columns, behavior_feature_list, seq_length_list, device='cpu')
        
        train_y_label = train_y[label]
        val_y_label = val_y[label]
        
        train_tensor_data, x_table = din.generate_loader(train_x, train_y_label, None, True)
        train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=2048, num_workers=0)
        
        sess = Session(din)
        sess.compile(0.001, 'adam', 'binary_crossentropy', ['auc'])
        
        print("training {0}......".format(label))
        for epoch in range(num_epochs):
            logs = sess.train(train_loader)
            print("Epoch {0}/{1}, loss: {2:.4f}".format(epoch + 1, num_epochs, logs['loss']))
            train_result = sess.evaluate(train_x, train_y_label, x_table, batch_size=2048)
            print("train auc: {:.6f}".format(train_result['auc']))
            test_result = sess.evaluate(val_x, val_y_label, x_table, batch_size=2048)
            print("test auc: {:.6f}".format(test_result['auc']))
"""  
# for test
if __name__ == '__main__':
    train_D = Dataset('./data/wechat_algo_data1/processed/data.csv', './data/wechat_algo_data1/processed/statistical.pkl')
    test_D = Dataset('./data/wechat_algo_data1/processed/test.csv', './data/wechat_algo_data1/processed/statistical.pkl')
    
    predict_label = ['read_comment', 'like', 'click_avatar', 'forward']
    all_num_epochs = [3, 3, 3, 3]
    all_predicts = []
    
    for i, epoch_and_label in enumerate(zip(all_num_epochs, predict_label)):
        
        num_epochs, label = epoch_and_label[0], epoch_and_label[1]
        
        feature_columns, behavior_feature_list, seq_length_list = train_D.get_feature_columns()
        din = DeepInterestNetwork(feature_columns, behavior_feature_list, seq_length_list, device='cuda:0')
        train_x, train_y = train_D.get_xy(train_D.data, feature_columns, path+'/data/wechat_algo_data1/processed/feed_embeddings.npy', has_y=True)
        test_x = test_D.get_xy(test_D.data, feature_columns, path+'/data/wechat_algo_data1/processed/feed_embeddings.npy', has_y=False)
        
        train_y = train_y[label]

        train_tensor_data, x_table = din.generate_loader(train_x, train_y, None, True)
        test_tensor_data = din.generate_loader(test_x, None, x_table,  False)
        
        train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=64, num_workers=0)
        test_loader = DataLoader(test_tensor_data, shuffle=False, batch_size=64, num_workers=0)
        sess = Session(din)
        sess.compile(0.0005, 'adam', 'binary_crossentropy', ['auc'])
        
        for epoch in range(num_epochs):
            logs = sess.train(train_loader)
            print("Epoch {0}/{1}, loss: {2:.4f}".format(epoch+1, num_epochs, logs['loss']))
            preds = sess.predict(test_loader)
            np.savetxt(label + '_result_' + str(epoch) + '.csv', preds, delimiter=',')
            torch.save(din.state_dict(), path+'/parameters/' + label + str(epoch) + '.pth') 
            
        #preds = sess.predict(test_loader)
        #np.savetxt(label +'_result.csv', preds, delimiter=',')
        #all_predicts.append(np.expand_dims(preds, axis=1))
        print('------' + label + ' is ok -----' )
    #result = np.concatenate(all_predicts, axis=1)
    #np.savetxt('result.csv', result, delimiter=',')
