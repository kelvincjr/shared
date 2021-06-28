"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import random

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

from mmoe import MMoE

SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)


# Fix TensorFlow graph-level seed for reproducibility
# tf.set_random_seed(SEED)
# tf_session = tf.Session(graph=tf.get_default_graph())
# K.set_session(tf_session)


# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def data_preparation():
    # The column names are from
    # https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    df = pd.read_csv("./data/lgb.csv")
    play_cols = ['is_finish', 'play_times', 'play', 'stay']
    y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
    max_day = 15
    train = df[~df['read_comment'].isna()].reset_index(drop=True)
    test = df[df['read_comment'].isna()].reset_index(drop=True)
    trn_x = train[train['date_'] < 14].reset_index(drop=True)
    val_x = train[train['date_'] == 14].reset_index(drop=True)
    trn_x = trn_x.dropna()
    # Load the dataset in Pandas
    # First group of tasks according to the paper
    label_columns = y_list[:4]

    # One-hot encoding categorical columns
    # categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
    #                        'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
    #                        'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
    #                        'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
    #                        'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
    #                        'vet_question']
    cols = [f for f in df.columns if f not in ['date_'] + play_cols + y_list]
    trn_x_raw_labels = trn_x[label_columns]
    val_x_raw_labels = val_x[label_columns]
    # transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    # transformed_other = pd.get_dummies(other_df.drop(label_columns, axis=1), columns=categorical_columns)

    # Filling the missing column in the other set
    # transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    # One-hot encoding categorical labels
    # ['read_comment', 'like', 'click_avatar', 'forward'
    trn_x_read_comment = to_categorical((trn_x_raw_labels.read_comment != 0.0).astype(int), num_classes=2)
    trn_x_like = to_categorical((trn_x_raw_labels.like != 0.0).astype(int), num_classes=2)
    trn_x_click_avatar = to_categorical((trn_x_raw_labels.click_avatar != 0.0).astype(int), num_classes=2)
    trn_x_forward = to_categorical((trn_x_raw_labels.forward != 0.0).astype(int), num_classes=2)

    val_x_read_comment = to_categorical((val_x_raw_labels.read_comment != 0.0).astype(int), num_classes=2)
    val_x_like = to_categorical((val_x_raw_labels.like != 0.0).astype(int), num_classes=2)
    val_x_click_avatar = to_categorical((val_x_raw_labels.click_avatar != 0.0).astype(int), num_classes=2)
    val_x_forward = to_categorical((val_x_raw_labels.forward != 0.0).astype(int), num_classes=2)

    # other_income = to_categorical((other_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    # other_marital = to_categorical((other_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)

    dict_outputs = {
        'read_comment': trn_x_read_comment.shape[1],
        'like': trn_x_like.shape[1],
        'click_avatar': trn_x_click_avatar.shape[1],
        'forward': trn_x_forward.shape[1]
    }
    dict_train_labels = {
        'read_comment': trn_x_read_comment,
        'like': trn_x_like,
        'click_avatar': trn_x_click_avatar,
        'forward': trn_x_forward,
    }
    dict_other_labels = {
        'read_comment': val_x_read_comment,
        'like': val_x_like,
        'click_avatar': val_x_click_avatar,
        'forward': val_x_forward,
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    trn_x = trn_x[cols]
    val_x = val_x[cols]
    validation_indices = val_x.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(val_x.index) - set(validation_indices))
    validation_data = val_x.iloc[validation_indices]
    validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data = val_x.iloc[test_indices]
    test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data = trn_x
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info


def main():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    # Set up the input layer
    input_layer = Input(shape=(num_features,))

    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=16,
        num_experts=8,
        num_tasks=4
    )(input_layer)

    output_layers = []

    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=128,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=output_info[index][0],
            name=output_info[index][1],
            activation='softmax',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    adam_optimizer = Adam()
    model.compile(
        loss={'read_comment': 'binary_crossentropy',
              'like': 'binary_crossentropy',
              'click_avatar': 'binary_crossentropy',
              'forward': 'binary_crossentropy'
              },
        optimizer=adam_optimizer,
        metrics=['accuracy']
    )

    # Print out model architecture summary
    model.summary()

    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ],
        batch_size=256,
        epochs=100
    )
    model.save("my_model.h5")

if __name__ == '__main__':
    main()
