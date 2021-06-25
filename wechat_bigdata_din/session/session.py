# -*- coding:utf-8 -*-

"""
@author : LMC_ZC

reference : https://github.com/shenweichen/DeepCTR-Torch/

"""


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score
from tqdm import tqdm


class Session(object):

    def __init__(self, model, verbose=1):

        self.model = model
        self.verbose = verbose

    def compile(self, lr, optimizer, loss=None, metrics=None):
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer, lr=lr)
        self.loss_func = self._get_loss(loss)
        self.metrics = self._get_metrics(metrics)

    """
    def get_emb(self, train_loader, test_loader):

        self.model.train()

        train_att_emb, test_att_emb = [], []

        with torch.no_grad():
            with tqdm(enumerate(test_loader), disable=self.verbose != 1) as t:
                for _, x_train in t:
                    x = x_train.to(self.model.device).float()
                    y_pred, att_emb = self.model(x)

                    test_att_emb += [att_emb]

            with tqdm(enumerate(train_loader), disable=self.verbose != 1) as t:
                for _, (x_train, y_train) in t:
                    x = x_train.to(self.model.device).float()
                    y = y_train.to(self.model.device).float()
                    y_pred, att_emb = self.model(x)

                    train_att_emb += [att_emb]

        return train_att_emb, test_att_emb
    """

    def train(self, train_loader):

        self.model.train()
        logs = {}
        total_loss_epoch = 0.0
        train_result = {}
        sample_num = len(train_loader)
        run_sample_num = [sample_num] * len(self.metrics)

        try:
            with tqdm(enumerate(train_loader), disable=self.verbose != 1) as t:
                for _, (x_train, y_train) in t:

                    x = x_train.to(self.model.device).float()
                    y = y_train.to(self.model.device).float()

                    y_pred = self.model(x).squeeze()
                    lls = self.loss_func(y_pred, y)
                    reg_loss = self.model.get_regularization_loss()
                    loss = lls + reg_loss

                    total_loss_epoch += loss.item()

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    if self.verbose > 0:
                        for i, data in enumerate(self.metrics.items()):
                            name, metric_fun = data[0], data[1]
                            if name not in train_result:
                                train_result[name] = []

                            try:
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float32")))
                            except ValueError:

                                # 这个地方的代码主要是因为, 样本取得有时候不均匀, 然后就可能会导致计算比如AUC的时候, 全部都是正/负样本,
                                # 然后代码就会报错, 但是这种情况不好在采样的时候避免, 因为我用DataLoader封装的, 所以直接咱就把这次计算
                                # 的指标作废, 简单而不失优雅
                                run_sample_num[i] = run_sample_num[i] - 1
                                continue

        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        logs["loss"] = total_loss_epoch / sample_num
        for i, data in enumerate(train_result.items()):
            name, result = data[0], data[1]
            logs[name] = np.sum(result) / run_sample_num[i]

        return logs

    def evaluate(self, val_x, val_y, table, batch_size=1024):
        eval_result = {}
        val_tensor_data = self.model.generate_loader(val_x, None, table, False)
        val_loader = DataLoader(val_tensor_data, shuffle=False, batch_size=batch_size)
        pred_ans = self.predict(val_loader)
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(val_y, pred_ans)

        return eval_result

    def predict(self, val_loader):

        self.model.eval()
        pred_ans = []
        with torch.no_grad():
            for x_test in val_loader:
                # pdb.set_trace()
                x = x_test.to(self.model.device).float()
                y_pred = self.model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def _get_optim(self, optimizer, **kwargs):

        if isinstance(optimizer, str):
            lr = kwargs['lr']
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.model.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.model.parameters(), lr=lr)
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.model.parameters(), lr=lr)
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.model.parameters(), lr=lr)
            else:
                raise NotImplementedError

        else:
            optim = optimizer

        return optim

    def _get_loss(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_
    
    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)
