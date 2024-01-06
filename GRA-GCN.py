#!/usr/bin/env python
# coding: utf-8
import csv

import argparse, time, math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from copy import deepcopy
import warnings
import os
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import json

warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.model_selection import KFold

np.random.seed(1234)
torch.manual_seed(1234)

def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}


def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata['h'] = torch.mm(h, self.weight)
        self.g.update_all(gcn_msg, gcn_reduce, self.node_update)
        h = self.g.ndata.pop('h')
        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(g, in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(g, n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, n_classes, None, dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        probas = F.softmax(logits)

        #         correct = torch.sum(indices == labels)
        #         return correct.item() * 1.0 / len(labels)
        # print('=======================labels.cpu().detach():\n', labels.cpu().detach())
        return metrics(labels.cpu().detach(), indices.cpu().detach(), probas.cpu().detach()[:, 1])


def metrics(y_true, y_pred, y_prob):
    # print(y_true)
    y_true, y_pred, y_prob = y_true.numpy(), y_pred.numpy(), y_prob.numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


    pos_acc = tp / sum(y_true)
    neg_acc = tn / (len(y_pred) - sum(y_true))  # [y_true=0 & y_pred=0] / y_pred=0
    accuracy = (tp + tn) / (tn + fp + fn + tp)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    specificity = tn / (tn + fp)
    sensitivity = tp/(tp+fn)

    # print('======================Counter(y_pred)：\n', Counter(y_pred))

    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
    print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
    print(
        'acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(
            accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc))
    # return (y_true, y_pred, y_prob), (accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc)
    return (y_true, y_pred, y_prob), [aupr, roc_auc, f1, accuracy, recall, specificity, sensitivity]


def main(g, features, labels, train_idx, n_hidden, n_layers, dropout, n_epochs, lr):
    device = torch.device('cpu')
    cuda = False
    num_nodes = g.number_of_nodes()
    train_mask = np.zeros(num_nodes, dtype='int64')
    train_mask[train_idx] = 1
    test_mask = 1 - train_mask
    print(Counter(train_mask), Counter(test_mask))
    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)

    g.ndata['feat'] = features  
    g.ndata['label'] = labels
    g.ndata['train_mask'] = train_mask
    g.ndata['test_mask'] = test_mask

    g = g.to(device)

    in_feats = features.shape[1]
    n_classes = 2
    n_edges = g.number_of_edges()

    features, labels = features.to(device), labels.to(device)

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    # n_hidden = 512  # 64 256 32
    # n_layers = 3  # (2) 4 5
    # dropout = 0.1  # 0.1-0.7
    model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                F.relu,
                dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    # lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr,
                                 weight_decay=5e-4)

    # initialize graph
    dur = []
    # n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()

        t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        # loss.requires_grad=True
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()  

        dur.append(time.time() - t0)

        print('=====Epoch {} | Time(s) {:.4f} | Loss {:.4f} | ETputs(KTEPS) {:.2f}'.format(epoch, np.mean(dur),
                                                                                           loss.item(),
                                                                                           n_edges / np.mean(
                                                                                               dur) / 1000))
        ys_train, metrics_train = evaluate(model, features, labels, train_mask)
        ys_test, metrics_test = evaluate(model, features, labels, test_mask)
    probs = ys_test[2]
    np.save('probs.npy', probs)




    return ys_train, metrics_train, ys_test, metrics_test




def run(task, isbalance, n_neigh):
    # task = 'Tp'   isbalance = True    n_neigh = 5
    # pwd = '/home/chujunyi/4_GNN/GAEMDA-miRNA-disease/0_data/'

    if isbalance:
        node_feature_label = pd.read_csv('D:/PhD Folder/PhD research/All papers/Insuline protein/GRA-GCN/code/all.csv', index_col=0)
       

    # else:
    # node_feature_label = pd.read_csv('../GCN data/node_feature_label__nobalance.csv', index_col = 0)




    train_test_id_idx = np.load("D:/PhD Folder/PhD research/All papers/Insuline protein/GRA-GCN/graph/Tp__imbalanced__testlabel0_knn_edge_train_test_index_all.npz",
                                allow_pickle=True)


    train_index_all = train_test_id_idx['train_index_all']
    print(train_index_all)
    test_index_all = train_test_id_idx['test_index_all']
    print(test_index_all)
    num_nodes = node_feature_label.shape[0]
    features = torch.FloatTensor(np.array(node_feature_label.iloc[:, 3:]))
    labels = torch.LongTensor(np.array(node_feature_label['2']))   # 2是label列


    # fold = 0
    aupr_best = 0
    result_best = []
    best_parameters = {}
    header = ['n_hidden', 'n_layers', 'dropout', 'n_epochs', 'k', 'lr']
    with open('parameters.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)  
        writer.writeheader()  
    columns = ['aupr', 'roc_auc', 'f1', 'accuracy', 'recall', 'specificity', 'sensitivity']
    with open('results.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)  
        writer.writeheader() 
    for n_epochs in [1000]:  # parameters
        for i in [3]:
            for n_hidden in [256]:
                for n_layers in [2]:
                    for dropout in [0.7]:
                        for lr in [0.001]:
                            fold = 0
                            # fold = 2
                            result = np.zeros((1, 7))
                            for train_idx, test_idx in zip(train_index_all, test_index_all):  
                                # print(test_idx)
                                test_array = np.array(test_idx)
                                np.save('id.npy', test_array)
                                print('=====Fold {}============================================='.format(fold))
                                pwd = "D:/PhD Folder/PhD research/All papers/Insuline protein/GRA-GCN/graph/"
                                knn_graph_file = "task_Tp__imbalanced__testlabel0_knn" + str(i) + "neighbors_edge__fold" + str(fold) + '.npz'
                                knn_neighbors_graph = sp.load_npz(pwd + knn_graph_file) 
                            
                                edge_src = knn_neighbors_graph.nonzero()[0]  
                                edge_dst = knn_neighbors_graph.nonzero()[1]  

                                g = dgl.DGLGraph()
                                g.add_nodes(num_nodes)
                                g.add_edges(edge_src, edge_dst)
                                g = dgl.add_self_loop(g)
                                print(g)

                                ys_train, metrics_train, ys_test, metrics_test = main(g, features, labels, train_idx, n_hidden, n_layers, dropout, n_epochs, lr)
                                result += metrics_test
                                fold += 1
                            result_ave = result / 5
                            aupr_score = result_ave[0][0]
                            parameters = [{'n_hidden': n_hidden, 'n_layers': n_layers, 'dropout': dropout, 'n_epochs': n_epochs, 'k': i, 'lr': lr}]
                            result_dict = [{'aupr': result_ave[0][0], 'roc_auc': result_ave[0][1], 'f1': result_ave[0][2], 'accuracy': result_ave[0][3], 'recall': result_ave[0][4], 'specificity': result_ave[0][5], 'sensitivity': result_ave[0][6]}]
                            with open('parameters.csv', 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=header)  
                                writer.writerows(parameters)  
                            with open('results.csv', 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=columns)  
                                writer.writerows(result_dict)  
                            if aupr_score > aupr_best:
                                aupr_best = aupr_score
                                best_parameters = {'n_hidden': n_hidden, 'n_layers': n_layers, 'dropout': dropout, 'n_epochs': n_epochs, 'k': i, 'lr': lr}
                                result_best = result_ave
    print("best parameters:", best_parameters)
    print("result_best:", result_best)
    return node_feature_label, train_index_all, test_index_all, knn_neighbors_graph, g, ys_train, metrics_train, ys_test, metrics_test


# task = "Tp"
# n_neigh = 5
for task in ['Tp']:
    for n_neigh in [5]:
        isbalance = "True"
        print('************** isbalance = {} | task = {} | n_neigh = {}'.format(isbalance, task, n_neigh))
        node_feature_label, train_index_all, test_index_all, knn_neighbors_graph, g, ys_train, metrics_train, ys_test, metrics_test = run(
            task,
            isbalance,
            n_neigh)

