import warnings

warnings.filterwarnings("ignore")
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir

from itertools import product
from collections import Counter
from copy import deepcopy
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

import random

random.seed(1234)
np.random.seed(1234)


# ======================================
def load_data(directory):
    #IP = pd.read_csv(directory +'\data.csv')
    IP=pd.read_csv('k-pssm-dwt-insulin.csv')
    IP = pd.DataFrame(IP).reset_index()
    IP.rename(columns={'index': 'id'}, inplace=True)
    IP['id'] = IP['id'] + 1
    # print('===============IP.shape:',IP.shape)   (1952, 401)
    return IP


def sample(directory, random_seed):
    all_associations = pd.read_csv('protein_label1.csv', names=['ID', '#', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    # print('==============================sample_df.shape:', sample_df.shape)
    return sample_df  # (492 rows × 3 columns)


def obtain_data(directory, isbalance):

    IP = load_data(directory)

    # isbalance = False 执行else后的程序

    if isbalance:
        dtp = pd.read_csv('protein_label1.csv',
                          names=['ID', '#', 'label'])  # [ 1952 rows x 3 columns]
    else:
        dtp = sample(directory, random_seed=1234)  
        # 保存成csv文件查看所有数据

    protein_ids = list(set(dtp['ID']))
    random.shuffle(protein_ids)
    print('# protein = {}'.format(len(protein_ids)))

    protein_test_num = int(len(protein_ids) / 5)
    print('# Test: protein = {}'.format(protein_test_num))
    # print(dtp)
    # print('# Test: miRNA = {} | Disease = {}'.format(mirna_test_num, disease_test_num))
    # knn_x = pd.read_csv(directory + '\data.csv')
    knn_x = pd.merge(dtp, IP, left_on='ID', right_on='id')

    # ========================================

    X = np.array(knn_x)
    print(X.shape)
    pd.DataFrame(X).to_csv(r"all.csv", index=None, encoding='utf-8')

    # ========================================


    label = dtp['label']
    # print(knn_x)
    knn_x.drop(labels=['ID', '#', 'label', 'id'], axis=1, inplace=True)  
    # print(knn_x)

    return IP, dtp, protein_ids, protein_test_num, knn_x, label




def generate_task_Tp_train_test_idx(knn_x):
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)

    train_index_all, test_index_all, n = [], [], 0
    train_id_all, test_id_all = [], []
    fold = 0

    for train_idx, test_idx in tqdm(kf.split(knn_x)):  # train_index与test_index为下标
        print('-------Fold ', fold)
        train_index_all.append(train_idx)
        test_index_all.append(test_idx)
 
        train_id_all.append(np.array(dtp.iloc[train_idx][['ID']]))
        test_id_all.append(np.array(dtp.iloc[test_idx][['ID']]))

        print('# Pairs: Train = {} | Test = {}'.format(len(train_idx), len(test_idx)))
        # print(train_index_all)

        fold += 1

    return train_index_all, test_index_all, train_id_all, test_id_all


def generate_task_Tm_Td_train_test_idx(item, ids, dtp):
    test_num = int(len(ids) / 5)

    train_index_all, test_index_all = [], []
    train_id_all, test_id_all = [], []

    for fold in range(5):
        print('-------Fold ', fold)
        if fold != 4:
            test_ids = ids[fold * test_num: (fold + 1) * test_num]
        else:
            test_ids = ids[fold * test_num:]

        train_ids = list(set(ids) ^ set(test_ids))
        print('# {}: Train = {} | Test = {}'.format(item, len(train_ids), len(test_ids)))

        test_idx = dtp[dtp[item].isin(test_ids)].index.tolist()
        train_idx = dtp[dtp[item].isin(train_ids)].index.tolist()
        random.shuffle(test_idx)
        random.shuffle(train_idx)
        print('# Pairs: Train = {} | Test = {}'.format(len(train_idx), len(test_idx)))
        # print('test_idx',test_idx)
        assert len(train_idx) + len(test_idx) == len(dtp)

        train_index_all.append(train_idx)
        test_index_all.append(test_idx)

        train_id_all.append(train_ids)
        test_id_all.append(test_ids)

    return train_index_all, test_index_all, train_id_all, test_id_all


'''
KNN
'''
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report


def generate_knn_graph_save(knn_x, label, n_neigh, train_index_all, test_index_all, pwd, task, balance):
    fold = 0
    for train_idx, test_idx in zip(train_index_all, test_index_all):
        print('-------Fold ', fold)

        knn_y = deepcopy(label)  
        knn_y[test_idx] = 0  


        knn = KNeighborsClassifier(n_neighbors=n_neigh)  
        knn.fit(knn_x, knn_y) 

        knn_y_pred = knn.predict(knn_x)  
        # print('=======================knn_y_pred:', knn_y_pred)
        knn_y_prob = knn.predict_proba(knn_x)
        # print('-------------knn_y_proba:', knn_y_prob)
        knn_neighbors_graph = knn.kneighbors_graph(knn_x, n_neighbors=n_neigh)  

        prec_reca_f1_supp_report = classification_report(knn_y, knn_y_pred, target_names=['label_0', 'label_1'])
        tn, fp, fn, tp = confusion_matrix(knn_y, knn_y_pred).ravel()

        pos_acc = tp / sum(knn_y)
        neg_acc = tn / (len(knn_y_pred) - sum(knn_y_pred))  # [y_true=0 & y_pred=0] / y_pred=0
        accuracy = (tp + tn) / (tn + fp + fn + tp)

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)

        roc_auc = roc_auc_score(knn_y, knn_y_prob[:, 1])
        prec, reca, _ = precision_recall_curve(knn_y, knn_y_prob[:, 1])
        aupr = auc(reca, prec)

        print(
            'acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(
                accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc))
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: ', Counter(knn_y_pred))
        print('y_true: ', Counter(knn_y))

        # print('knn_score = {:.4f}'.format(knn.score(knn_x, knn_y)))

        sp.save_npz(pwd + 'task_' + task + balance + '__testlabel0_knn' + str(n_neigh) + 'neighbors_edge__fold' + str(
            fold) + '.npz', knn_neighbors_graph)
        fold += 1
    return knn_x, knn_y, knn, knn_neighbors_graph


'''
Run
'''
# for isbalance in [True, False]:
# for isbalance in [False, True]:  # 只保留False，只跑平衡数据
# for isbalance in [False]:
for isbalance in [True]:
    print('************isbalance = ', isbalance)

    for task in ['Tp']:
        print('=================task = ', task)

        IP, dtp, protein_ids, protein_test_num, knn_x, label = obtain_data('D:\\PhD Folder\PhD research\All papers\Insuline protein\GRA-GCN\data',
                                                                           isbalance)

        if task == 'Tp':
            train_index_all, test_index_all, train_id_all, test_id_all = generate_task_Tp_train_test_idx(knn_x)

        if isbalance:
            balance = '__imbalanced'

        else:
            balance = '__balanced'

        np.savez_compressed(
            'D:/PhD Folder/PhD research/All papers/Insuline protein/GRA-GCN/graph/' + task + balance + '__testlabel0_knn_edge_train_test_index_all.npz',
            train_index_all=train_index_all,
            test_index_all=test_index_all,
            train_id_all=train_id_all,
            test_id_all=test_id_all)

        pwd = 'D:/PhD Folder/PhD research/All papers/Insuline protein/GRA-GCN/graph/'
        for n_neigh in [3, 5, 7, 9, 11, 13]:
            print('--------------------------n_neighbors = ', n_neigh)
            knn_x, knn_y, knn, knn_neighbors_graph = generate_knn_graph_save(knn_x, label, n_neigh, train_index_all,
                                                                             test_index_all, pwd, task, balance)

pwd = 'D:/PhD Folder/PhD research/All papers/Insuline protein/GRA-GCN/graph/'

directory = "D:\\PhD Folder\PhD research\All papers\Insuline protein\GRA-GCN\data"
