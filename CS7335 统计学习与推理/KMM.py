"""
Kernel Mean Matching
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from cvxopt import matrix, solvers
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse


def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta


def knn_classify(Xs, Ys, Xt, Yt, k=1, norm=False):
    model = KNeighborsClassifier(n_neighbors=k)
    Ys = Ys.ravel()
    Yt = Yt.ravel()
    if norm:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        Xt = scaler.fit_transform(Xt)
    model.fit(Xs, Ys)
    Yt_pred = model.predict(Xt)
    return Yt_pred
    # acc = accuracy_score(Yt, Yt_pred)
    # print(f'Accuracy using kNN: {acc * 100:.2f}%')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', action='store_true')
    args = parser.parse_args()
    
    feature_dim = 2048

    # source_df = pd.read_csv('OfficeHome/features/Art_Art.csv')
    # target_df = pd.read_csv('OfficeHome/features/Art_RealWorld.csv')
    source_df = pd.read_csv('OfficeHome/features/Clipart_Clipart.csv')
    target_df = pd.read_csv('OfficeHome/features/Clipart_RealWorld.csv')

    source_features = source_df.iloc[:, :feature_dim].values
    source_labels = source_df.iloc[:, feature_dim:].values.reshape(-1).astype(int)
    target_features = target_df.iloc[:, :feature_dim].values
    target_labels = target_df.iloc[:, feature_dim:].values.reshape(-1).astype(int)

    # print(f"Source Domain Name: Art, Features: {source_features.shape}, Labels: {source_labels.shape}")
    print(f"Source Domain Name: Clipart, Features: {source_features.shape}, Labels: {source_labels.shape}")
    print(f"Target Domain Name: RealWorld, Features: {target_features.shape}, Labels: {target_labels.shape}")

    kmm = KMM(kernel_type='rbf', B=10)
    beta = kmm.fit(source_features, target_features)
    Xs_new = beta * source_features
    predictions = knn_classify(Xs_new, source_labels, target_features, target_labels, k=1, norm=args.norm)

    print("acc count:", (predictions==target_labels).sum())
    print("total count:", len(target_labels))
    print("success rate:", (predictions==target_labels).sum()/len(target_labels))
    target_label_set = list(set(target_labels))
    acc_list = [0 for _ in range(len(target_label_set))]
    total_list = [0 for _ in range(len(target_label_set))]
    for pred, label in zip(predictions, target_labels):
        total_list[label] += 1
        if pred == label:
            acc_list[label] += 1 

    rate = []
    acc_count = 0
    for idx in target_label_set:
        rate.append(acc_list[idx]/total_list[idx])
        acc_count += acc_list[idx]
    canvas = plt.figure()
    plt.cla()
    plt.bar(target_label_set, rate)
    plt.xlabel("Labels")
    plt.ylabel("Accuracy")
    # plt.title("Art->RealWorld Domain Adaptation Predictions")
    plt.title("Clipart->RealWorld Domain Adaptation Predictions")
    plt.xlim(-1,65)
    # plt.savefig('KMM_A2R')
    plt.savefig('KMM_C2R')
    plt.clf()
    plt.show()
