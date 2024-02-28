# encoding=utf-8
"""
    Created on 16:31 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors


class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        np.linalg.multi_dot([Xs, scipy.linalg.fractional_matrix_power(cov_src, -0.5), scipy.linalg.fractional_matrix_power(cov_tar, 0.5)])
        Xs_new = np.real(np.dot(Xs, A_coral))
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        return y_pred


if __name__ == '__main__':
    feature_dim = 2048

    source_df = pd.read_csv('OfficeHome/features/Art_Art.csv')
    target_df = pd.read_csv('OfficeHome/features/Art_RealWorld.csv')
    # source_df = pd.read_csv('OfficeHome/features/Clipart_Clipart.csv')
    # target_df = pd.read_csv('OfficeHome/features/Clipart_RealWorld.csv')

    source_features = source_df.iloc[:, :feature_dim].values
    source_labels = source_df.iloc[:, feature_dim:].values.reshape(-1).astype(int)
    target_features = target_df.iloc[:, :feature_dim].values
    target_labels = target_df.iloc[:, feature_dim:].values.reshape(-1).astype(int)

    print(f"Source Domain Name: Art, Features: {source_features.shape}, Labels: {source_labels.shape}")
    # print(f"Source Domain Name: Clipart, Features: {source_features.shape}, Labels: {source_labels.shape}")
    print(f"Target Domain Name: RealWorld, Features: {target_features.shape}, Labels: {target_labels.shape}")

    coral = CORAL()
    predictions = coral.fit_predict(source_features, source_labels, target_features, target_labels)

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
    plt.title("Art->RealWorld Domain Adaptation Predictions")
    # plt.title("Clipart->RealWorld Domain Adaptation Predictions")
    plt.xlim(-1,65)
    plt.savefig('CORAL_A2R')
    # plt.savefig('CORAL_C2R')
    plt.clf()
    plt.show()