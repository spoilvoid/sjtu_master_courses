import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__=="__main__":
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

    SVM = LinearSVC()
    SVM.fit(source_features, source_labels)
    predictions = SVM.predict(target_features)

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
    # plt.savefig('SVM_A2R')
    plt.savefig('SVM_C2R')
    plt.clf()
    plt.show()