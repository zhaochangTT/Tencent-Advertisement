# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

import hashlib, csv, math, os, pickle, subprocess

#hashlib提供了常见的摘要算法，如MD5，SHA1
#import hashlib
#md5 = hashlib.md5()
#md5.update('how to use md5 in python hashlib?')
#print md5.hexdigest()
#计算结果如下：d26a53750bc40b38b65a520292f69306


def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

#这个函数暂时没看懂
def gen_hashed_fm_feats(feats, nr_bins = int(1e+6)):
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

# 特征分类

combined_feats = ['kw1','kw2','kw3','topic1','topic2','topic3', 'interest1','interest2','interest3',
                  'interest4','interest5','ct', 'marriageStatus', 'os']
common_feats = ['creativeId','campaignId','LBS','advertiserId','appIdInstall', 'appIdAction', 'aid', 'advertiserId', 'campaignId', 'creativeId', 'age','creativeSize']
one_hot_feats = []
# 没有用的特征
remain_feats = ['uid', 'label', '', 'uid.1','uid.2','uid.3','uid.4','uid.5','uid.6','uid.7','uid.8']
print ("reading data")
f = open('encoded_test_file.csv','r')

features = f.readline().strip().split(',')

# 抽取one-hot特征
for feat in features:
    if feat not in combined_feats and feat not in common_feats and feat not in remain_feats:
        one_hot_feats.append(feat)
print('one hot 特征收集完毕...')
# print(one_hot_feats)

dict = {}
num = 0
for line in f:
    datas = line.strip().split(',')
    for i,d in enumerate(datas):
        if  features[i] not in dict:
            dict[features[i]] = []     #dict['LBS'] = []
        dict[features[i]].append(d)
    num += 1

f.close()
print('dict has done')
print ("transforming data")

output = open('encoded_test_file.ffm','w')

for i in range(num):
    print(i)
    feats = []
    for j, f in enumerate(one_hot_feats,1):#从1开始编号遍历，返回index value
        field = j
        feats.append((field, f+'_'+dict[f][i]))

    for j, f in enumerate(common_feats,1):
        field = j + len(one_hot_feats)
        feats.append((field, f+'_'+dict[f][i]))

    for j, f in enumerate(combined_feats,1):
        field = j + len(one_hot_feats) + len(common_feats)
        xs = dict[f][i].split(' ')
        for x in xs:
            feats.append((field, f+'_'+x))

    feats = gen_hashed_fm_feats(feats)
 
    # output.write(dict['label'][i] + ' ' + ' '.join(feats) + '\n')
    output.write((' '.join(feats) + '\n'))  # 测试集不包含label列
print('overahh')

output.close()


