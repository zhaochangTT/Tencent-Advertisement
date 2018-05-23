# encoding=utf-8
'''
	合并文件，用户特征和广告特征按照uid,aid拼接
'''
import pandas as pd
import time


start = time.time()
print('读取文件...')
train = pd.read_csv('preliminary_contest_data/train.csv')
adfeats = pd.read_csv('preliminary_contest_data/adFeature.csv')
userfeats = pd.read_csv('preliminary_contest_data/userFeature.csv')
test = pd.read_csv('preliminary_contest_data/test2.csv')
print('文件读取完成...')

train.loc[train['label']==-1,'label'] = 0	# label为-1改为0
test['label'] = -1	# 默认测试集label为-1

print('产生中间文件...')
train.to_csv('train_mid.csv')
test.to_csv('test_mid.csv')

print('拼接训练集和测试集...')
data = pd.concat([train, test])	# 上下拼接

print('拼接特征...')
data = pd.merge(data, adfeats, on='aid', how='left')
data = pd.merge(data, userfeats, on='uid', how='left')
data = data.fillna('-1')	# 缺失值用-1表示
print('拼接特征完成...')

combined_train = data[data.label != -1]	# 训练集
combined_test  = data[data.label == -1]	# 测试集

print('写入文件...')
combined_train.to_csv('combined_train.csv', index=False)
combined_test.to_csv('combined_test.csv', index=False)

end = time.time()
print('总耗时： ', end - start)

