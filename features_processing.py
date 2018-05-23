# encoding=utf-8

import pandas as pd
from sklearn import preprocessing

# 2.首先处理类别变量，特征每一个取值为一新特征
def processCatagries(train_set, encoded_file, feats):
	user = train_set['uid']  # 获取用户ID列，后面用作拼接的轴
	data = train_set  
	# print('共要处理特征数：' , feats.length)
	print('准备处理...')
	for f in feats:  # 依次处理每个特征
		print('开始处理特征 ', f)
		df = pd.get_dummies(data[f], prefix=f) # 列名前缀是原特征名

		print('开始拼接...')
		data = pd.concat([data, df], axis=1) # 拼接，结果赋值给data
		print('拼接完成...')
		data.drop(f,axis=1, inplace=True)  # 删除原来的列
		print('完成处理特征 ', f)

	print('开始写入文件...')
	data.to_csv(encoded_file)
	print('文件写入完成...')


if __name__ == '__main__':

	# 1.划分特征为多种类别，包括类别特征、多值特征、一般特征
	one_hot_feats = ['carrier','consumptionAbility','education','gender','house','adCategoryId', 'productId', 'productType']
	combined_feats = ['kw1','kw2','kw3','topic1','topic2','topic3', 'interest1','interest2','interest3',
	                  'interest4','interest5','ct', 'marriageStatus', 'os']
	common_feature = ['creativeId','campaignId','LBS','advertiserId','appIdInstall', 'appIdAction', 'aid', 'advertiserId', 'campaignId', 'creativeId', 'age']
	# train_set['creativeId'].value_counts() # 选择出现次数较少的作为类别特征

	# 训练集
	print('读取训练集...')
	train_set = pd.read_csv('combined_train.csv')
	print('训练集读取完毕...')
	print('开始处理训练集...')
	processCatagries(train_set, 'encoded_train_file.csv', one_hot_feats)
	
	# 测试集
	print('读取测试集...')
	test_set = pd.read_csv('combined_test.csv')
	print('测试集读取完毕...')
	print('开始处理测试集...')
	processCatagries(test_set, 'encoded_test_file.csv',one_hot_feats)









