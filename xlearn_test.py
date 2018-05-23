import xlearn as xl
import time
import pandas as pd

def ffm_train():
	ffm_model = xl.create_ffm()
	ffm_model.setOnDisk()
	ffm_model.setTrain('encoded_train_file.ffm')
	# ffm_model.setValidate('test.ffm')

	params = {
	    'epoch'         : 15,
	    'metric'        : 'auc',
	    'task'          : 'binary',
	    'k'             : 4,
	    'lr'            : 0.2,
	    'lambda'        : 0.00002,
	    # 'stop_window'   : 3,
	    # 'fold'          : 10,
	}
	# ffm_model.cv(param)
	start_Time = time.time()
	
	ffm_model.fit(params, 'model/model2.out')
	# ffm_model.cv(params)	# 交叉验证
	
	end_Time = time.time()
	print('总耗时: ', (end_Time - start_Time))

def ffm_predict():
	ffm_model = xl.create_ffm()
	ffm_model.setOnDisk()
	ffm_model.setTest('encoded_test_file.ffm')
	ffm_model.setSigmoid()

	ffm_model.predict('model/model2.out', 'output/output2.res')


# 结果文件拼接到测试文件
def tranferToSubmission():
	test = pd.read_csv('preliminary_contest_data/test2.csv')
	res = []
	with open('output/output2.res', 'r') as f:
		for line in f.readlines():
			res.append(float(line))
	test['score'] = res
	test.to_csv('submission.csv', index=False)

if __name__ == '__main__':
	ffm_train()
	ffm_predict()
	tranferToSubmission()
