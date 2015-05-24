import numpy as np
import logging
import random
from random import randint
import CSVRW

from sklearn.ensemble import RandomForestClassifier

#logging setting
log_file = "./RandForest.log"
log_level = logging.DEBUG

logger = logging.getLogger("RandForest")
handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter("[%(levelname)s][%(funcName)s]\
[%(asctime)s]%(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)

#Parameters 
forestsize = 100    # how many trees in a forest

if __name__ == "__main__":
	##
	logger.info("Step 1: Read data")
	labels = []    # 0:dead, 1:survive
	X = []        # Taining data
	T = []        # Testing data
	CSVRW.CSV_read(labels, X, T)    # Read data from csv files
	#in csv read i delete some feature because i think that is useless(like name of passengers)
	#in future i want the useless information can be deleted automatically
	logger.debug("labels[0]~[4]")
	for i in range(5):
		logger.debug(labels[i])
	logger.debug("X[0]~[4]")
	for i in range(5):
		logger.debug(X[i])
	logger.debug("T[0]~[4]")
	for i in range(5):
		logger.debug(T[i])

	#Kown Constants
	F = len(X[0])		# F -> number of features; cols
	N = len(X)			# N -> number of data; rows
	NT = len(T)			# NT -> number of test data; rows

	#Find non numeric features
	non_numeric_F = []
	test_F = X[1] #each element in X[1] has value, use it to find who should be skiped
	for ith_feature in range(F):
		try:
			tmp = float(test_F[ith_feature])# convert to float OK, skip this feature
			#skip_condition.append(ith_feature)
		except Exception, e:
			non_numeric_F.append(ith_feature)
			pass #convert fail, need converstion dict
	logger.debug("non numeric feature")
	logger.debug(non_numeric_F)

	#Prepare Feature Condition
	featurecondit = []
	for ith_feature in non_numeric_F:
		tmp = []
		for ith_data in range(N):
			if (len(tmp) == 0 or X[ith_data][ith_feature] not in tmp):
				tmp.append(X[ith_data][ith_feature])
		featurecondit.append(tmp)

	logger.debug("Feature Condition")
	logger.debug(featurecondit)

	#Prepare Feature Convert dict
	#Convert non numeric condition to real number
	feature_convrt_dict = []
	for ith_feature in range(len(featurecondit)):
		total_condit_num = len(featurecondit[ith_feature])
		tmp_dict = {}
		for jth_condit in range(total_condit_num):
			key = featurecondit[ith_feature][jth_condit]
			tmp_dict.update({key:jth_condit})
		feature_convrt_dict.append(tmp_dict)

	logger.debug("Feature Convert table")
	logger.debug(feature_convrt_dict)
	
	#Use convert dict rebuild training and testing data
	X_rebuild = []
	T_rebuild = []	

	for ith_data in range(N):
		tmp_list = []
		offset = 0
		for ith_feature in range(F):
			if (ith_feature not in non_numeric_F): #already real number, just put in list
				tmp_value = X[ith_data][ith_feature]
				try:
					tmp_value = float(tmp_value)
					pass
				except Exception, e:
					tmp_value = -1 #some blank in numeric feature

				tmp_list.append(tmp_value)				
				offset += 1
			else: #need conversion
				cur_condit = X[ith_data][ith_feature]
				tmp_list.append(feature_convrt_dict[ith_feature - offset][cur_condit])
		X_rebuild.append(tmp_list)

	logger.debug("Rebuild X[0]~X[4]")
	for i in range(5):
		logger.debug(X_rebuild[i])

	for ith_data in range(NT):
		tmp_list_t = []
		offset = 0
		for ith_feature in range(F):			
			if (ith_feature not in non_numeric_F): #already real number, just put in list
				tmp_value = T[ith_data][ith_feature]
				try:
					tmp_value = float(tmp_value)
					pass
				except Exception, e:
					tmp_value = -1

				tmp_list_t.append(tmp_value)				
				offset += 1
			else: #need conversion
				cur_condit = T[ith_data][ith_feature]
				tmp_list_t.append(feature_convrt_dict[ith_feature - offset][cur_condit])
		T_rebuild.append(tmp_list_t)

	logger.debug("Rebuild T[0]~T[4]")
	for i in range(5):
		logger.debug(T_rebuild[i])
	
	logger.info("Step 1 finish")

	##
	logger.info("Step 2: Training")

	model = RandomForestClassifier(n_estimators  = forestsize)
	#train
	model.fit(X_rebuild,labels)

	logger.info("Step 2 finish")

	##
	logger.info("Step 3: Predict")

	pred_y = model.predict(T_rebuild)
	logger.debug("Predict value:")
	for i in range(10):
		logger.debug(pred_y[i])

	logger.info("Step 3 finish")

	##
	logger.info("Step 4: Output to csv")

	CSVRW.CSV_write("Titanic_RandForest", NT, pred_y)

	logger.info("Step 4 finish")
	logger.info("RandForest finish")