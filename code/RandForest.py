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
	
	#Prepare Feature Condition
	featurecondit = []
	for j in range(F):
		tmp = []
		for i in range(N):
			if (len(tmp) == 0 or X[i][j] not in tmp):
				tmp.append(X[i][j])
		featurecondit.append(tmp)

	logger.debug("Feature Condition")
	logger.debug(featurecondit)
	#Prepare Feature Convert dict
	#Convert abstract condition to real number
	#skip some condition which is already real number
	skip_condition = [0,2,3,4,5] #0:pclass, 2:AGE, #4:Passenger fare
	skip_F = list(range(F)) #condiction list afer del members which is already real number
	offset = 0
	for s in skip_condition:
		s -= offset
		offset += 1
		del(skip_F[s])
	
	feature_convrt_dict = []
	for i in skip_F:
		f_num = len(featurecondit[i])
		tmp_dict = {}
		for f in range(f_num):
			key = featurecondit[i][f]
			tmp_dict.update({key:f})
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
			#print ith_feature			
			if (ith_feature not in skip_F): #already real number, just put in list
				tmp_value = X[ith_data][ith_feature]
				try:
					tmp_value = float(tmp_value)
					pass
				except Exception, e:
					tmp_value = -1

				tmp_list.append(tmp_value)
				#tmp_list.append( float(X[ith_data][ith_feature]) )
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
			#print ith_feature			
			if (ith_feature not in skip_F): #already real number, just put in list
				tmp_value = T[ith_data][ith_feature]
				try:
					tmp_value = float(tmp_value)
					pass
				except Exception, e:
					tmp_value = -1

				tmp_list_t.append(tmp_value)
				#tmp_list_t.append(T[ith_data][ith_feature])
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
	X = np.array(X)

	logger.debug("Convert X to numpy array")
	for i in range(5):
		logger.debug(X_rebuild[i])
	model.fit(X_rebuild,labels)

	logger.info("Step 2 finish")

	##
	logger.info("Step 2: Training")

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