import csv
import logging

#logging setting
log_file = "./CSVRW.log"
log_level = logging.DEBUG

logger = logging.getLogger("CSVRW")
handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter("[%(levelname)s][%(funcName)s]\
[%(asctime)s]%(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)

#logging.disable(logging.CRITICAL)

feature = 10 #total feature number
def CSV_read(Labels, X, T):
	skip = [3,8,10]# filter some VARIABLEs for useless
	with open('../data/train.csv', 'r') as incsv:
	#with open('../data/train0.csv', 'r') as incsv:
		train_data = csv.reader(incsv)    # read training data and lables
		next(train_data)    # skip first row
		for row in (train_data):
			Labels.append(row[1])
			tmp = []
			for i in range(2,feature+2):
				if (i not in skip):
					tmp.append(row[i])
			X.append(tmp)

	with open('../data/test.csv', 'r') as incsv:
	#with open('../data/test0.csv', 'r') as incsv:
		test_data = csv.reader(incsv)    # read testing data
		next(test_data)    # skip first row
		for row in test_data:
			tmp = []
			for i in range(1,feature+1):
				if (i+1 not in skip):
					tmp.append(row[i])
			T.append(tmp)
			#T.append((list)(row[1]) + (list)(row[3:6]) + (list)(row[8:]))


def CSV_write(Fname, Leng, WTdata):
	with open('../data/%s.csv' % Fname, 'w') as outcsv:
		csv_writer = csv.writer(outcsv)
		csv_writer.writerow(["PassengerId", "Survived"])
		for y in range(Leng):
			csv_writer.writerow([y + 892, WTdata[y]])

if __name__ == "__main__":
	Labels = []
	X = []
	T = []
	CSV_read(Labels, X, T)

	PassNumber = 2
	TestData = []
	TestData.append(1)
	TestData.append(0)
	CSV_write("CSVRW_Titanic", PassNumber, TestData)

	logger.debug("in main")
	logger.debug("Labels:")
	logger.debug(Labels[0])
	logger.debug("X:")
	logger.debug(X[0])
	logger.debug("T:")
	logger.debug(T[0])