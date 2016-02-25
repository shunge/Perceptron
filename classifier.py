
from collections import Counter
import operator
import numpy as np

trainingDataA = []
testingDataA = []
trainingDataB = []
testingDataB = []

def loadData():
	global testingDataA, testingDataB, trainingDataA, trainingDataB
	f = open('hw4atrain.txt', 'r')
	trainingDataA = [map(int, line.split()) for line in f]
	f = open('hw4btrain.txt', 'r')
	trainingDataB = [map(int, line.split()) for line in f]
	f = open('hw4atest.txt', 'r')
	testingDataA = [map(int, line.split()) for line in f]
	f = open('hw4btest.txt', 'r')
	testingDataB = [map(int, line.split()) for line in f]

def perceptron(dataset,w, num):
	for i in range(num):
		for vector in dataset:
			label = vector[-1]
			if(label == 0 ):
				label = -1
			else:
				label = 1
			a = np.array(vector[:-1])
			if( np.multiply(label, np.dot(a,w)) <= 0):
				w= np.add(w, np.multiply(label, a))
	return w

def votedPecptron(dataset, w, num):
	c = 1
	setCW = []
	count = 0

	for i in range(num):
		for vector in dataset:
			label = vector[-1]
			if(label == 0 ):
				label = -1
			else:
				label = 1
			a = np.array(vector[:-1])
			if( np.multiply(label, np.dot(w,a)) <= 0):
				setCW.append([w,c])
				w= np.add(w, np.multiply(label, a))
				c = 1
			else:
				c = c+1
			count += 1
	setCW.append([w,c-1])
	#count += c

	return  setCW

def VotedClassifer(VPec, testingData):
	#sign = 0
	result = []
	for testData in testingData:
		sign = 0
		for VC in VPec:
			if( np.dot(VC[0], testData[:-1] ) >= 0):
				sign += VC[1]
			else:
				sign += -VC[1]
		if(sign >= 0 ): result.append([testData[:-1], 6])
		else: result.append([testData[:-1], 0])
	return result

def AvgClassifer(APec, testingData):
	result = []
	WSum = [0] * 784

	for VC in APec:
		# VC[0] is w (featurs), VC[1] is y (labels)
		WSum += np.multiply(VC[0],VC[1])


	for testData in testingData:
		if( np.dot(WSum, testData[:-1] ) >= 0):
			result.append([testData[:-1], 6])
		else:
			result.append([testData[:-1], 0])

	return result

def PecClassfier(w, testingData):
	result = []
	for testData in testingData:
		if( np.dot(w, testData[:-1] ) >= 0):
			result.append([testData[:-1], 6])
		else:
			result.append([testData[:-1], 0])
	return result

def OVAPecClassfier(w, testingData):
	if (np.dot(w, testingData[:-1]) >= 0):
		return 1
	else:
		return 0

def OVAperceptron(dataset, w, l):
	for vector in dataset:
  		label = vector[-1]
 		if(label == l ):
  			label = 1
 		else:
 			label = -1
  		a = np.array(vector[:-1])
  		if( np.multiply(label, np.dot(a,w)) <= 0):
  			w= np.add(w, np.multiply(label, a))
  	return w

def onevsall():
	loadData()
	w = []
	for i in range(10):
		w.append(OVAperceptron(trainingDataB, np.array([0] * 784), i))
	matrix = [[0 for i in range(10)] for j in range(11)]

	n = [0 for i in range(10)]
	for i in range(len(testingDataB)):
		n[testingDataB[i][-1]] += 1

	for data in testingDataB:
		know = False 
		for i in range(10):
			result = 0
			mark = -1
			for j in range(10):
				if(OVAPecClassfier(w[j], data) == 1):
					mark = j
				result += OVAPecClassfier(w[j], data) 
			if(result == 1 and mark == i):
				matrix[i][data[-1]] += 1
				know = True
				break;
		if(know == False):
			matrix[10][data[-1]] += 1

	for i in range(11):
		for j in range(10):
			matrix[i][j] = float(matrix[i][j]) / float(n[j])
	
	print '   ',
	for i in range(len(matrix[i])):
		print '{:4}'.format(i),
	print ' '
	for i in range(len(matrix)):
		print (str(i)+"|"),
		print ' ',
		for j in range(len(matrix[i])):
			print '{:4.2f}'.format((matrix[i][j])),
		print ' '


def ErrorTester(result, testData):
	errorNum = 0
	SumNum = 0
	for i in range(len(result)):
		if(result[i][1] != testData[i][-1]):
			errorNum += 1
		SumNum += 1
	return float(errorNum)/float(SumNum)

def main():
	loadData()
	w = np.array([0] * 784)
	w = perceptron(trainingDataA, w, 3)

	newW =  np.array([0] * 784)
	setCW = votedPecptron(trainingDataA, newW, 3)

	VotedResult = VotedClassifer(setCW, testingDataA)

	AvgResult = AvgClassifer(setCW, testingDataA)

	PecResult = PecClassfier(w, testingDataA)

	print "Perceptron", ErrorTester(PecResult,testingDataA)
	print "Voted", ErrorTester(VotedResult,testingDataA)
	print "Avg", ErrorTester(AvgResult,testingDataA)

main()
#onevsall()
