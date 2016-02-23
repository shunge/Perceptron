
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
	trainingDataA = [map(int, line.split()) for line in f]
	f = open('hw4atest.txt', 'r')
	testingDataA = [map(int, line.split()) for line in f]
	f = open('hw4btest.txt', 'r')
	testingDataB = [map(int, line.split()) for line in f]

def perceptron(dataset,w):
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

def votedPecptron(dataset, w):
	c = 1
	setCW = []
	for vector in dataset:
		label = vector[-1]
		if(label == 0 ):
			label = -1
		else:
			label = 1
		a = np.array(vector[:-1])
		if( np.multiply(label, np.dot(a,w)) <= 0):
			setCW.append([w,c])
			w= np.add(w, np.multiply(label, a))
			c = 1
		else:
			c = c+1
	return  setCW

def VotedClassifer(VPec, testingData):
	sign = 0
	result = []
	for testData in testingData:
		for VC in VPec:
			#print VC[1]
			if( np.dot(VC[0], testData[:-1] ) >= 0):
				sign += VC[1]*1
			else:
				sign += VC[1]*-1
		if(sign >= 0 ): result.append([testData[:-1], 6])
		else: result.append([testData[:-1], 0])
	return result

def AvgClassifer(APec, testingData):
	result = []
	WSum = [0] * 784
	for VC in APec:
		WSum += np.multiply(VC[:-1],VC[-1])
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

# def getAccuary(testSet, predicts, conMatrix):
# 	accuarcy = 0
# 	for x in range(len(testSet)):
# 		if testSet[x][-1] == predicts[x]:
# 			accuarcy += 1
# 		conMatrix[predicts[x]][testSet[x][-1]] +=1
# 	return (accuarcy/float(len(testSet)))*100

def main():

	loadData()

	#w = [0] * 784
	w = np.array([0] * 784)
	w = perceptron(trainingDataA, w)

	setCW = votedPecptron(trainingDataA, w)
	#print setCW

	VotedResult = VotedClassifer(setCW, testingDataA)
	print VotedResult
	AvgResult = AvgClassifer(setCW, testingDataA)
	print AvgResult

	# k = 3
	# testSet = testingData
	# predicts = []
    #
	# for i in range(len(testSet)):
	# 	neighbors = getNeighbors(trainingData, testSet[i], k)
	# 	label = getVote(neighbors)
	# 	predicts.append(label)
    #
    #
	# f = open('hw2test.txt', 'r')
	# labels = [line.split()[-1] for line in f]
    #
    #
	# countLabel = sorted(Counter(labels).most_common())
	# matrix = [[0 for i in range(10)] for j in range(10)]
    #
	# accuarcy = getAccuary(testSet, predicts, matrix)
    #
	# print '   ',
	# for i in range(len(matrix[i])):
	# 	print '{:4}'.format(i),
	# print ' '
	# for i in range(len(matrix)):
	# 	print (str(i)+"|"),
	# 	print ' ',
	# 	for j in range(len(matrix[i])):
	# 		matrix[i][j] = float((matrix[i][j]) / float((countLabel[j])[1]))
	# 		print '{:4.2f}'.format((matrix[i][j])),
	# 	print ' '
    #
	# print("accuarcy: "+repr(accuarcy) +"%")

main()
