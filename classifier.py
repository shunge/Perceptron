
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

# def getNeighbors(trainingData, testInput, k):
# 	distances = []
# 	length = len(testInput)-1
# 	for x in range(len(trainingData)):
# 		dist = getDistance(trainingData[x], testInput, length)
# 		distances.append((trainingData[x], dist))
# 	distances.sort(key=operator.itemgetter(1))
# 	neighbors = []
# 	for x in range(k):
# 		neighbors.append(distances[x][0])
# 	return neighbors
#
# def getDistance(trainingData, testInput, length):
# 	distance = 0
# 	a = np.array(trainingData[:-1])
# 	b = np.array(testInput[:-1])
# 	distance = np.linalg.norm(a-b)
# 	return distance
#
# def getVote(neighbors):
# 	votes = {}
# 	for x in range(len(neighbors)):
# 		result = neighbors[x][-1]
# 		if result in votes:
# 			votes[result] += 1
# 		else:
# 			votes[result] = 1
# 	sortedVotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
# 	return sortedVotes[0][0]
#
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
	print w

	setCW = []
	setCW = votedPecptron(trainingDataA, w)
	print setCW
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
