
from collections import Counter
import operator
import numpy as np

trainingData = []
validationData = []
testingData = []

def loadData():
	f = open('hw2train.txt', 'r')
	trainingData = [int(line.split())for line in f]
	f = open('hw2validate.txt', 'r')
	validationData = [int(line.split()) for line in f]
	f = open('hw2test.txt', 'r')
	testingData = [int(line.split()) for line in f]

def getNeighbors(trainingData, testInput, k):
	distances = []
	length = len(testInput)-1
	for x in range(len(trainingData)):
		dist = getDistance(trainingData[x], testInput, length)
		distances.append((trainingData[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getDistance(trainingData, testInput, length):
	distance = 0
	a = np.array(trainingData[:-1])
	b = np.array(testInput[:-1])
	distance = np.linalg.norm(a-b)
	return distance

def getVote(neighbors):
	votes = {}
	for x in range(len(neighbors)):
		result = neighbors[x][-1]
		if result in votes:
			votes[result] += 1
		else:
			votes[result] = 1
	sortedVotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuary(testSet, predicts, conMatrix):
	accuarcy = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predicts[x]:
			accuarcy += 1
		conMatrix[predicts[x]][testSet[x][-1]] +=1
	return (accuarcy/float(len(testSet)))*100

def main():
	f = open('hw2train.txt', 'r')
	trainingData = [map(int, line.split()) for line in f]
	f = open('hw2validate.txt', 'r')
	validationData = [map(int, line.split()) for line in f]
	f = open('hw2test.txt', 'r')
	testingData = [map(int, line.split()) for line in f]

	print testingData

	k = 3
	testSet = testingData
	predicts = []

	for i in range(len(testSet)):
		neighbors = getNeighbors(trainingData, testSet[i], k)
		label = getVote(neighbors)
		predicts.append(label)


	f = open('hw2test.txt', 'r')
	labels = [line.split()[-1] for line in f]


	countLabel = sorted(Counter(labels).most_common())
	matrix = [[0 for i in range(10)] for j in range(10)]

	accuarcy = getAccuary(testSet, predicts, matrix)

	print '   ',
	for i in range(len(matrix[i])):
		print '{:4}'.format(i),
	print ' '
	for i in range(len(matrix)):
		print (str(i)+"|"),
		print ' ',
		for j in range(len(matrix[i])):
			matrix[i][j] = float((matrix[i][j]) / float((countLabel[j])[1]))
			print '{:4.2f}'.format((matrix[i][j])),
		print ' '

	print("accuarcy: "+repr(accuarcy) +"%")

main()
