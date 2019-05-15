import csv
import random
import math
import operator

''' some of the code is taken from http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/ by Jason Brownlee
'''

# a function to open a csv file and load the data
def loadDataset(filename, split, trainingSet=[], testSet=[]):
	n_features = 4 # there are 4 features in the dataset
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(n_features): 
                    #print (x,y)
                    #print (float(dataset[x][y]))
	            dataset[x][y] = float(dataset[x][y])
		# split the dataset randomly into train and test datasets
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
 
# calculate Euclidean distance between two instances
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
# find k nearest neighbours to a test instance from a training set
def getNeighbours(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbours = []
	# find k nearest neighbours
	for x in range(k):
		neighbours.append(distances[x][0])
	return neighbours
 
 # a function for getting the majority voted response from a number of neighbours. The class (response) is the last attribute ([-1]) for each neighbour.
def getResponse(neighbours):
	classVotes = {}
	for x in range(len(neighbours)):
		response = neighbours[x][-1] # the class of the instance
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
# calculate accuracy of a prediction
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# main function where the prediction is done 
def main():
	# prepare data
	trainingSet = []
	testSet = []
	split = 0.67 # the ratio between train and test datasets
	loadDataset('iris.data', split, trainingSet, testSet)
	print ('Train set: ' + repr(len(trainingSet)))
	print ('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 4 # the number of neighbours
	for x in range(len(testSet)):
		neighbours = getNeighbours(trainingSet, testSet[x], k)
		result = getResponse(neighbours)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(round(accuracy, 3)) + '%')


############## executing the main program ################
main()
