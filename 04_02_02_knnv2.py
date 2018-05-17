import csv
import random
import math
import operator
from time import time


def loadDataset(filename, divFactor, trainData=[], testData=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range(len(dataset) - 1):
            for j in range(len(dataset[0])):
                dataset[i][j] = float(dataset[i][j])
            if random.random() < divFactor:
                trainData.append(dataset[i])
            else:
                testData.append(dataset[i])


def Distance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)


def kNearestNeighbors(trainData, test, k):
    distances = []
    length = len(test) - 1
    for i in range(len(trainData)):
        currentDistance = Distance(test, trainData[i], length)
        distances.append((trainData[i], currentDistance))
    distances.sort(key=operator.itemgetter(1))
    kNeighbors = []
    for i in range(k):
        kNeighbors.append(distances[i][0])
    return kNeighbors


def Classify(neighbors):
    classVotesDict = {}
    for i in range(len(neighbors)):
        someClass = neighbors[i][-1]
        if someClass in classVotesDict:
            classVotesDict[someClass] += 1
        else:
            classVotesDict[someClass] = 1
    sortedCVDict = sorted(classVotesDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedCVDict[0][0]


def Accuracy(testData, predictions):
    hits = 0
    for i in range(len(testData)):
        if testData[i][-1] == predictions[i]:
            hits += 1
    return (hits / float(len(testData))) * 100.0


def feature_normalize(X):
    mean = 0.0
    std = 0.0
    total = 0.0
    columnlist = []

    for col in range(0, len(X[0]) - 1):
        for row in range(0, len(X)):
            total += X[row][col]
            columnlist.append(X[row][col])

        mean = total / len(X)
        # std = 0.0000000000001 + max(columnlist) - min(columnlist)
        std = 0.001 + max(columnlist) - min(columnlist)

        for row in range(0, len(X)):
            X[row][col] = (X[row][col] - mean) / std

        columnlist[:] = []
        mean = 0.0
        std = 0.0
        total = 0.0


def feature_normalize2(X):
    std = 0.0
    columnlist = []

    for col in range(0, len(X[0]) - 1):
        for row in range(0, len(X)):
            columnlist.append(X[row][col])

        # std = 0.0000000000001 + max(columnlist) - min(columnlist)
        std = 0.0000001 + max(columnlist) - min(columnlist)

        for row in range(0, len(X)):
            X[row][col] = (X[row][col] - min(columnlist)) / std

        columnlist[:] = []
        std = 0.0


trainData = []
testData = []
K = range(1, 27, 2)
divFactor = 0.8
averageAccuracy = 0
numberOfTests = 100

accuracies = []
averageAccuracies = []
predictions = []
times = []

KsForPlot = []
AccuraciesForPlot = []

for x in range(len(K)):
    for n in range(100):
        loadDataset('./csv/knn.csv', divFactor, trainData, testData)
        # feature_normalize2(trainData)
        # feature_normalize2(testData)

        t0 = time()
        for i in range(len(testData)):
            kNeighbors = kNearestNeighbors(trainData, testData[i], K[x])
            predict = Classify(kNeighbors)
            predictions.append(predict)

        times.append(round(time() - t0, 3))
        accuracy = Accuracy(testData, predictions)
        accuracies.append(accuracy)

        KsForPlot.append(K[x])
        AccuraciesForPlot.append(accuracy)

        del trainData[:]
        del testData[:]
        del predictions[:]

    averageTime = sum(times) / float(len(times))
    averageAccuracy = sum(accuracies) / float(len(accuracies))

    print('Average Time for K=' + repr(K[x]) + " after " + repr(
        numberOfTests) + " tests with different data split: " + repr(round(averageTime, 6)) + ' sec')
    print('Average accuracy for K=' + repr(K[x]) + " after " + repr(
        numberOfTests) + " tests with different data split: " + repr(averageAccuracy) + ' %')
    print("")
    # print 'Accuracies for each of the ', repr(numberOfTests), " tests with K =",K[x], ":", accuracies
    # print("#####################################################################################################################")
    del accuracies[:]
    del times[:]

print("")
print('Checking predictions with K = 3:')
K = 3
loadDataset('./csv/knn.csv', divFactor, trainData, testData)
# feature_normalize2(trainData)
# feature_normalize2(testData)

for i in range(len(testData)):
    kNeighbors = kNearestNeighbors(trainData, testData[i], K)
    predict = Classify(kNeighbors)
    predictions.append(predict)
    print('Predicted class: ' + repr(predict) + ', Real class: ' + repr(testData[i][-1]))

accuracy = Accuracy(testData, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

############################################################################################################
# Visualization part:
############################################################################################################
import matplotlib.pyplot as plt
import numpy as np

left, width = .35, 0.5
bottom, height = .27, .7
right = left + width
top = bottom + height

Ks = np.array(KsForPlot)
Accs = np.array(AccuraciesForPlot)

# Data and prediction line
fig, ax = plt.subplots(figsize=(15, 10))
# ax.plot(Ks, Accs, 'b', label='Prediction')
ax.scatter(Ks, Accs, label='Traning Data', color='r')
# ax.scatter(Ks, Accs, color='r')

ax.legend(loc=3)
ax.set_xlabel('K (from 1 to 25)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracies against K used. For each K 100 accuracies plotted')
# ax.text(right, top, 'Accuracies against K used. For each K 100 accuracies plotted', size=18,horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
plt.show()

