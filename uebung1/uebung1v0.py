# -*- coding: utf-8 -*-

import sys
import numpy as np
import heapq
from numba import vectorize
import matplotlib.pyplot as plt

@vectorize('float64(float64,float64)', target='cuda')
def euclidean_(x, y):
    return (x-y)**2

@vectorize('float64(float64,float64)', target='cuda')
def sum_(x, y):
    return(x+y)

def euclidean(x, y):
    """
    Help functions that calculates the euclidean distance between two arbitrary vectors of equal dimensions
    
    Throws exception for len(x) != len(y)
    """
    if(len(x) != len(y)):
        raise Exception('Vector-like arguments x and y have to be of the same length')
    sumSquares = 0
    for i in range(len(x)):
        sumSquares += (x[i]-y[i])**2
    return(np.sqrt(sumSquares))

class Classifier:
    trainData = []
        
    def __init__(self, data):
        """Initializes a list of training data"""
        self.trainData = data
        
    def kNN(self, k, data, dist=euclidean):
        """Runs the kNN algorithm for k closest neighbours and returns the 'closest' class/number 
        by calculating the distance of every vector in the training data and voting using the k closest vectors."""
        #kClosest = heapq.nlargest(k, [(x[0], dist(data, x[1])) for x in self.trainData], key = lambda x: x[1])
        kClosest = heapq.nlargest(k, [(x[0], sum(euclidean_(data, x[1]))) for x in self.trainData], key = lambda x: x[1])
        """squareDifs = np.reshape(squareDifs, [self.trainSamples,256])
        sums = np.zeros(self.trainSamples)
        for i in range(256):
            sums = sum_(sums, squareDifs[:,i])"""
                
        #kClosest = heapq.nlargest(k, [(self.trainDataClass[i], sums[i]) for i in range(self.trainSamples)], key = lambda x: x[1])
        counts = [[x,0] for x in range(10)]
        for elem in kClosest:
            counts[elem[0]][1] += 1
        return(max(counts, key = lambda x: x[1])[0])

def readfile(filename):
    """Opens file with given filename and reads in data line after line into a numpy array"""
    data = []
    f = open(filename)
    for line in f:
        ls = line.split(' ')
        digit = int(float(ls[0]))
        matrix = np.array([float(val) for val in ls[1:257]])
        data.append((digit,matrix))
    f.close()
    return(data)
    
def writematrix(matrix, filename):
    """Writes the confusion matrix in tabular form to a file with the specified name"""
    file = open(filename,'w')
    file.write('\n')
    for i in range(10):
        line = str(i) + ' '
        for j in range(9):
            line += ("%.2f" % matrix[i,j]) + '& '
        line += str(matrix[i,9]) + '\n'
        file.write(line)    
    file.close()
    
def visualize(data,i):
    """For a list of numpy arrays with 256 greyscale values, visualizes the greyscale data creating a greyscale plot of the reshaped 16x16 numpy array"""
    plt.gray()
    plt.imshow(data[i][1].reshape([16,16]))


"""
Main script:

Reads greyscala data from zip.train and zip.test and runs k-NN for specified k
"""
train = readfile('zip.train')
test = readfile('zip.test')

while(True):
    try:
        k = int(input(r'Run k-NN of data for k: '))
        if(int(k) < 0 or int(k) > len(test)):
            print('k has to be a natural number greater than 0 and smaller than ' +  str(len(test)) + '.')
        else:
            break
    except ValueError:
        print('k has to be a natural number greater than 0 and smaller than ' +  str(len(test)) + '.')

c = Classifier(train)
confusionMatrix = np.zeros([10,10],dtype='float32')
digitCount = [0 for i in range(10)]
samplesize = len(test)
for i in range(samplesize):
    x = test[i]
    confusionMatrix[x[0], c.kNN(k,x[1])] += 1
    sys.stdout.write('\rProgress: ' + str(i+1) + '/' + str(samplesize) + ' classified.')
print('Confusion matrix:')
print(confusionMatrix)

errorRate = 1- sum(np.diag(confusionMatrix))/sum(confusionMatrix.flatten())
print('Error rate: ' +  str(errorRate))

normConfusionMatrix = np.copy(confusionMatrix)
for i in range(10):
    sumRow = sum(normConfusionMatrix[i,:])
    if(sumRow>0):
        normConfusionMatrix[i,:] = normConfusionMatrix[i,:]/sumRow
print('Normalized confusion matrix:')
print(normConfusionMatrix)

writematrix(confusionMatrix, 'confusion_' + str(k) + '.txt')
writematrix(normConfusionMatrix, 'norm_confusion_' + str(k) + '.txt')