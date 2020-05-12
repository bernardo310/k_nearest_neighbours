""" 
    This script containg utility functions used for k nearest neighbours algorithm
    Author : Bernardo Cardenas Domene
    Institution : Universidad de Monterrey
    First Created : 6/May/2020
    Email : bernardo.cardenas@udem.edu
"""

import numpy as np
import math
import sys

def load_data(file, includes_y, split:100, flag):
    """
        Reads data from csv file and returns numpy type array for x data and y data
        inputs: 
            file: string of csv file
            includes_y: if set to 1, csv file must include y column at the end. If set to 0, csv file only contains x data columns
            split: indicates percentage to be used as training data, residual percentage is used for testing data, for example split=80 means 80% of the data will be used for training and 20% for testing
            flag: if flag = 1, the training data is printed. if flag = 0, data is not printed
        output: numpy type arrays x_data and y_data representing the data found in file

    """
    #read data from file
    try:
        headers = np.genfromtxt(file, delimiter=',', max_rows=1)
        data = np.genfromtxt(file, delimiter=',', skip_header=1)
    except:
        print('File not found:', file)
        sys.exit()
    
    #shuffle data order
    np.random.shuffle(data)

    if(includes_y):
        #separate data into x and y. (y is assumed to be last column in csv file)
        number_x_columns = len(headers) - 1
        #get splitting point to divide data 
        splitting_point = math.floor(len(data) * (split / 100))
        #get training data from data
        x_training_data = data[:splitting_point, :number_x_columns]
        y_training_data = data[:splitting_point, number_x_columns:]
        #get testing data from data
        x_testing_data = data[splitting_point:, :number_x_columns]
        y_testing_data = data[splitting_point:, number_x_columns:]
        #print data if flag is set
        if(flag):
            print('-'*100)
            print('Training Data and Y outputs')
            print('-'*100)
            for i in range(len(data)):
                print(x_training_data[i], y_training_data[i])
            print('\n')
        #return x and y training and testing data
        return x_training_data, y_training_data, x_testing_data, y_testing_data
    else: #doesnt inclyde y column (used for reading testing data)
        return data



def scale_data(data, flag, mean=0, std=0):
    """
        Returns feature scaling applied to data so range in values are -1 >= x <= 1
        input:
            data: numpy type array containing data to be scaled
            flag: if flag = 1, the scaled training data is printed. if flag = 0, data is not printed
            mean: (optional) mean array from training data
            std: (optional) std array from training data
        output: 
            scaled_data: numpy type array containing scaled data
            mean: mean from data
            std: standard deviation from data
    """
    scaled_data = []
    n_columns = len(data[0])
    if(not isinstance(mean, np.ndarray) and not isinstance(std, np.ndarray)):
        mean = []
        std = []
        #calculate mean and standerd deviation
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        #scale data 
        for i in range(len(data)):
            temp = []
            for j in range(0, n_columns):
                temp.append((data[i][j] - mean[j]) / std[j])
            scaled_data.append(temp)

        #convert to numpy array
        scaled_data = np.array(scaled_data)

        #print data if flag is set
        if(flag):
            print('-'*100)
            print('Scaled Training Data and Y outputs')
            print('-'*100)
            for i in range(len(scaled_data)): print(scaled_data[i])
            print('\n')
        return scaled_data, mean, std
    else:
        #scale testing data 
        for i in range(len(data)):
            temp = []
            for j in range(0, n_columns):
                temp.append((data[i][j] - mean[j]) / std[j])#2
            scaled_data.append(temp)

        #convert to numpy array
        scaled_data = np.array(scaled_data)
        return scaled_data


def compute_euclidean_distance(testing_point, x):
    """
    calculates the ecludian distance between a testing point and x data
    input parameter: 
        testing_point: numpy type array containing the testing point location. ex [0,0,0]
        x: numpy type array containing x data
    output: 
        distance: numpy type array containing disance from x data and testing point per row of x data
    """
    #calculate squared difference
    squared_difference = (x - testing_point) ** 2
    #create empty array to store distances
    distances = np.zeros((len(squared_difference),1))
    #calculate square root of sum of squared difference and store in distances array
    for i in range(len(squared_difference)):
        dist = math.sqrt(np.sum(squared_difference[i]))
        distances[i] = dist
    #return distances
    return distances

def compute_conditional_probabilities(k, testing_point, x_train, y_train):
    """
    calculates the contidional probability. In other words, whether a testing point belongs to positive class or negative class by calculating the distance between the testing point and all x data, and selecting the nearest k elements
    input parameter: 
        k: int representing k value for number of nearest neighbours to find
        testing_point: numpy type array containing the testing point location. ex [0,0,0]
        x_train: numpy type array containing x training data (95% of data)
        y_train: mumpy type array containing y training data for x training data
    output: 
        probability: outcome of probability, 1 if belongs to positive class, 0 if belongs to negative class.
    """    
    #get distances of testing point to x training data
    distances = compute_euclidean_distance(testing_point, x_train)
    #attach label to distances (y values)
    distances_with_label = (np.hstack([distances, y_train]))
    #sort ascending based on distances
    sorted_distances_with_label = distances_with_label[distances_with_label[:,0].argsort()]
    #create zero list for storing total number of positive and negative classes found in nearest k elements
    total_classes = np.array([0,0]) #total_classes[0] = true class count,   total_classes[1] = false class count
    for i in range(k):   #get first k elements and count classes
        if(sorted_distances_with_label[i][1] == 1): total_classes[0] += 1
        elif(sorted_distances_with_label[i][1] ==0): total_classes[1] +=1
    #calculate conditional probabilities
    p_true = 1 / len(x_train) * total_classes[0]
    p_false = 1 / len(x_train) * total_classes[1]
    #determine which is majority
    probability = 0
    if(p_true > p_false): probability = 1
    #return probability
    return  probability
 

def predict(k, x_train, y_train, x_testing):
    """
    predicts y values for every x testing using KNN
    input parameter: 
        x_train: numpy type array containing x training data (95% of data)
        y_train: mumpy type array containing y training data for x training data
        x_testing: numpy type array containing x testing data (5% of data)
    output: 
        predictions: numpy type array containing y predictions (values are 1 or 0)
    """ 
    #create zeros list to store predictions
    predictions = np.zeros((len(x_testing),1))
    #iterate every x testing element
    for i in range(len(x_testing)):
        #calculate conditional probabilities
        p = compute_conditional_probabilities(k, x_testing[i], x_train, y_train)
        #save prediction to list
        predictions[i] = p        
    #return list of predictions
    return predictions


def get_confusion_matrix(predictions, y):
    """
    obtains confusion matrix values using predictions and y testing data
    input parameter: 
        predictions: numpy type array containing predictions
        y: numpy type array containing y testing data
    output: 
        matrix: list containing confusion matrix values
    """
    matrix = [0,0,0,0]
    #iterate predictions and compare with y data
    for i in range(len(predictions)):
        if(predictions[i] == 1 and y[i] == 1): #true positive
            matrix[0] += 1
        elif(predictions[i] == 0 and y[i] == 0): #true negative
            matrix[1] += 1
        elif(predictions[i] == 1 and y[i] == 0): #false positive
            matrix[2] += 1
        elif(predictions[i] == 0 and y[i] == 1): #false negative
            matrix[3] += 1
    return matrix


def print_performance_metrics(m):
    """
    prints confusion matrix and performance metrics using confusion matrix
    input parameter: 
        m: list containing confusion matrix values
    """
    #calculate metrics
    accuracy = (m[0] + m[1]) / (m[0] + m[1] + m[2] + m[3])
    precision = m[0] / (m[0] + m[2])
    recall = m[0] / (m[0] + m[3]) 
    specificity = m[1] / (m[1] + m[2])
    f1 = 2 * ((precision * recall) / (precision + recall))
    #print matrix
    print('-'*120)
    print('Confusion matrix')
    print('-'*120)

    print('|','{0: >36}'.format(""), "|", '{0: >36}'.format("Actual has diabetes (1)"), "|", '{0: >36}'.format("Actual doesn't have diabetes (0)", "|"))
    print('-'*120)
    print('|','{0: >36}'.format("Predicted has diabetes (1)"), "|", '{0: >36}'.format(m[0]), "|", '{0: >36}'.format(m[2]), "|")
    print('-'*120)
    print('|','{0: >36}'.format("Predicted does not have diabetes (0)"), "|", '{0: >36}'.format(m[3]), "|", '{0: >36}'.format(m[1]), "|")
    print('-'*120, '\n')
    #print metrics
    print('\n', '-'*120)
    print('Performance metrics')
    print('-'*120)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Specificity:', specificity)
    print('F1 Score:', f1)
    return 
