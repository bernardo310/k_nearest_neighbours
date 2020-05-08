""" 
    This script tests k nearest neighbours algorithm
    Author : Bernardo Cardenas Domene
    Institution : Universidad de Monterrey
    First Created : 6/May/2020
    Email : bernardo.cardenas@udem.edu
"""

import numpy as np
import time
import utility_functions as uf

def main():
    """
        Main function that runs the k nearest neighbours algorithm
    """
    initial_time = time.time()
    #load data from csv
    #if flag = 1, the training data is printed. if flag = 0, data is not printed
    flag = 0
    x_train, y_train, x_testing, y_testing = uf.load_data('diabetes.csv', 1, 95, flag)
    
    #scale training data
    x_train, mean, std = uf.scale_data(x_train, flag)

    #initialize hyperparameters
    k = 3

    #run KNN algorithm
    #predictions = uf.k_nearest_neighbours(k, x, y)

    testing_point = np.array([[0.0,4.0,3.0]])
    x = np.array([[0,3,0],[2,0,0],[0,1,3],[0,1,2],[-1,0,1],[1,1,1]])
    y = np.array([[0],[0],[0],[1],[1],[1]])
    p = uf.compute_conditional_probabilities(k, testing_point, x, y)
    print(uf.predict(p))
    #scale testing data
   # x_testing = uf.scale_data(x_testing, flag, mean, std)



    #predict using testing data
    #predictions = uf.predict(x_testing, w)

    #obtain confusion matrix
    #confusion_matrix = uf.get_confusion_matrix(predictions, y_testing)

    #print matrix and performance metrics
    #uf.print_performance_metrics(confusion_matrix)

    print('Computing time in seconds:', float(time.time() - initial_time))

#call main function
main()