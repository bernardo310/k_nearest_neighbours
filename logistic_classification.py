""" 
    This script tests logistic classification algorithm
    Author : Bernardo Cardenas Domene
    Institution : Universidad de Monterrey
    First Created : 6/April/2020
    Email : bernardo.cardenas@udem.edu
"""

import numpy as np
import time
import utility_functions as uf

def main():
    """
        Main function that runs the logistic classification algorithm
    """
    initial_time = time.time()
    #load data from csv
    #if flag = 1, the training data is printed. if flag = 0, data is not printed
    flag = 0
    x_train, y_train, x_testing, y_testing = uf.load_data('diabetes.csv', 1, 80, flag)
    
    #scale training data
    x_train, mean, std = uf.scale_data(x_train, flag)

    #initialize hyperparameters
    learning_rate = 0.001
    stopping_criteria = 0.01

    #initialize w with zeros of size of columns in x data + 1 (w0...wn)
    w = np.zeros((len(x_train[0]) + 1, 1)) 

    #run gradient descent
    w = uf.gradient_descent(x_train, y_train, w, learning_rate, stopping_criteria)

    #scale testing data
    x_testing = uf.scale_data(x_testing, flag, mean, std)

    #printing w parameters
    print('-'*120)
    print('w Parameters')
    print('-'*120)
    for i in range(len(w)): print('w'+ str(i) + ': ' + str(w[i][0]))
    print('\n')

    #predict using testing data
    predictions = uf.predict(x_testing, w)

    #obtain confusion matrix
    confusion_matrix = uf.get_confusion_matrix(predictions, y_testing)

    #print matrix and performance metrics
    uf.print_performance_metrics(confusion_matrix)

    print('Computing time in seconds:', float(time.time() - initial_time))

#call main function
main()