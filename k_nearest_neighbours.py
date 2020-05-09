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
    
    #print testing data
    print('-'*100)
    print('First 10 Testing Data')
    print('-'*100)    
    print('{0: >20}'.format("Pregnancies"),'{0: >20}'.format("Glucose"),'{0: >20}'.format("BloodPressure"),'{0: >20}'.format("SkinThickness"),'{0: >20}'.format("Insulin"),'{0: >20}'.format("BMI"),'{0: >20}'.format("DiabetesPedigreeFunction"),'{0: >20}'.format("Age"))
    for i in range(len(x_testing)): print('{0: >20}'.format(x_testing[i][0]),'{0: >20}'.format(x_testing[i][1]),'{0: >20}'.format(x_testing[i][2]),'{0: >20}'.format(x_testing[i][3]),'{0: >20}'.format(x_testing[i][4]),'{0: >20}'.format(x_testing[i][5]),'{0: >20}'.format(x_testing[i][6]),'{0: >20}'.format(x_testing[i][7]))
    print('\n')

    #scale training data
    x_train, mean, std = uf.scale_data(x_train, flag)

    #scale testing data
    x_testing = uf.scale_data(x_testing, flag, mean, std)

    #print testing data
    print('-'*100)
    print('Scaled Testing Data')
    print('-'*100)    
    print('{0: >20}'.format("Pregnancies"),'{0: >20}'.format("Glucose"),'{0: >20}'.format("BloodPressure"),'{0: >20}'.format("SkinThickness"),'{0: >20}'.format("Insulin"),'{0: >20}'.format("BMI"),'{0: >20}'.format("DiabetesPedigreeFunction"),'{0: >20}'.format("Age"))
    for i in range(len(x_testing)): print('{0: >20}'.format(x_testing[i][0]),'{0: >20}'.format(x_testing[i][1]),'{0: >20}'.format(x_testing[i][2]),'{0: >20}'.format(x_testing[i][3]),'{0: >20}'.format(x_testing[i][4]),'{0: >20}'.format(x_testing[i][5]),'{0: >20}'.format(x_testing[i][6]),'{0: >20}'.format(x_testing[i][7]))
    print('\n')

    #initialize hyperparameters
    k = 5

    #run KNN algorithm for testing data and obtain predictions
    predictions = uf.predict(k, x_train, y_train, x_testing)

    #obtain confusion matrix
    confusion_matrix = uf.get_confusion_matrix(predictions, y_testing)



    #print matrix and performance metrics
    uf.print_performance_metrics(confusion_matrix)

    print('Computing time in seconds:', float(time.time() - initial_time))

#call main function
main()