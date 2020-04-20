""" 
    This script visualizes data from csv file in histograms
    Author : Bernardo Cardenas Domene
    Institution : Universidad de Monterrey
    First Created : 20/April/2020
    Email : bernardo.cardenas@udem.edu
"""

import numpy as np
import matplotlib.pylab as plt
import time
import utility_functions as uf

def main():
    """
        Main function used for visualization of histograms by columns in csv file
    """
    #get data
    x_train, y_train, x_testing, y_testing = uf.load_data('diabetes.csv', 1, 100, 0)
    #show histograms
    f1 = plt.figure()
    plt.hist(x_train[:,[0]])
    plt.title('Pregnancies')
    plt.xlabel('Pregnancies')
    plt.ylabel('Number of entries')
    plt.show(block=False)

    f2 = plt.figure()
    plt.hist(x_train[:,[1]])
    plt.title('Glucose')
    plt.xlabel('Glucose')
    plt.ylabel('Number of entries')

    plt.show(block=False)

    f3 = plt.figure()
    plt.hist(x_train[:,[2]])
    plt.title('Blood Pressure')
    plt.xlabel('Blood Pressure')
    plt.ylabel('Number of entries')
    plt.show(block=False)

    f4 = plt.figure()
    plt.hist(x_train[:,[3]])
    plt.title('Skin Thickness in Triceps')
    plt.xlabel('Skin Thickness in Triceps')
    plt.ylabel('Number of entries')
    plt.show(block=False)

    f5 = plt.figure()
    plt.hist(x_train[:,[4]])
    plt.title('Insulin')
    plt.xlabel('Insulin')
    plt.ylabel('Number of entries')
    plt.show(block=False)

    f6 = plt.figure()
    plt.hist(x_train[:,[5]])
    plt.title('BMI')
    plt.xlabel('BMI')
    plt.ylabel('Number of entries')
    plt.show(block=False)

    f7 = plt.figure()
    plt.hist(x_train[:,[6]])
    plt.title('Diabetes Pedigree Function')
    plt.xlabel('Diabetes Pedigree Function')
    plt.ylabel('Number of entries')
    plt.show(block=False)

    f8 = plt.figure()
    plt.hist(x_train[:,[7]])
    plt.title('Age')
    plt.xlabel('Age')
    plt.ylabel('Number of entries')
    plt.show(block=False)

    f9 = plt.figure()
    plt.hist(y_train[:,[0]])
    plt.title('Outcome')
    plt.xlabel('Outcome')
    plt.ylabel('Number of entries')
    plt.show(block=False)










    plt.show()
#call main function
main()