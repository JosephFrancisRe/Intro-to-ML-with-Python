# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 23:17:21 2022

@author: JosephRe
"""

import mglearn
import matplotlib.pyplot as plt
import numpy as np

graphDisplay = None

while graphDisplay != 3:
    
    try:
        graphDisplay = int(input("Guide: 1 = make_forge(). 2 = mage_wave. 3 = Exit.\n\nEnter a graph display value: "))
    except ValueError:
        print("\nThat's not an integer value.")
    
    # make_forge()
    # displays when user inputs 1
    if graphDisplay == 1:
        X, y = mglearn.datasets.make_forge()
    
        mglearn.discrete_scatter(X[:,0], X[:,1], y)
        plt.legend(["Class 0", "Class 1"], loc=4)
        plt.xlabel("First feature")
        plt.ylabel("Second feature")
        
        plt.show()
    # make_wave
    # displays when user inputs 2
    elif graphDisplay == 2:
        X, y = mglearn.datasets.make_wave(n_samples = 40)

        plt.plot(X, y, 'o')
        plt.ylim(-3,3)
        plt.xlabel('Feature')
        plt.ylabel('Target')
        
        plt.show()
    elif graphDisplay == 3:
        print('Exiting program.')
    else:
        print("\nImproper display value.")