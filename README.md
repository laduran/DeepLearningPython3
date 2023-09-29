## Overview

### neuralnetworksanddeeplearning.com integrated scripts for Python 3.11.1

These scrips are updated ones from the **neuralnetworksanddeeplearning.com** gitHub repository in order to work with Python 3.5.2

The testing file (**test.py**) contains all three networks (network.py, network2.py, network3.py) from the book and it is the starting point to run (i.e. *train and evaluate*) them.

## Just type at shell: **python3.5 test.py**

In test.py there are examples of networks configurations with proper comments. I did that to relate with particular chapters from the book.

## Other Implementations ##

### mnist_average_darkness.py: ###

This is a very naive implementation of predicting digits
based on the average darkness of the image file.

### mnist_svm.py: ###

This is an implementation using Scikit-learn library for ML
Uses a Support Vector Machine to learn the handwriting data

### mnist_tf_svm.py: ###

This is an implementation using the TensorFlow library for ML
Uses a Support Vector Machine to learn the handwriting data
TODO: convert this code so that it will choose correct GPU in a two GPU system.
NOTE: This will only use GPU support if Tensorflow has the correct GPU libraries 
installed.
