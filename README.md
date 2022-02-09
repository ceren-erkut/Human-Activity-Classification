# Classify 6 types of human activity from time series sensor data

Consider classifying human activity (downstairs=1, jogging=2, sit- ting=3, standing=4, upstairs=5, walking=6) from movement signals measured with three sensors simultaneously. The file assign3_data3.h5 contains time series of training and testing data (trX and tstX), and their corresponding labels (trY and tstY). The length of each time series is 150 units. The training set consists of 3000 samples, and the test set consists of 600 samples. Implement fundamental recurrent neural network architectures, trained with back propagation through time to solve a multi-class time series classification problem.

## Part A

Using the **_back propagation through time algorithm_**, implement a _**single layer recurrent neural network**_ with 128 neurons and _**hyperbolic tangent**_ activation function, followed by a _**multi-layer perceptron**_ with a _**softmax**_ function for classification. Use: a _**stochastic gradient descent**_ algorithm, _**mini-batch**_ size of 32 samples, learning rate of η = 0.1, _**momentum rate**_ of α = 0.85, maximum of 50 epochs, and weights/biases initialized with _**Xavier Uniform distribution**_. Adjust the parameters, and number of hidden layers of the classification neural network to improve network performance. The algorithm should be stopped based on the _**categorical cross-entropy error**_ on a validation data (10% samples selected from the training data). Report the following: Validation error as a function of epoch number, accuracy measured over the test dataset, _**confusion matrix**_ for the training and test set, and discussion of your results.

## Part B

For the time-series data, it is vital to summarize the past observations in the hidden state and to control this information. For this reason, we consider a better alternative which is a long-short term memory or _**LSTM**_ neural network. Repeat part a for LSTM. Report the following: Validation error as a function of epoch number, accuracy measured over the test set, confusion matrix for the training and test set, discussion of your results, and comparison with the performance in part a.

## Part C

Finally, we consider an alternative to LSTM neural networks, called gated recurrent units (_**GRU**_ in short). Repeat part a for GRU. Report the following: Validation error as a function of epoch number, accuracy measured over the test set, confusion matrix for the training and test set, discussion of your results, and comparison with the performance in parts a and b.

