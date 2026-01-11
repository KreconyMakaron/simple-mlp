# Simple MLP

A very simple implementation of a multilayer perceptron built with NumPy. This was mainly done as a learning exercise for a machine learning course.

It uses stochastic gradient descent with momentum, with an optional L2 regularisation term. There’s also some basic logging of loss and accuracy so you can plot them if you want.

All hidden layers use ReLU activations, with a sigmoid on the output layer.

----

You can see an example of how it’s used in the `mlp.ipynb` notebook. In that notebook, the model is trained on a [binary breast cancer dataset from Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset). To that end, I added a small `util.py` file to make loading the dataset a bit cleaner.
