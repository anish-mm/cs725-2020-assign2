import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90
WEIGHT_INIT_RANGE = (-1, 1)

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		def relu(w):
			return np.clip(w, 0, None)

		# initialize weights
		f = np.random.uniform
		
		weights, biases, layers = [], [], []

		# weight matrix for hidden layer 1 is of different size: d -> M.
		if num_layers > 0:
			weight = f(*WEIGHT_INIT_RANGE, (NUM_FEATS, num_units))
			bias = f(*WEIGHT_INIT_RANGE, num_units)
			layer = []
			
			weights.append(weight)
			biases.append(bias)
			layers.append(layer)

		# more hidden layers if required.
		for i in range(num_layers - 1):
			weight = f(*WEIGHT_INIT_RANGE, (num_units, num_units))
			bias = f(*WEIGHT_INIT_RANGE, num_units)
			layer = []

			weights.append(weight)
			biases.append(bias)
			layers.append(layer)
		
		# add final layer.
		if num_layers > 0:
			weight = f(*WEIGHT_INIT_RANGE, (num_units, 1))
		else:
			weight = f(*WEIGHT_INIT_RANGE, (NUM_FEATS, 1))
		bias = f(*WEIGHT_INIT_RANGE, 1)
		layer = []

		weights.append(weight)
		biases.append([bias])
		layers.append(layer)

		self.weights = weights
		self.biases = biases
		self.layers = layers		

		self.forward_pass_done = False

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.
		
		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''

		layers = self.layers

		try:
			for i, layer in enumerate(layers):
				temp = X.dot(self.weights[i])
				temp += self.biases[i]

				# relu activation
				temp = np.clip(temp, 0, None)

				layer = temp

			self.forward_pass_done = True

		except Exception as e:
			print(e)

			# print context.
			self.__summary()
			print("X = {}".format(X))
			print("Error occured while processing layer {}.".format(i))
			
			raise Exception


	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing bacward pass.
		'''
		raise NotImplementedError
	
	def __summary(self):
		print("Network Summary.")

		# print all the weight and bias shapes
		print("Input: (n, d)")
		prev = ("n", "d")
		for i, weight in enumerate(self.weights):
			print("Layer {}: weights: {}, biases: {}, output: {}"
					.format(i + 1, weight.shape, self.biases[i].shape, (prev[0],) + weight.shape[1:]))
		


class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Stochastic Gradient Descent (SGD) based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.learning_rate = learning_rate

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		for weight in weights:
			weight -= self.learning_rate * delta_weights
		
		for bias in biases:
			bias -= self.learning_rate * delta_biases


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	n = y.shape[0]
	diff = y - y_hat
	loss = diff.T.dot(diff) / n
	return loss

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	reg = 0

	for weight in weights:
		# weight matrix for a layer. d|M -> M
		l2_vecs = weight.T.dot(weight)
		l2 = l2_vecs.T.dot(l2_vec)
		reg += l2
	
	for bias in biases:
		# bias vector for layer.
		l2 = bias.dot(bias)
		reg += l2
	
	return reg

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	return loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	n = y.shape[0]
	diff = y - y_hat
	loss = diff.T.dot(diff) / n
	return np.sqrt(loss) 


def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.
	'''
	raise NotImplementedError

def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	return net(X)

def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	raise NotImplementedError

	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 128


	learning_rate = 0.001
	num_layers = 1
	num_units = 64
	lamda = 0.1 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()
