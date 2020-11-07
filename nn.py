import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90
# NUM_FEATS = 2
WEIGHT_INIT_RANGE = (-1, 1)

def relu(w):
    return np.clip(w, 0, None)

def reludash(w):
        return 1 if w > 0 else 0
    
relu_dash = np.vectorize(reludash)
    

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

        # initialize weights
        f = lambda size: np.random.uniform(*WEIGHT_INIT_RANGE, size)

        weights, biases = [], []

        # weight matrix for hidden layer 1 is of different size: d -> M.
        if num_layers > 0:
            weight = f((NUM_FEATS, num_units)) # (d, m)
            bias = f(num_units) # (m,)

            weights.append(weight)
            biases.append(bias)

        # more hidden layers if required.
        for i in range(num_layers - 1):
            weight = f((num_units, num_units))
            bias = f(num_units) # shape (num_units, )

            weights.append(weight)
            biases.append(bias)

        # add final layer.
        if num_layers > 0:
            weight = f((num_units, 1))
        else:
            weight = f((NUM_FEATS, 1))
        bias = f(1) # shape (1,)

        weights.append(weight)
        biases.append(bias)

        # ds to save the output of each layer including input and output layer.
        layer_outs = [None for _ in range(num_layers + 2)]
        self.weights = weights
        self.biases = biases
        self.layer_outs = layer_outs

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
        try:
            self.layer_outs[0] = X # (n, d)
            prev_out = X # n * d

            # TODO: check the indices of iteration. 
            for i in range(len(self.layer_outs) - 1):
#                 print("prev_out = {}".format(prev_out))
                temp = prev_out.dot(self.weights[i]) # (n, d) * (d, m) -> (n, m)
                temp += self.biases[i]

                # relu activation
#                 print("layer {} before relu: {}".format(i, temp))
                if i != len(self.layer_outs) - 2:
                    temp = relu(temp)
#                 print("layer {} after relu: {}".format(i, temp))
                # temp = np.clip(temp, 0, None)

                self.layer_outs[i + 1] = temp
                prev_out = temp


        except Exception as e:
            print(e)
            print("weights = ", self.weights)
            print("len(weights) = ", len(self.weights))
            # print context.
            self.summary()
            print("X = {}".format(X))
            print("Error occured while processing layer {}.".format(i))

            raise Exception

#         print("layer_outs = ", self.layer_outs)
        return self.layer_outs[-1]

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
        # perform forward pass.
        y_pred = self.__call__(X) # (n, 1)

        # init the del vectors.
        del_W = [None for _ in self.weights]
        del_b = [None for _ in self.biases]

        # delta for output layer weights.
        n = len(y)
        del_cost = (y_pred-y) # (n, 1)
        del_relu = relu_dash(y_pred) # (n, 1)
        delta_j = del_cost #* del_relu # element-wise mult. (n, 1)
        del_b_temp = np.sum(delta_j, 0) / n # (1,)
        del_w_temp = self.layer_outs[-2].T.dot(delta_j) / n # (m, n) x (n, 1) -> (m, 1)

        # derivative of regularization term. 2 omitted.
        del_w_temp += lamda * self.weights[-1]
        del_b_temp += lamda * self.biases[-1]

        del_W[-1] = del_w_temp
        del_b[-1] = del_b_temp

        # d -> m -> m'
        for i in range(2, len(self.weights) + 1):
            del_relu = relu_dash(self.layer_outs[-i]) # (n, m)
            # delta_k is deltas of next layer. (n, m')
            # w_kj is next layer weights. (m, m')
            # delta_k = delta_j.copy() # (n, m')
            # weights[-i + 1] is (m, m')
            del_temp = delta_j.dot(self.weights[-i + 1].T) # (n, m)
            delta_j = del_temp * del_relu # (n, m); elementwise mult.

            del_b_temp = np.sum(delta_j, 0) / n # (m,)
            del_w_temp = self.layer_outs[-i - 1].T.dot(delta_j) / n # (d, n)x(n, m) -> (d, m)

            # del of regularization
            del_w_temp += lamda * self.weights[-i]
            del_b_temp += lamda * self.biases[-i]

            del_W[-i] = del_w_temp # (d, m)
            del_b[-i] = del_b_temp # (m, )

        return del_W, del_b

    def summary(self):
        print("Network Summary.")

        # print all the weight and bias shapes
        print("Input: (n, d)")
        prev = ("n", "d")
        for i, weight in enumerate(self.weights):
            print("Layer {}: weights: {}, biases: {}, output: {}"
                    .format(i + 1, weight.shape,
                            self.biases[i].shape,
                            (prev[0],) + weight.shape[1:]))
#             print("weight: ", weight)
#             print("bias: ", self.biases[i])



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
        for weight, delta_weight in zip(weights, delta_weights):
            weight -= self.learning_rate * delta_weight

        for bias, delta_bias in zip(biases, delta_biases):
            bias -= self.learning_rate * delta_bias


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
    print("shape y: {}, y_hat: {}".format(y.shape, y_hat.shape))
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
        reg += np.square(weight).sum()

    for bias in biases:
        # bias vector for layer.
        reg += np.square(bias).sum()

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
    print("Training...")
    for epoch in range(max_epochs):
        for i in range(0, len(train_input), batch_size):
            batch_x = train_input[i: i + batch_size]
            batch_y = train_target[i: i + batch_size]

            del_W, del_b = net.backward(batch_x, batch_y, lamda)
#             print("Gradients:")
#             print("del_W: {}".format(del_W))
#             print("del_b: {}".format(del_b))
            optimizer.step(net.weights, net.biases, del_W, del_b)

        # epoch done.
        train_preds = net(train_input)
        dev_preds = net(dev_input)
#         print("train preds: {}, dev_preds: {}".format(train_preds, dev_preds))
        print("Epoch {}: train loss: {}, dev loss: {}, dev RMSE: {}".format(
            epoch + 1,
            loss_fn(train_target, train_preds, net.weights, net.biases, lamda),
            loss_fn(dev_target, dev_preds, net.weights, net.biases, lamda),
            rmse(dev_target, dev_preds)
        ))

    return rmse(dev_target, net(dev_input))


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
    return net(inputs)

def read_data():
    '''
    Read the train, dev, and test datasets
    '''
    df = pd.read_csv("../input/cs725-autumn-2020-programming-assignment-2/dataset/train.csv")
    train_target = df["label"].to_numpy().reshape((-1, 1))
    train_input = df.iloc[:, 1:].to_numpy().reshape((-1, NUM_FEATS))
    
    df = pd.read_csv("../input/cs725-autumn-2020-programming-assignment-2/dataset/dev.csv")
    dev_target = df["label"].to_numpy().reshape((-1, 1))
    dev_input = df.iloc[:, 1:].to_numpy().reshape((-1, NUM_FEATS))
    
    df = pd.read_csv("../input/cs725-autumn-2020-programming-assignment-2/dataset/test.csv")
    test_input = df.to_numpy().reshape((-1, NUM_FEATS))

    return train_input, train_target, dev_input, dev_target, test_input

def dummy_data():
    # y = 4x1 + 2x2 -1
    train_input = np.array([[0.5, 3.1],
       [1. , 2. ],
       [5. , 4. ],
       [0.1, 0.2],
       [1. , 1. ],
       [2. , 2. ]])
    train_target = np.array([[ 7.2],
       [ 7. ],
       [27. ],
       [-0.2],
       [ 5. ],
       [11. ]])
    
    dev_input = np.array([[ 3. ,  3. ],
       [ 0.2, -0.1],
       [-4. ,  2. ]])

    dev_target = np.array([[ 17. ],
       [ -0.4],
       [-13. ]])
    return train_input, train_target, dev_input, dev_target, None
    

def main():

    # These parameters should be fixed for Part 1
    max_epochs = 50
    batch_size = 128


    learning_rate = 0.001
    num_layers = 1
    num_units = 64
    lamda = 0 #0.00001 # Regularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data() #dummy_data() 
    net = Net(num_layers, num_units)
    net.summary()
    optimizer = Optimizer(learning_rate)
    rmse_dev = train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )
    print("final rmse on dev = {}.".format(rmse_dev))
#     get_test_data_predictions(net, test_input)


if __name__ == '__main__':
    main()
