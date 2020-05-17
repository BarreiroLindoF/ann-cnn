from builtins import object
import numpy as np

from layers import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - linear - relu - linear - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize Weights and Biases
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        # Assuming a shape identical to the input image for the conv layer output
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W // 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in layers.py                  #
        ############################################################################
        # Convolution -> ReLu -> 2x2 Max Pool -> Linear Layer -> ReLu -> Linear Layer -> Softmax
        
        out1, cache1 = conv_forward(X, W1, b1, conv_param)
        out2, cache2 = relu_forward(out1)
        out3, cache3 = max_pool_forward(out2, pool_param)
        out4, cache4 = linear_relu_forward(out3, W2, b2)
        out5, cache5 = linear_forward(out4, W3, b3)
        scores = out5

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = None, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        W1_reg = np.sum(self.reg * 0.5 * W1 ** 2)
        W2_reg = np.sum(self.reg * 0.5 * W2 ** 2)
        W3_reg = np.sum(self.reg * 0.5 * W3 ** 2)

        loss = loss + W1_reg + W2_reg + W3_reg

        dx5, grads['W3'], grads['b3'] = linear_backward(dscores, cache5)
        dx4, grads['W2'], grads['b2'] = linear_relu_backward(dx5, cache4)
        dx3 = max_pool_backward(dx4, cache3)
        dx2 = relu_backward(dx3, cache2)
        dx1, grads['W1'], grads['b1'] = conv_backward(dx2, cache1)

        grads['W3'] += self.reg * self.params['W3']
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def train(self, X, y, learning_rate=1e-3, num_epochs=10,
              batch_size=2, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train = X.shape[0]
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        acc_history = []

        for epoch in range(1, num_epochs):
            print("Epoch", epoch)
            loss_epoch = []
            acc_epoch = []

            for it in range(iterations_per_epoch):
                print("Batch", it)
                X_batch = None
                y_batch = None

                indices = np.random.choice(num_train, batch_size)
                X_batch = X[indices]
                y_batch = y[indices]

                # evaluate loss and gradient
                loss, grads = self.loss(X_batch, y_batch)
                print("loss", loss)
                loss_epoch.append(loss)

                # perform parameter update
                #########################################################################
                # TODO:                                                                 #
                # Update the weights using the gradient and the learning rate.          #
                #########################################################################
                self.params['W3'] -= learning_rate * grads['W3']
                self.params['W2'] -= learning_rate * grads['W2']
                self.params['W1'] -= learning_rate * grads['W1']

                #########################################################################
                #                       END OF YOUR CODE                                #
                #########################################################################

                acc_epoch.append(np.mean(self.predict(X_batch) == y_batch))
                print("accuracy", acc_epoch[-1])

            loss_history.append(np.mean(loss_epoch))
            acc_history.append(np.mean(acc_epoch))

            if verbose and epoch % 10 == 0:
                print('epoch {} / {} : loss {}'.format(epoch, num_epochs, loss), end='\r')

        if verbose:
            print(''.ljust(70), end='\r')

        return loss_history, acc_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        out1, cache1 = conv_forward(X, self.params['W1'], self.params['b1'], conv_param)
        out2, cache2 = relu_forward(out1)
        out3, cache3 = max_pool_forward(out2, pool_param)
        out4, cache4 = linear_relu_forward(out3, self.params['W2'], self.params['b2'])
        out5, cache5 = linear_forward(out4, self.params['W3'], self.params['b3'])
        exp = np.exp(out5)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        y_pred = np.argmax(probs, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred
