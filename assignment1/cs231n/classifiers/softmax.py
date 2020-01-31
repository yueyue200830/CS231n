from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    single_loss = np.zeros(num_train)

    for i in range(num_train):
        # Compute exp score.
        scores = X[i].dot(W)
        scores = np.exp(scores - np.amax(scores))
        class_probability = scores / np.sum(scores)

        # Compute current loss
        single_loss[i] = - np.log(class_probability[y[i]])

        # Compute dW
        dW += class_probability * X[i].reshape(-1, 1)
        dW[:, y[i]] -= X[i]

    loss = np.mean(single_loss) + reg * np.sum(W * W)

    dW = dW / num_train + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    single_loss = np.zeros(num_train)

    scores = X.dot(W)
    scores = np.exp(scores - np.max(scores, axis = 1).reshape(-1, 1))
    class_probability = scores / np.sum(scores, axis = 1).reshape(-1, 1)

    single_loss = - np.log(class_probability[np.arange(num_train), y])
    loss = np.mean(single_loss) + reg * np.sum(W * W)

    correct_matrix = np.zeros_like(class_probability)
    correct_matrix[np.arange(num_train), y] = 1
    dW = X.T.dot(class_probability - correct_matrix)
    dW = dW / num_train + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
