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
    num_class = W.shape[1]

    for i in range(num_train):
        score = X[i].dot(W)
        exp_score = np.exp(score)
        loss += - np.log(exp_score[y[i]] / exp_score.sum())
        for j in range(num_class):
            if j == y[i]:
                dW[:, y[i]] += (exp_score[y[i]] / exp_score.sum() - 1) * X[i].T
            else:
                dW[:, j] += (exp_score[j] / exp_score.sum()) * X[i].T

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    dW /= num_train
    dW += reg * W

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
    num_class = W.shape[1]
    
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    correct_class_exp_scores = exp_scores[range(num_train), y]
    loss_individual = - np.log(correct_class_exp_scores / exp_scores.sum(axis=1))
    loss += loss_individual.sum()
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    # Unnecessary to start with np.zeros()
    coeff_dW = exp_scores / exp_scores.sum(axis=1).reshape((-1, 1))
    coeff_dW[range(num_train), y] -= 1
    dW += np.dot(X.T, coeff_dW)
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
