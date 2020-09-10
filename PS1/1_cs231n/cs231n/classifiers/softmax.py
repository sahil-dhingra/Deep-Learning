import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  N = X.shape[1]
  z = np.dot(W, X)
  p = np.exp(z-z.max())/np.sum(np.exp(z-z.max()), axis = 0).reshape((1, z.shape[1]))
  labels = np.zeros((y.size, y.max()+1))
  labels[np.arange(y.size), y] = 1
  L = (-1/N)*np.sum(labels*np.log(p.T))
  # L = -(1/N)*np.sum([np.log(p[y[k]]) for k in range(N)])
  R = np.sum(np.square(W))
  loss = L + reg*R
  dW = (-1/N)*np.dot((labels.T-p), X.T) + 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
