import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:, y[i]] += -X[i]
        dW[:, j] += X[i] # gradient update for incorrect rows
        loss += margin

  # Average gradients as well
  dW /= num_train

  # Add regularization to the gradient
  dW += reg * W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  print 'X.shape: ', X.shape
  print 'y.shape: ', y.shape
  print 'W.shape: ', W.shape

  scores = X.dot(W) # 500 x 10 matrix

  print 'scores.shape: ', scores.shape

  correct_scores = np.ones(scores.shape) * y[:,np.newaxis] # 500 x 10

  deltas = np.ones(scores.shape) # 1 matrix, 500 x 10

  L = scores - correct_scores + deltas

  print 'L.shape: ', L.shape

  L[L < 0] = 0 # set all negative values to 0, replaces max(0, scores - scores[y] + 1)
  L[np.arange(0, scores.shape[0]), y] = 0 # don't count y_i

  # sum losses of single image per row, results in column vector: 500 x 1
  loss = np.sum(L, axis=1)

  # caluclate final average loss
  loss = np.sum(loss) / X.shape[0]

  # Add L2 regularization
  loss += 0.5 * reg * np.sum(W * W)

  print 'loss', loss


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #L[L > 0] = 1
  #L[np.arange(0, scores.shape[0]), y] = -1 * np.sum(L, axis=1)
  #dW = np.dot(L, X.T)

  dW = np.gradient(scores)

  # Average over number of training examples
  #num_train = X.shape[0]
  #dW /= num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
