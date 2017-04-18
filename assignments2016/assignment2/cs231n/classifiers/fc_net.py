import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################

    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

  def sigmoid(self, x):
    return 1 / ( 1 + np.exp(-x))

  def relu(self, x):
    return np.maximum(0, x)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################

    # If X has more than 2 dimensions then reshape so that X has 2 dimensions
    if len(X.shape) > 2:
      X =  np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))

    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    reg = self.reg

    N = X.shape[0]

    a1 = np.dot(X, W1) + b1
    hidden_layer = self.relu(a1) # ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(N),y])
    data_loss = np.sum(corect_logprobs)/N
    reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    grads = {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    # compute the gradient on scores
    dscores = probs
    dscores[range(N),y] -= 1
    dscores /= N

    # W2 and b2
    grads['W2'] = np.dot(hidden_layer.T, dscores)
    grads['b2'] = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.hidden_layers = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################

    # print
    # print 'Initialize the parameters of the network'
    # print

    for i in range(1, self.num_layers+1):
      W = 'W'+str(i)
      b = 'b'+str(i)

      if i == 1:
        self.params[W] = weight_scale * np.random.randn(input_dim, hidden_dims[i-1])
        self.params[b] = np.zeros(hidden_dims[i-1])

      elif i == self.num_layers:
        self.params[W] = weight_scale * np.random.randn(hidden_dims[i-2], num_classes)
        self.params[b] = np.zeros(num_classes)

      else:
        self.params[W] = weight_scale * np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
        self.params[b] = np.zeros(hidden_dims[i-1])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def sigmoid(self, x):
    return 1 / ( 1 + np.exp(-x))


  def relu(self, x):
    return np.maximum(0, x)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """

    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    cache_dict = {}
    cache_dict_counter = 0

    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    if len(X.shape) > 2:
      X =  np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))


    # for p in self.params:
    #   d = {k: v for k, v in self.params.iteritems()}
    #   p = d

    N = X.shape[0]
    hidden_layer = X
    reg_loss = 0

    for i in range(1, self.num_layers+1):

      cache_dict[i] = {}

      W = 'W'+str(i)
      b = 'b'+str(i)

      # Compute regularisation loss
      reg_loss += 0.5 * self.reg * np.sum(self.params[W]*self.params[W])

      # For every layer but not the last
      # if i < self.num_layers:

      # affine
      a, cache_dict[i]['affine'] = affine_forward(hidden_layer, self.params[W], self.params[b])
      #print 'forward affine'
      cache_dict_counter += 1

      # if last layer then only make affine_forward
      if i == self.num_layers:
        scores = a
        break

      # batch norm
      # ToDo

      # relu
      hidden_layer, cache_dict[i]['relu'] = relu_forward(a)
      #print 'forward relu'

      cache_dict_counter += 1

      # dropout
      if self.use_dropout:
        hidden_layer, cache_dict[i]['dropout'] = dropout_forward(hidden_layer, self.dropout_param)
        #print 'forward dropout'

        cache_dict_counter += 1

        self.hidden_layers[i] = hidden_layer

        # OLD
        # a = np.dot(hidden_layer, self.params[W]) + self.params[b]
        # hidden_layer = self.relu(a) # ReLU activation
        # self.hidden_layers[i] = hidden_layer

      # For last layer
      # else:
      #   scores, cache_dict[i]['affine'] = affine_forward(hidden_layer, self.params[W], self.params[b])
      #   #print 'forward affine'
      #
      #   cache_dict_counter += 1

        # OLD
        # scores = np.dot(hidden_layer, self.params[W]) + self.params[b]



    #print 'softmax'

    # dscores = dx

    # old
    # compute the class probabilities
    # exp_scores = np.exp(scores)
    # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    #
    # # average cross-entropy loss and regularization
    # correct_logprobs = -np.log(probs[range(N),y])
    # data_loss = np.sum(correct_logprobs)/N

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      #print '---- return scores - no back ----'
      return scores

    grads = {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # print
    # print 'Implement the backward pass'
    # print

    # OLD
    # compute the gradient on scores
    # dscores = probs
    # dscores[range(N),y] -= 1
    # dscores /= N

    # Loop backwards
    #print "len(cache_dict)", len(cache_dict)

    #for k in list(reversed(sorted(cache_dict.keys()))):

      #print "len(cache_dict[k])", len(cache_dict[k])

      # affine backward
      #dx, dw, db = affine_backward(dx, cache_dict[k])

      # dropout backward

      # relu backward

      # batch norm backward


      #print "len(cache_dict[k])", len(cache_dict[k])

    # for cache in cache_dict:
    #   d = {k: v for k, v in cache_dict.iteritems()}
    #   print d

    # softmax loss
    loss, dx = softmax_loss(scores, y)

    loss += + reg_loss

    # Loop backwards, i = 1 is last value
    for i in xrange(self.num_layers, 0, -1):

      W = 'W'+str(i)
      b = 'b'+str(i)

      # dropout
      if 'dropout' in cache_dict[i]:
        #print 'back dropout'

        dx = dropout_backward(dx, cache_dict[i]['dropout'])

      # relu
      if 'relu' in cache_dict[i]:
        #print 'back relu'

        dx = relu_backward(dx, cache_dict[i]['relu'])

      if 'batch' in cache_dict[i]:
        #print 'back batch'
        ok = 1

      # affine
      if 'affine' in cache_dict[i]:
        #print 'back affine'

        dx, dw, db = affine_backward(dx, cache_dict[i]['affine'])

      grads[W] = dw
      # add regularisation
      grads[W] += self.reg * self.params[W]

      grads[b] = db


      # if i == self.num_layers:
      #
      #   grads[W] = np.dot(self.hidden_layers[i-1].T, dscores)
      #   grads[b] = np.sum(dscores, axis=0)
      #
      #   # add regularisation
      #   grads[W] += self.reg * self.params[W]
      #
      #   # next backprop into hidden layer
      #   dhidden = np.dot(dscores, self.params[W].T)
      #
      #   # backprop the ReLU non-linearity
      #   dhidden[self.hidden_layers[i-1] <= 0] = 0
      #
      # elif i > 1:
      #
      #   grads[W] = np.dot(self.hidden_layers[i-1].T, dhidden)
      #   grads[b] = np.sum(dhidden, axis=0)
      #
      #   # add regularisation
      #   grads[W] += self.reg * self.params[W]
      #
      #   # next backprop into hidden layer
      #   dhidden = np.dot(dhidden, self.params[W].T)
      #
      #   # backprop the ReLU non-linearity
      #   dhidden[self.hidden_layers[i-1] <= 0] = 0
      #
      # else:
      #   # finally into W,b
      #   grads[W] = np.dot(X.T, dhidden)
      #   grads[b] = np.sum(dhidden, axis=0)
      #
      #   # add regularization gradient contribution
      #   grads[W] += self.reg * self.params[W]


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
