"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function
import os
import sys
import timeit

import numpy
import scipy.io as sio

import theano
import tensorflow as tf
import theano.tensor as T
from theano.tensor.signal import pool
#import my_pool as pool
#import original_pool as pool
#from pylearn2.models import pca
import numpy as np
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
import matplotlib.pyplot as plt
from ReadStylesDataset import read_cmu_style, read_sub_cmu_style, read_two_cmu_style
from scipy.misc import imsave, toimage
import matplotlib.image as mpimg

#theano.config.compute_test_value = 'raise'
theano.config.exception_verbosity='high'
num_pass = 0

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        
        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        print ("-------------------------------PASS ONCE = ", num_pass, " -------------------------------")
def shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')
def load_mocap():
    test = sio.loadmat('test.mat')
    train = sio.loadmat('train.mat')
    test_x = test['set_test']
    test_y = test['set_test_label']
    train_x = train['set_train']
    train_y = train['set_train_label']

    #Test size 100
    test_x = train_x[500:550, :, :]
    test_y = train_y[500:550, :]
    test_x = numpy.concatenate((test_x,test_x,test_x,test_x,test_x,test_x,test_x,test_x,test_x,test_x), axis = 0)
    test_y = numpy.concatenate((test_y, test_y, test_y, test_y, test_y, test_y, test_y, test_y, test_y, test_y), axis = 0)
    test_y = test_y.reshape(-1)

    #valid size 50
    valid_x = train_x[550:train_x.shape[0], :, :]
    valid_x = numpy.concatenate((valid_x, valid_x[0:4,:,:]), axis = 0)# copy 4 additional sample to valid
    valid_y = train_y[550:train_y.shape[0], :]
    valid_y = numpy.concatenate((valid_y, valid_y[0:4,:]), axis = 0)# copy 4 additional sample to valid

    valid_x = numpy.concatenate((valid_x,valid_x,valid_x,valid_x,valid_x,valid_x,valid_x,valid_x,valid_x,valid_x), axis =0)
    valid_y = numpy.concatenate((valid_y,valid_y,valid_y,valid_y,valid_y,valid_y,valid_y,valid_y,valid_y,valid_y), axis =0)
    valid_y = valid_y.reshape(-1)

    #Train size 500
    train_x = train_x[0:500, :, :]
#    train_x = numpy.concatenate((train_x, train_x, train_x), axis = 0)
    train_y = train_y[0:500, :]
#    train_y = numpy.concatenate((train_y, train_y, train_y), axis = 0)
    train_y = train_y.reshape(-1)

    #x.reshape((batch_size, 1, 96, 96))
    test_x = test_x.reshape((test_x.shape[0], 96*96))
    train_x = train_x.reshape((train_x.shape[0], 96*96))
    valid_x = valid_x.reshape((valid_x.shape[0], 96*96))

    print (test_x.shape)
    print (test_y.shape)
    print (train_x.shape)
    print (train_y.shape)
    print (valid_x.shape)
    print (valid_y.shape)

    test_x, test_y = shared_dataset(test_x, test_y)
    valid_x, valid_y = shared_dataset(valid_x, valid_y)
    train_x, train_y = shared_dataset(train_x, train_y)
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval
def saveImages(data, fileName): 
    img = toimage(data, low=0, high=255, mode='P')    
    imP = img.convert('RGBA')    
    img = mpimg.pil_to_array(imP)
    imgP = img[:,:,0]    
    plt.imsave(fileName, imgP, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
    return imP
def load_mocap_style():
#    clips, labels = read_cmu_style()
#    clips, labels = read_sub_cmu_style()
    clips, labels = read_two_cmu_style()
    print(clips.shape, labels.shape)
    #Test to save clips to images forlder
    for i in range(clips.shape[0]):
        img = clips[i, :, :]
        la = labels[i]
        saveImages(img, 'images/' + str(la) + '_img' + str(i) + '.png')
#        imsave('images/img'+ str(i), img, 'PNG')    
    train_x = clips[0:100, :, :]
    train_y = labels[0:100]

    valid_x = clips[100:200, :, :]
    valid_y = labels[100:200]
    valid_x = numpy.concatenate((valid_x,valid_x), axis = 0)
    valid_y = numpy.concatenate((valid_y,valid_y), axis = 0)

    test_x = clips[200:300, :, :]
    test_y = labels[200:300]
    test_x = numpy.concatenate((test_x, test_x[0:20,:,:]), axis = 0)# copy 7 additional sample to valid
    test_y = numpy.concatenate((test_y, test_y[0:20]), axis = 0)# copy 7 additional sample to valid
    test_x = numpy.concatenate((test_x,test_x), axis = 0)
    test_y = numpy.concatenate((test_y,test_y), axis = 0)

#    train_y = train_y.reshape(-1)
#    valid_y = valid_y.reshape(-1)
#    test_y = test_y.reshape(-1)

    print(test_x.shape)
    print(test_y.shape)
    print(train_x.shape)
    test_x = test_x.reshape((test_x.shape[0], 66*66))
    train_x = train_x.reshape((train_x.shape[0], 66*66))
    valid_x = valid_x.reshape((valid_x.shape[0], 66*66))
    print (test_x.shape)
    print (test_y.shape)
    print (train_x.shape)
    print (train_y.shape)
    print (valid_x.shape)
    print (valid_y.shape)
    print (test_y)

    test_x, test_y = shared_dataset(test_x, test_y)
    valid_x, valid_y = shared_dataset(valid_x, valid_y)
    train_x, train_y = shared_dataset(train_x, train_y)
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

    return rval    

def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=100):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    #datasets = load_mocap()
    datasets = load_mocap_style()
    #datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print (type(test_set_x))
    print(type(test_set_y))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    print('n_train_batches = ', n_train_batches)
    print('n_valid_batches = ', n_valid_batches)
    print('n_test_batches = ', n_test_batches)
#    n_train_batches = 5
#    n_valid_batches = 5
#    n_test_batches = 5

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 66, 66))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 66, 66),
        filter_shape=(nkerns[0], 1, 7, 7),
        poolsize=(4, 4)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(3, 3)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=batch_size,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=4)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    #theano.printing.debugprint(test_model)

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    validation_frequency = 10
    print('validation_frequency = ', validation_frequency)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            #if iter % 100 == 0:
            print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            print('cost_ij = ', cost_ij)

            if (iter + 1) % validation_frequency == 0:
                print ('Validating: iter ', iter)

                # compute zero-one loss on validation set
                print (range(n_valid_batches))
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                print (validation_losses)
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss <= best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    print (test_losses)
                    print (len(test_losses))
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':  
    evaluate_lenet5()

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
