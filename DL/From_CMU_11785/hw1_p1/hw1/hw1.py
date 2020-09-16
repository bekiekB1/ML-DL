"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        
        #self.linear_layers = [Linear(input_size,output_size,weight_init_fn,bias_init_fn)] 
        self.layesSize_list = [input_size] + hiddens + [output_size]
        self.linear_layers = [Linear( inp,opt,weight_init_fn,bias_init_fn) 
                             for inp,opt in zip(self.layesSize_list[:-1],self.layesSize_list[1:])]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        
        # if self.bn:
        #     self.bn_layers = []

        if self.bn:
            self.bn_layers = []
            for i in range(self.num_bn_layers):
                #self.bn_layers.append(BatchNorm(self.layesSize_list[i+1]))
                self.bn_layers.append(BatchNorm(self.layesSize_list[i+1]))


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
       ######3.5.1 (forward)
        
        # z = self.linear_layers[0](x)
        # y_hat = self.activations[0](z)
        # return y_hat
        y_k = x
        for k in range(self.nlayers):
            z_k = self.linear_layers[k].forward(y_k)
            if self.bn:
                if k < self.num_bn_layers:
                    if self.train_mode:
                        z = self.bn_layers[k].forward(z)
                    else:
                        z = self.bn_layers[k].forward(z, eval=True)
            y_k = self.activations[k].forward(z_k)
        return y_k

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)
        if self.momentum:
            for i in range(len(self.linear_layers)):
                # update momentum
                self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
                self.linear_layers[i].momentum_b = self.momentum * self.linear_layers[i].momentum_b - self.lr * self.linear_layers[i].db
                # update weights and biases
                self.linear_layers[i].W += self.linear_layers[i].momentum_W 
                self.linear_layers[i].b += self.linear_layers[i].momentum_b
        else:
            for i in range(len(self.linear_layers)):
                # Update weights and biases here
                self.linear_layers[i].W -= (self.lr * self.linear_layers[i].dW)
                self.linear_layers[i].b -= (self.lr * self.linear_layers[i].db)
        # Do the same for batchnorm layers
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma -= (self.lr * self.bn_layers[i].dgamma)
                self.bn_layers[i].beta -= (self.lr * self.bn_layers[i].dbeta)

        

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        
        loss = self.criterion(self.activations[self.nlayers-1].state,labels)
        dL_dyhat = self.criterion.derivative()

        # Hidden layers
        dz = [] # input of activation
        dy = [] # output of activation
        dnorm = []
        dy.append(dL_dyhat)
        for i in range(self.nlayers - self.num_bn_layers):
            dz.append(dy[i] * self.activations[self.nlayers-i-1].derivative())
            dy.append(self.linear_layers[self.nlayers-i-1].backward(dz[i]))
        # Batch norm
        for i in range(self.nlayers - self.num_bn_layers, self.nlayers):
            dnorm.append(np.multiply(dy[i], self.activations[self.nlayers-i-1].derivative()))
            dz.append(self.bn_layers[self.nlayers-i-1].backward(dnorm[i-(self.nlayers-self.num_bn_layers)]))
            dy.append(self.linear_layers[self.nlayers-i-1].backward(dz[i]))

        # dyhat_dz = dL_dyhat * self.activations[0].derivative() ## delta for linear|| Also derivative of component wise fn 
        # self.linear_layers[0].backward(dyhat_dz)



    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    X_train, Y_train = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):
        #data 
        np.random.shuffle(idxs)
        x_train = X_train[idxs]
        y_train = Y_train[idxs]
        
        ##set to training 
        mlp.train()
        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):
            #zero out 
            mlp.zero_grads()
            
            #forward propagation
            y_pred_t = mlp.forward(x_train[b:b+batch_size])
            y_true_t = y_train[b:b+batch_size]

            #backward prop
            mlp.backward(y_true_t)

            #optimizer step
            mlp.step()

            #calculate the loss 
            loss = []
            for element in SoftmaxCrossEntropy().forward(y_pred_t, y_true_t):
                loss.append(element)
            training_losses[e] += sum(loss)
            
            ##count the wrong labeled data
            for i in range(y_pred_t.shape[0]):
                if np.argmax(y_pred_t[i]) != np.argmax(y_true_t[i]):
                    training_errors[e] += 1
            
        mlp.eval()
        
        for b in range(0, len(valx), batch_size):
            # Validate ...
            # 1. Zerofill derivatives after each batch
            mlp.zero_grads()
            # 2. Forward
            y_pred_v = mlp.forward(valx[b:b+batch_size])
            y_true_v = valy[b:b+batch_size]
            # 3. Calculate validation loss
            loss = []
            for element in SoftmaxCrossEntropy().forward(y_pred_v, y_true_v):
                loss.append(element)
            validation_losses[e] += sum(loss)
            # 4. Calculate validation error count
            for i in range(y_pred_v.shape[0]):
                if np.argmax(y_pred_v[i]) != np.argmax(y_true_v[i]):
                    validation_errors[e] += 1

            # Val ...

        # Accumulate data...
        training_losses[e] = training_losses[e] / trainx.shape[0]
        validation_losses[e] = validation_losses[e] / valx.shape[0]
        training_errors[e] = training_errors[e] / trainx.shape[0]
        validation_errors[e] = validation_errors[e] / valx.shape[0]

    # Cleanup ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

