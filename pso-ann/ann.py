# -*- coding: utf-8 -*-

import numpy as np
import scipy.special
#np.random.seed(123)

class MultiLayerPerceptron:
    #np.random.seed(4512)

    def __init__(self, shape, weights=None):
        #print(np.random.get_state()[1][0])

        self.shape = shape
        self.num_layers = len(shape)
        if weights is None:
            self.weights = []
            for i in range(self.num_layers-1):
                W = np.random.uniform(size=(self.shape[i+1], self.shape[i] + 1))
                self.weights.append(W)
        else:
            self.weights = weights


    def run(self, data,activationLayer):

        layer = data.T
        for i in range(self.num_layers-1):
            prev_layer = np.insert(layer, 0, 1, axis=0)
            o = np.dot(self.weights[i], prev_layer)
            # sigmoid
            if activationLayer=="sigmoid":

                layer = scipy.special.expit(o)
            elif activationLayer=="tanh":
                layer = np.tanh(o)
            elif activationLayer=="relu":
                layer = np.maximum(0,o)
            else:
                layer = o
            
            #print(o)
            #tanh
            #layer = np.tanh(o)
     
        return layer
