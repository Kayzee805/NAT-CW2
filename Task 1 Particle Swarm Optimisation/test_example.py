# -*- coding: utf-8 -*-

import functools
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection

import pso
import ann


def dim_weights(shape):
    dim = 0
    for i in range(len(shape)-1):
        dim = dim + (shape[i] + 1) * shape[i+1]
    return dim

def weights_to_vector(weights):
    w = np.asarray([])
    for i in range(len(weights)):
        v = weights[i].flatten()
        w = np.append(w, v)
    return w

def vector_to_weights(vector, shape):
    weights = []
    idx = 0
    for i in range(len(shape)-1):
        r = shape[i+1]
        c = shape[i] + 1
        idx_min = idx
        idx_max = idx + r*c
        W = vector[idx_min:idx_max].reshape(r,c)
        weights.append(W)
    return weights

def eval_neural_network(weights, shape, X, y):
    mse = np.asarray([])
    for w in weights:
        weights = vector_to_weights(w, shape)
        nn = ann.MultiLayerPerceptron(shape, weights=weights)
        y_pred = nn.run(X)
        mse = np.append(mse, sklearn.metrics.mean_squared_error(np.atleast_2d(y), y_pred))
    return mse

def print_best_particle(best_particle):
    print("New best particle found at iteration #{i} with mean squared error: {score}".format(i=best_particle[0], score=best_particle[1]))



def sine(X):
    return np.sin(X)
#loading Starts here

def sq(X):
    return np.power(X,2)
num_classes = 2
f = open("two_spirals.dat","r")
data = np.loadtxt(f)

X = data[:,0:2]+sine(data[:,0:2])+sq(data[:,0:2])
#X = data[:,0:2]+sine(data[:,0:2])
#X = data[:,0:2]
y = data[:,2]
y = y.astype(int)
train_test_split = int(0.5*(len(X)))
#print(train_test_split)
X_train,X_test = X[:train_test_split],X[train_test_split:]
y_train,y_test = y[:train_test_split],y[train_test_split:]


# # Load MNIST digits from sklearn

# num_classes = 10
# mnist = sklearn.datasets.load_digits(num_classes)
# X, X_test, y, y_test = sklearn.model_selection.train_test_split(mnist.data, mnist.target)

num_inputs = X.shape[1]
print("Number of inputs = ",num_inputs)
y_true = np.zeros((len(y), num_classes))
for i in range(len(y)):
    y_true[i, y[i]] = 1

y_test_true = np.zeros((len(y_test), num_classes))
for i in range(len(y_test)):
    y_test_true[i, y_test[i]] = 1

# Set up
shape = (num_inputs, 6, num_classes)

cost_func = functools.partial(eval_neural_network, shape=shape, X=X, y=y_true.T)

swarm = pso.ParticleSwarm(cost_func, num_dimensions=dim_weights(shape), num_particles=30,chi=0.72984 ,phi_p=2.05,phi_g=2.05)

# Train...
i = 0
best_scores = [(i, swarm.best_score)]
print_best_particle(best_scores[-1])
while swarm.best_score>1e-6 and i<1000:
    swarm._update()
    i = i+1
    #print("index = ",i)
    if i%100==0 and swarm.best_score < best_scores[-1][1]:
        best_scores.append((i, swarm.best_score))
        print_best_particle(best_scores[-1])

# Test...
best_weights = vector_to_weights(swarm.g, shape)
best_nn = ann.MultiLayerPerceptron(shape, weights=best_weights)
y_test_pred = np.round(best_nn.run(X_test))

#print(y_test_true)d
#print(y_test_pred.T)
print("Length match y true = ? ",len(y_test_true))
print("Length for y_pred = ",len(y_test_pred.T))
print(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))




#just xy error = 0.22938741634873416
#with sinx sin y error = 0.2325498216710004
#with sin and sq error = 0.21790465807135315