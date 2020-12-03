# PSO, GA and GP for Neural Network Optimisation (Gradient-Free Methods)
Natural Computing 2020/2021: Assignment 2

## Abstract
We explore optimisation of Neural Networks with differing amounts of hidden neurons and layers using gradient-free algorithms. We see that in general, classical optimisation methods reminiscent of Stochastic Gradient Descent tend to perform better than gradient-free methods such as Particle Swarm Optimisation, Genetic Algorithms, and Genetic Programming. We see that although optima reached by gradient-free methods are not as 'good' as classical gradient-based methods, they are nevertheless good enough. That is, we reach very small errors on test sets generalising to unseen data. The main difference being time to converge.

## Overview
We will explore gradient-free optimisation methods for optimising Neural Networks (NN). Including Particle Swarm Optimisation (PSO), Genetic Algorithms (GA), and Genetic Programming (GP). 

Gradient-free algorithms will then be compared to baseline NNs optimised using standard gradient-based methods (view Appendix 1 for a description of baseline). Including the Stochastic gradient descent (SGD) classical way to optimise NNs, Adam an optimiser similar to SGD, and Limited memory Broyden Fletcher Goldfarb Shanno (LBFGS) quasi-Newton method for optimisation.

## Install dependencies

```
pip install -r requirements.txt
```

## Task 1: Particle Swarm Optimisation

****

Run the notebook "Task1.ipynb" under the directory "Task 1 Particle swarm optimisation" and follow the instructions

## Task 2: Genetic Algorithm

****

Run the notebook "GeneticAlgorithm.ipynb" under the directory "Task 2 Genetic Algorithms" and follow the instructions

## Task 3: Genetic Programming

****

Run the notebook "GeneticProgramming.ipynb" under the directory "Task 3 Genetic Programming" and follow the instructions

