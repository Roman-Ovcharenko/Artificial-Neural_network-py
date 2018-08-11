import math

import numpy as np

###############################################################################
# Define neural network
###############################################################################
class Net(object):

# Fitting threshold for relative loss function 
    _eps_loss_func_change = 1.0e-2 
    _eps_loss_func = 1.0e-2
# Maximum number of iteration
    _iter_max = 1000000
# Default learning rate
    _def_learning_rate = 1.0
# Default decay for weights
    _def_weight_decay = 0.0
# Default number of layers
    _def_num_layers = 3

###############################################################################
# num_layers - number of layers for neural network
# num_neurons_per_layer - number of neurons per each layer
# weights, bias - fitting parameters of the neural network
# learning_rate - how fast it learns
# weight_decay - keeps fitting coefficients small
###############################################################################
    def __init__(self):
        print("\nDefine topology of the neural network")
        self.num_layers = self._get_num_layers()
        self.num_neurons_per_layer = self._get_num_neurons_per_layer() 
        self.weights, self.bias = self._gen_initial_weights()
        print("\nDefine learning properties")
        self.learning_rate = self._get_learning_rate()
        self.weight_decay = self._get_weight_decay()
        return None

###############################################################################
# Read decay factor for weights
###############################################################################
    def _get_weight_decay(self):
        while True:
            decay = float(input("Weight decay (default {:5.3f}): "
                                .format(self._def_weight_decay)) 
                          or self._def_weight_decay)  
            if (decay < 0.0) or (decay > 1.0):
                print("Weight decay should be between 0 and 1")
                continue
            break
        return decay

###############################################################################
# Read learning rate
###############################################################################
    def _get_learning_rate(self):
        while True:
            rate = float(input("Learning rate (default {:5.3f}): "
                              .format(self._def_learning_rate)) 
                         or self._def_learning_rate)
            if (rate < 0.0) or (rate > 1.0):
                print("Learning rate should be between 0 and 1")
                continue
            break
        return rate

###############################################################################
# Read number of layers
###############################################################################
    def _get_num_layers(self):
        while True:
            num_layers = int(input("Number of layers (default {:2d}): "
                                  .format(self._def_num_layers)) 
                             or self._def_num_layers)
            if num_layers <= 1:
                print("Number of layers should be >1")
                continue
            break
        return num_layers

###############################################################################
# Read number of neurons per each layer
###############################################################################
    def _get_num_neurons_per_layer(self):
        while True:
            str_ = input("Number of neurons per a layer: ") 
            lst = str_.split()
            if len(lst) != self.num_layers:
                print("Wrong number of layers")
                continue
            num_neurons_per_layer = [int(i) for i in lst]
            for i in num_neurons_per_layer:
                if i <= 0:
                    print("Number of neurons should be >0")
                    continue
            break
        return num_neurons_per_layer

###############################################################################
# Choose initial weights and bias randomly
###############################################################################
    def _gen_initial_weights(self):
        weights = []
        bias = []
        for ilay in range(self.num_layers-1):
                weights.append(np.random.random(
                    (self.num_neurons_per_layer[ilay+1],
                     self.num_neurons_per_layer[ilay])))
                bias.append(np.random.random(
                    self.num_neurons_per_layer[ilay+1]))
        return weights, bias

###############################################################################
# Train neural network using the train data set and labels
###############################################################################
    def train(self, training_set, labels):
        print()
        print("Training...")
        num_exampl = len(training_set)
        if num_exampl != len(labels):
            raise Exception("Training set size is not consistent with labels")
        loss_func_prev = 0.0
        iter_ = 0
        while True:
            grad_matrix_sum = self._init_grad_matrix()
            grad_bias_sum = self._init_grad_bias()
            decay_term = self._calc_decay_term()
            loss_func = 0.0
            for iex in range(num_exampl):
                input_ = training_set[iex]
                label = labels[iex]
                activations = self._forward_propagation(input_)
                loss_func_per_exampl = 0.5 * np.linalg.norm(
                    activations[self.num_layers-1] - label)**2 
                loss_func = (loss_func + loss_func_per_exampl / num_exampl 
                             + decay_term)
                errors = self._back_propagation(label, activations)
                self._addup_grad_matrix(grad_matrix_sum, grad_bias_sum,
                                        errors, activations, num_exampl)
            loss_func_change = self._calc_loss_func_change(iter_, 
                                                          loss_func,
                                                          loss_func_prev)
#            print(" {:7d}        {:10.6f}        {:10.6f}"
#                  .format(iter_, loss_func_change, loss_func))
            self._printout_loss_func(iter_, loss_func, "loss_func.dat")
            self._printout_loss_func(iter_, loss_func_change, 
                                     "loss_func_change.dat")
            if self._if_converged(loss_func_change, loss_func, iter_): 
                break
            else:
                loss_func_prev = loss_func
                self._update_weights_bias(num_exampl, grad_matrix_sum,
                                          grad_bias_sum)
                iter_ = iter_ + 1

        print("Finished in {} iterations".format(iter_))
        print("Loss function vs iterations is printed out to the 'loss_func.dat' file")
        print("Loss function change vs iterations is printed out to the 'loss_func_change.dat'"
              "file\n")
        return None

###############################################################################
# Check if minimization is converged
###############################################################################
    def _if_converged(self, loss_func_change, loss_func, iter_):
        if ((loss_func_change < self._eps_loss_func_change) 
            and (loss_func < self._eps_loss_func)):
            print("Specified accuracy is reached")
            return True
        elif iter_ > self._iter_max:
            print("The maximum number of iterations is reached")
            print("Current loss function: {:10.6f}".format(loss_func))
            print("Current loss function change: {:10.6f}".format(loss_func_change))
            return True
        else:
            return False

###############################################################################
# print the (iter_, loss function) to the file
###############################################################################
    def _printout_loss_func(self, iter_, loss_func, file_):
        if iter_ == 0:
            f = open(file_, "w")
            f.close()
        else:
            with open(file_, "a") as f:
                f.write("{:6d}   {:8.3f}\n".format(iter_, loss_func))
        return None
###############################################################################
# Calculate relative change in the loss function as compared with the previous
# iteration
###############################################################################
    def _calc_loss_func_change(self, iter_, loss_func, loss_func_prev):
        change = abs(loss_func - loss_func_prev) / self._eps_loss_func
        return change

###############################################################################
# Add up gradients of the loss function 
###############################################################################
    def _addup_grad_matrix(self, grad_matrix_sum, grad_bias_sum, 
                          errors, activations, num_exampl):
        for ilay in range(self.num_layers-1):
            grad_matrix_sum[ilay] = (grad_matrix_sum[ilay] 
                                     + self._calc_grad_matrix(errors[ilay+1],
                                                             activations[ilay]) 
                                     / num_exampl 
                                     + self.weight_decay * self.weights[ilay])
            grad_bias_sum[ilay] = (grad_bias_sum[ilay] 
                                   + self._calc_grad_bias(errors[ilay+1]) 
                                   / num_exampl)
        return None

###############################################################################
# Calculate partial derivatives of the loss function with respect to a single
# example and single layer 
###############################################################################
    def _calc_grad_bias(self, error):
        size = error.shape[0]
        grad_bias = np.zeros((size))
        for i in range(size): 
            grad_bias[i] = error[i] 
        return grad_bias

###############################################################################
# Calculate the weight decay term for the loss function
###############################################################################
    def _calc_decay_term(self):
        sum_ = 0.0
        for ilay in range(self.num_layers-1):
            sum_ = sum_ + np.linalg.norm(self.weights[ilay])**2
        return self.weight_decay * sum_ / 2.0

###############################################################################
# Initiate the gradient with zeros
###############################################################################
    def _init_grad_bias(self):
        grad_bias = []
        for ilay in range(self.num_layers-1):
            grad_bias.append(np.zeros(self.num_neurons_per_layer[ilay+1]))
        return grad_bias

###############################################################################
# Initiate the gradient matrix with zeros
###############################################################################
    def _init_grad_matrix(self):
        grad_matrix = []
        for ilay in range(self.num_layers-1):
            grad_matrix.append(np.zeros((self.num_neurons_per_layer[ilay+1], 
                                         self.num_neurons_per_layer[ilay])))
        return grad_matrix

###############################################################################
# Run the trained neural network with unseen user's data
###############################################################################
    def run(self):
        line = ""
        while line != "exit":
            line = input("Type trial vector or exit: ") 
            if "exit" in line:
                break
            lst = line.split()
            if len(lst) != self.num_neurons_per_layer[0]:
                print("Size of the input vector is wrong")
                continue
            in_vec = np.array([float(i) for i in lst])
            out_vec = self.apply(in_vec)
            print("Lable: {}".format(out_vec))
            print()
        print("Work with neural network is finished")
        return None

###############################################################################
# Pass a trial vector forward through the trained network
###############################################################################
    def apply(self, input_vec):
        activations = self._forward_propagation(input_vec)
        label = activations[self.num_layers-1]
        return label

###############################################################################
# Calculate partial derivatives of the loss function with respect to a single
# example and single layer 
###############################################################################
    def _calc_grad_matrix(self, error, activation):
        num_row = error.shape[0]
        num_col = activation.shape[0]
        grad_matrix = np.zeros((num_row, num_col))
        for i in range(num_row): 
            for j in range(num_col):
                grad_matrix[i, j] = activation[j] * error[i] 
        return grad_matrix

###############################################################################
# Update weights and bias after each iteration ###############################################################################
    def _update_weights_bias(self, num_exampl, grad_matrix_sum, grad_bias_sum):
        for ilay in range(self.num_layers-1):
            self.weights[ilay] = (self.weights[ilay] - self.learning_rate 
                                  * grad_matrix_sum[ilay])
            self.bias[ilay] = (self.bias[ilay] - self.learning_rate
                               * grad_bias_sum[ilay])
        return None

###############################################################################
# Back propagation to get errors produced by each layer 
###############################################################################
    def _back_propagation(self, label, activations):
        error = (-(label - activations[self.num_layers-1]) 
                 * self._activ_func_deriv(activations[self.num_layers-1]))
        errors = []
        errors.append(error)
        error_prev = error
        for ilay in range(self.num_layers-2, -1, -1):
            error = (np.matmul(self.weights[ilay].T, error_prev) 
                 * self._activ_func_deriv(activations[ilay])) 
            errors.insert(0, error)
            error_prev = error
        return errors

###############################################################################
# Calculate derivative of the activation function
###############################################################################
    def _activ_func_deriv(self, activations):
        try:
            dim = activations.shape[0]
        except AttributeError:
            res = self._logistic_func_deriv(activations)
            return res
        res = np.zeros((dim))
        for i in range(dim):
            res[i] = self._logistic_func_deriv(activations[i])
        return res

###############################################################################
# Calculate analytical derivative of the logistic function  
###############################################################################
    def _logistic_func_deriv(self, x):
        return x * ( 1 - x )

###############################################################################
# Perform the feedforward propagation to obtain activations for a single 
# example
###############################################################################
    def _forward_propagation(self, input_):
        activations = []
        activations.append(self._activ_func(input_))
        for ilay in range(1, self.num_layers):
            input_ = (np.dot(self.weights[ilay-1], activations[ilay-1]) 
                      + self.bias[ilay-1])
            activations.append(self._activ_func(input_))
        return activations

###############################################################################
# Define the activation function  
###############################################################################
    def _activ_func(self, input_):
        try:
            dim = input_.shape[0]
        except IndexError:
            res = self._logistic_func(input_)
            return res
        res = np.zeros((dim))
        for i in range(dim):
            res[i] = self._logistic_func(input_[i])
        return res

###############################################################################
# Define the logistic function  
###############################################################################
    def _logistic_func(self, x):
        return 1/(1 + math.exp(-x))
