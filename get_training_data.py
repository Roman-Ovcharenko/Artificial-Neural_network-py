import numpy as np

# Default files containing training data
def_file_training_set = "training_set.dat"
def_file_labels = "labels.dat"

###############################################################################
# Get training data
###############################################################################
def get_training_data(nnet):
    print("\nDefine training data")
    training_set = get_training_set(nnet)
    labels = get_labels(nnet)
    return training_set, labels

###############################################################################
# Read training set from a file
###############################################################################
def get_training_set(nnet):
    file_training_set = (
        input("File containing training set (default training_set.dat): ") 
        or def_file_training_set)
    training_set = []
    with open(file_training_set, "r") as f:
        while True:
            line = f.readline()
            if not line.strip():
                break
            lst = line.split()
            if len(lst) != nnet.num_neurons_per_layer[0]:
                raise Exception("Training set is not size-consistent")
            training_set.append(np.array([float(i) for i in lst]))
    return training_set

###############################################################################
# Read training labels from a file
###############################################################################
def get_labels(nnet):
    file_labels = (input("File containing labels (default labels.dat): ") 
                    or def_file_labels) 
    labels = []
    with open(file_labels, "r") as f:
        while True:
            line = f.readline()
            if not line.strip():
                break
            lst = line.split()
            if len(lst) != nnet.num_neurons_per_layer[-1]:
                raise Exception("Labels are not size consistent")
            labels.append(np.array([float(i) for i in lst]))
    return labels

