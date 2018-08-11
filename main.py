#!/usr/local/bin/python3.7

import math

import numpy as np

import net
import get_training_data as gtd

nnet = net.Net()
training_set, labels = gtd.get_training_data(nnet)
nnet.train(training_set, labels)
nnet.run()



