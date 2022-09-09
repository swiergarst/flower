import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from nn_model import nn_model2
from fed_common.config_functions import get_data, init_params
from fed_common.nn_common import model_common
from fed_classifiers.NN.v6_simpleNN_py.model import model
from collections import OrderedDict

import flwr as fl



class FlClient(fl.client.NumPyClient):
    def __init__(self, client_id, lr, dataset, model_choice, lepochs, lbatches, seed, init_norm=True):
        super (FlClient, self).__init__()
        torch.manual_seed(seed)
        #net = nn_model2(dataset, model_choice, None)
        net = model(dataset, model_choice, None)
        self.net = net.double()
        if init_norm:
            params = init_params(dataset, model_choice, zeros=False)
            self.net.set_params(params)
        self.X_train, self.y_train, self.X_test, self.y_test = get_data(dataset, client_id)
        self.lr = lr
        self.lepo = lepochs
        self.lbatches = lbatches
        
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)  

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()

        self.net.train(self.X_train, self.y_train,opt, crit, self.lr, self.lepo,self.lbatches, None, False, False, None)
        return self.get_parameters(), 10, {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        results =  self.net.test(self.X_test, self.y_test, None)
        #loss = 1-accuracy
        loss = results["accuracy"]
        return float(loss), 10, {"accuracy": float(results["accuracy"])}

