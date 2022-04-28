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

from nn_model import nn_model
from fed_common.config_functions import get_data
from fed_common.nn_common import model_common

from collections import OrderedDict

import flwr as fl



class FlClient(fl.client.NumPyClient):
    def __init__(self, client_id, lr, dataset, model_choice, lepochs, lbatches):
        super (FlClient, self).__init__()
        self.net = nn_model(dataset, model_choice).double()
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
        self.net.train(self.X_train, self.y_train, self.lr, self.lepo,self.lbatches)
        return self.get_parameters(), 10, {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.net.test(self.X_test, self.y_test)
        loss = 1-accuracy
        return float(loss), 10, {"accuracy": float(accuracy)}

