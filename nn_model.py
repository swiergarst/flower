
import torch
import sys
import math
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from fed_common.nn_common import model_common


class nn_model(model_common):
    def __init__(self,dataset, model_choice):
        super(nn_model, self).__init__(dataset, model_choice)


    def train(self, X_train, y_train, lr, lepochs = 1, batches = 1):
        batch_size = math.floor(X_train.size()[0]/batches)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for e in range (lepochs):
            for batch in range(batches):
                X_train_batch = X_train[batch* batch_size: (batch+1) * batch_size]
                y_train_batch = y_train[batch* batch_size: (batch+1) * batch_size]
                # zero the optimizer gradients
                optimizer.zero_grad()
                #print(datapoint)
                ### forward pass, backward pass, optimizer step
                out = self.forward(X_train_batch)
                loss = criterion(out, y_train_batch)
                loss.backward()
                optimizer.step()