#from FlClient import FlClient
import flwr as fl
import threading
import logging
import time
import argparse
import numpy as np
from strategies import SaveAccStrategy, VanillaStrategy
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from fed_common.config_functions import get_save_str
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="run script for flower experiments")

parser.add_argument('--nro',type=int,default=100, help="number of rounds")
parser.add_argument('--strat',type=int,default=0, help="strategy to be used")
parser.add_argument('--nc',type=int,default=2, help="number of clients")
parser.add_argument('--sstr', type=str, default="../datafiles/default.npy", help="save string")
args = parser.parse_args()

print("running flower server with parameters:")
print("number of rounds:", args.nro)
print("number of clients:", args.nc)
print("strategy:", args.strat)


frac_fit = 1
frac_eval = 1

def get_strat(strat_int):

    if strat_int == 0:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit = frac_fit,
            fraction_eval = frac_eval,
            min_fit_clients = args.nc,
            min_available_clients = args.nc
        )
    elif strat_int == 1:
        strategy = SaveAccStrategy(            
            fraction_fit = frac_fit,
            fraction_eval = frac_eval,
            min_fit_clients = args.nc,
            min_available_clients = args.nc)
    elif strat_int == 2:
        strategy = VanillaStrategy(
            fraction_fit = frac_fit,
            fraction_eval = frac_eval,
            min_fit_clients = args.nc,
            min_available_clients = args.nc
        )
    return strategy



if args.strat == None:
    raise(ValueError("unknown server strategy"))
else:
    strategy = get_strat(args.strat)

hist = fl.server.start_server(config={"num_rounds": args.nro}, strategy=strategy)


array = np.zeros((args.nc,args.nro))

# check if files exist
for r in range(1, args.nro+1):
    file = "round" + str(r) + "accuracies.npy"
    if os.path.exists(file):
        array[:,r-1] = np.load(file)
        os.remove(file)



# take the mean from all clients
mean_acc = np.mean(array, axis = 0)

#save into a file
with open(args.sstr, "wb") as f:
    np.save(f, mean_acc)


#print("final array: ", mean_acc)
plt.plot(np.arange(args.nro), mean_acc)
#plt.plot(np.arange(args.nro), array.T)
plt.grid(True)
#plt.show()