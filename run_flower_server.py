#from FlClient import FlClient
import flwr as fl
import threading
import logging
import time
import argparse

parser = argparse.ArgumentParser(description="run script for flower experiments")

parser.add_argument('--nro',type=int,default=100, help="number of rounds")
parser.add_argument('--strat',type=int,default=0, help="strategy to be used")
parser.add_argument('--nc',type=int,default=2, help="number of clients")
args = parser.parse_args()

print("running flower server with parameters:")
print("number of rounds:", args.nro)
print("number of clients:", args.nc)
print("strategy:", args.strat)

#TODO
def get_strat(strat_int):
    if strat_int == 0:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit = 1,
            fraction_eval = 1,
            min_fit_clients = args.nc,
            min_available_clients = args.nc,
        )

if args.strat == None:
    raise(ValueError("unknown server strategy"))
else:
    strategy = get_strat(args.strat)

fl.server.start_server(config={"num_rounds": args.nro}, strategy=strategy)