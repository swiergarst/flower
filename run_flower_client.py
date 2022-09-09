from FlClient import FlClient
import flwr as fl
import threading
import logging
import time
import argparse

#TODO: seed meegeven

parser = argparse.ArgumentParser(description="run script for flower client")
parser.add_argument("--cid", type=int, default=None,help="client id")
parser.add_argument("--dset", type=str, default="MNIST_2class", help = "dataset to be run")
parser.add_argument("--ci", type=bool, default=False, help="whether the class imbalance variant has to be selected")
parser.add_argument("--si", type=bool, default=False, help="whether the sample imbalance variant has to be selected")
parser.add_argument("--lr", type=float,default=0.5, help="learning rate")
parser.add_argument("--mc", type=str, default="FNN", help="model choice")
parser.add_argument("--lepo", type=int, default=1, help="amount of local epochs per round")
parser.add_argument("--lbat", type=int, default=1, help="amount of local batches per epoch")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--inorm", action="store_true")
parser.add_argument("--no-inorm", dest='inorm', action='store_false')

args = parser.parse_args()

print("inorm:", args.inorm)
#print("ci:", args.ci)
fl.client.start_numpy_client("[::]:8080", client=FlClient(args.cid,args.lr, args.dset, args.mc, args.lepo, args.lbat, args.seed, init_norm=args.inorm))
