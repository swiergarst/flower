from FlClient import FlClient
import flwr as fl
import threading
import logging
import time


strategy = fl.server.strategy.FedAvg(
    fraction_fit = 1,
    fraction_eval = 1,
    min_fit_clients = 2,
    min_available_clients = 2
)
def thread_fn_client(id):
    fl.client.start_numpy_client("[::]:8080", client=FlClient(id,0.05))

def thread_fn_server():
    fl.server.start_server(config={"num_rounds": 100}, strategy=strategy)

num_clients = 10

th_server = threading.Thread(target=thread_fn_server)

th_server.start()

time.sleep(10)

th_client_0 = threading.Thread(target=thread_fn_client, args=(0,))
th_client_1 = threading.Thread(target=thread_fn_client, args=(1,))


th_client_0.start()
time.sleep(5)
th_client_1.start()

'''
for i in range(num_clients):
    th = threading.Thread(target=thread_fn_client, args=(i,))
    th.start()
    time.sleep(5)
'''