import flwr as fl
from FlClient import FlClient

fl.client.start_numpy_client("[::]:8080", client=FlClient(6,0.05))