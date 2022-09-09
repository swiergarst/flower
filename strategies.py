import flwr as fl
from typing import List, Optional, Tuple, Dict
import numpy as np
from flwr.common.parameter import parameters_to_weights, weights_to_parameters


class SaveAccStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self,
     rnd: int,
     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
     failures: List[BaseException],
     ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""


        #print("aggregating evaluations..")
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] for _, r in results]
        examples = [r.num_examples for _, r in results]


        print("saving local accuracies to temporary file")
        round_file = "round" + str(rnd) + "accuracies.npy"
        with open(round_file , "wb") as f:
            np.save(f, accuracies)

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

class VanillaStrategy(fl.server.strategy.FedAvg):

    def aggregate_evaluate(self,
     rnd: int,
     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
     failures: List[BaseException],
     ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""


        #print("aggregating evaluations..")
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] for _, r in results]
        examples = [r.num_examples for _, r in results]


        print("saving local accuracies to temporary file")
        round_file = "round" + str(rnd) + "accuracies.npy"
        with open(round_file , "wb") as f:
            np.save(f, accuracies)

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


    def aggregate_fit(self, rnd, results, failures):
        #print("hi mum!")

        parameters = np.array([ parameters_to_weights(res.parameters) for _, res in results], dtype=object)
        #print (parameters[0,:])

        agg_params = parameters[0,:]
        
        # this does an unweighted average for now
        for l_i in range(len(parameters[0,:])): # loop over all layers
            for p_i in range(1, parameters.shape[0]): #loop over all clients
                agg_params[l_i] += parameters[p_i,:][l_i]
            agg_params[l_i] /= parameters.shape[0]

        
        return weights_to_parameters(agg_params), {}
'''
class OwnFedAvg(fl.server.strategy.Strategy):
    def initialize_parameters(self, client_manager):
        pass
        # Your implementation here
        

    def configure_fit(self, rnd, parameters, client_manager):
        # Your implementation here
        pass
    def aggregate_fit(self, rnd, results, failures):
        # Your implementation here
        pass
    def configure_evaluate(self, rnd, parameters, client_manager):
        # Your implementation here
        pass
    def aggregate_evaluate(self, rnd, results, failures):
        # Your implementation here
        pass
    def evaluate(self, parameters):
        # Your implementation here
'''