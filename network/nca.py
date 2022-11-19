import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)

def linear(x):
    return x

act_dict = {"linear":linear, "sigmoid":torch.sigmoid, "relu":F.relu, "tanh":torch.tanh}

class NCA(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_layer_sizes: 'list[int]', acts: 'list[int]'):
        super(NCA, self).__init__()
        if len(hidden_layer_sizes) == 0:
            self.layers = [nn.Linear(input_size, output_size)]
            self.hidden_acts = [act_dict[acts[0]]]
        else:
            self.layers = [nn.Linear(input_size, hidden_layer_sizes[0])]

            self.hidden_acts = list(map(lambda k: act_dict[k], acts))
            
            for i in range(1, len(hidden_layer_sizes)):
                self.layers.append(nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))

            self.layers.append(nn.Linear(hidden_layer_sizes[-1], output_size))

    def forward(self, x: 'list[float]') -> 'list[float]':
        x = torch.tensor(x)
        #hedwig.debug("Packing in tensor")
        for id, layer in enumerate(self.layers):
            #hedwig.debug(f"Handling layer number {id}")
            x = self.hidden_acts[id](layer(x))
        #hedwig.debug(f"returning")
        return x.detach().numpy()

    def get_num_of_layers(self) -> int: 
        return len(self.layers)
    
    def get_num_of_neurons(self, layer_id: int) -> int:
        return len(self.layers[layer_id].weight)

    def get_num_of_weights(self, layer_id: int, neuron_id: int) -> int:
        return len(self.layers[layer_id].weight[neuron_id])

    def get_num_of_biases(self, layer_id: int) -> int:
        return len(self.layers[layer_id].bias)

    def get_num_of_parameters(self):
        num_of_params = 0
        for layer in range(len(self.layers)):
            for neuron in range(len(self.layers[layer].weight)):
                num_of_params += len(self.layers[layer].weight[neuron])
            num_of_params += len(self.layers[layer].bias)
        return num_of_params

    def add_to_weight(self, layer_id: int, neuron_id: int, weight_id: int, val: float) -> None:
        with torch.no_grad():
            self.layers[layer_id].weight[neuron_id][weight_id] += val

    def add_to_bias(self, layer_id: int, neuron_id: int, val: float) -> None:
        with torch.no_grad():
            self.layers[layer_id].bias[neuron_id] += val

    def set_weight(self, layer_id: int, neuron_id: int, weight_id: int, val: float) -> None:
        with torch.no_grad():
            self.layers[layer_id].weight[neuron_id][weight_id] = val

    def set_bias(self, layer_id: int, neuron_id: int, val: float) -> None:
        with torch.no_grad():
            self.layers[layer_id].bias[neuron_id] = val

    def save(self, path):
        torch.save(self, path)
        
    @staticmethod
    def load(path) -> 'NCA':
        return torch.load(path)