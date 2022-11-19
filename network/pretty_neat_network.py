import prettyNEAT
import numpy as np
from network.network_interface import NetworkInterface

  #case 1  -- Linear
  #case 2  -- Unsigned Step Function
  #case 3  -- Sin
  #case 4  -- Gausian with mean 0 and sigma 1
  #case 5  -- Hyperbolic Tangent [tanh] (signed)
  #case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
  #case 7  -- Inverse
  #case 8  -- Absolute Value
  #case 9  -- Relu
  #case 10 -- Cosine
  #case 11 -- Squared (not in use because of overflow exception)

class PrettyNeatNetwork(NetworkInterface):
    
    def build(self, graph: 'dict[int, list[int]]', inputs: 'list[int]', hiddens: 'list[int]', outputs: 'list[int]', acts: 'dict[int, int]', bias: 'dict[int, int]', weights: 'dict[(int, int), float]', neat_config: dict) -> None:
        self.nInput = len(inputs)
        nHidden = len(hiddens)
        self.nOutput = len(outputs)
        nodeId = [0] + inputs + outputs + hiddens
        node = np.empty((3,len(nodeId)))
        node[0,:] = nodeId

        # Node types: [1:input, 2:output, 3:hidden, 4:bias]
        node[1,0] = 4 # Bias
        node[1,1:self.nInput+1] = 1 # Input Nodes
        node[1,(self.nInput+1):(self.nInput+self.nOutput+1)] = 2 # Output Nodes
        node[1,(self.nInput+self.nOutput+1):(self.nInput+self.nOutput+nHidden+1)] = 3 #Hidden Nodes
        
        node[2, 0] = 1

        for i in range(1, len(nodeId)):
            node[2,i] = acts[node[0,i]] + 1 #change from range 0-9 to 1-10
    
        connId = 0
        conn = [[] for _ in range(5)]
        for id in hiddens + outputs: #Bias node
            conn[0].append(connId)                # Connection Id
            conn[1].append(0)                     # Source Node
            conn[2].append(id)                    # Destination Node
            conn[3].append(PrettyNeatNetwork.range_convert(bias[id], neat_config) * neat_config["bias_multiplier"])    # Weight
            conn[4].append(True)                  # Enabled
            connId += 1

        for id in graph:
            for otherId in graph[id]:                               #Other nodes
                conn[0].append(connId)                              # Connection Id
                conn[1].append(id)                                  # Source Node
                conn[2].append(otherId)                             # Destination Node
                conn[3].append(PrettyNeatNetwork.range_convert(weights[id, otherId], neat_config) * neat_config["weight_multiplier"])    # Weight
                conn[4].append(True)                                # Enabled
                connId += 1
        
        self.network = prettyNEAT.ind.Ind(conn, node)
        self.network.express()
        self.wMat = self.network.wMat
        self.aVec = self.network.aVec
    
    @staticmethod
    def range_convert(old_value, neat_config):
            if neat_config.get("parameter_range") is None:
                return (old_value - 0.5) * 2 
            old_max = neat_config["parameter_range"]["max"]
            old_min = neat_config["parameter_range"]["min"]
            old_range = old_max - old_min
            if old_range == 0:
                return -1
            return (old_value - old_min) * 2 / old_range + (-1)
            
    def run(self, input: 'list[float]', nOutput : int) -> 'list[float]':
        res = prettyNEAT.ann.act(self.wMat, self.aVec, len(input), nOutput, input)
        return res.tolist()[0]

    def save(self, path) -> None:
        prettyNEAT.ann.exportNet(path, self.wMat, self.aVec)
        
    @staticmethod
    def load(path) -> 'PrettyNeatNetwork':
        net = PrettyNeatNetwork()
        wVec, aVec, _ = prettyNEAT.ann.importNet(path)
        if np.ndim(wVec) < 2:
            nNodes = int(np.sqrt(np.shape(wVec)[0]))
            wMat = np.reshape(wVec, (nNodes, nNodes))
        else:
            wMat = wVec
        wMat[np.isnan(wMat)]=0
        
        net.aVec = aVec
        net.wMat = wMat
        return net