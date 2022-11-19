from abc import abstractmethod

class NetworkInterface:
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'build') and 
                callable(subclass.build) and 
                hasattr(subclass, 'run') and 
                callable(subclass.run) or 
                NotImplemented)
        
    @abstractmethod
    def build(self, graph: 'dict[int, list[int]]', inputs: 'list[int]', hiddens: 'list[int]', outputs: 'list[int]') -> None:
        #Builds the network 
        raise NotImplementedError

    @abstractmethod
    def run(self, input: 'list[float]', n_output : int) -> 'list[float]':
        #Runs the network
        raise NotImplementedError
    
    