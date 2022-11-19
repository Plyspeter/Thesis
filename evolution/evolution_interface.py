from abc import abstractmethod

class EvolutionInterface:
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'set_parameters') and 
                callable(subclass.build) and 
                hasattr(subclass, 'run_evolution') and 
                callable(subclass.run) and 
                hasattr(subclass, 'get_population') and 
                callable(subclass.run) or 
                NotImplemented)

    @abstractmethod
    def set_parameters(self, **config):
        pass

    @abstractmethod
    def run_evolution(self, callback):
        pass

    @abstractmethod
    def get_population(self):
        pass
    
    