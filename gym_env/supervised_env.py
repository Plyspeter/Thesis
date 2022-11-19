import math
from gym_env.supervised_dataset_loader import SupervisedDatasetLoader
from network.network_interface import NetworkInterface
from sklearn.metrics import mean_squared_error

class SupervisedEnv:

    def __init__(self, env_name):
        loader = SupervisedDatasetLoader(env_name)
        self.__data = loader.data
        self.__target = loader.target
        self.__data_val = loader.data_val
        self.__target_val = loader.target_val
        self.__input_size = loader.input_size
        self.__output_size = loader.output_size

    def run_network(self, network : NetworkInterface) -> float:
        score = len(self.__data)

        for i in range(len(self.__data)):
            results = network.run(self.__data[i], self.__output_size)

            expected = [0] * self.__output_size
            expected[self.__target[i]] = 1

            score -= mean_squared_error(expected, results)

        return score

    def test_accuarcy(self, network : NetworkInterface) -> float:
        score = 0

        for i in range(len(self.__data)):
            results = network.run(self.__data[i], self.__output_size)
            
            if self.__target[i] == results.index(max(results)):
                score += 1

        return score / len(self.__data)

    def test_validation_accuarcy(self, network : NetworkInterface) -> float:
        score = 0

        for i in range(len(self.__data_val)):
            results = network.run(self.__data_val[i], self.__output_size)

            if self.__target_val[i] == results.index(max(results)):
                score += 1

        return score / len(self.__data_val)

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size
