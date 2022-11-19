import gym
import time
import hedwig
from network.nca import NCA
from network.network_interface import NetworkInterface
from gym import spaces
gym.logger.set_level(40) #Removes warnings

class GymEnv:

    def __init__(self, gym_env_name) -> None:
        self.env = gym.make(gym_env_name)
        self.name = gym_env_name
        self.render = False
        
        if type(self.env.observation_space) is spaces.Box:
            self.__input_size = self.env.observation_space.shape[-1]
        elif type(self.env.observation_space) is spaces.Discrete:
            self.__input_size = self.env.observation_space.n
        else:
            hedwig.warning(f"input_size unknown type {type(self.env.observation_space)}")

        if type(self.env.action_space) is spaces.Box:
            self.__output_size = self.env.action_space.shape[-1]
        elif type(self.env.action_space) is spaces.Discrete:
            self.__output_size = self.env.action_space.n
        else:
            hedwig.warning(f"output_size unknown type {type(self.env.action_space)}")

    def run_network(self, network : NetworkInterface) -> int:
        #hedwig.debug("Running network")
        observation = self.env.reset()
        score = 0
        while True:
            results = network.run(observation, self.__output_size)

            if self.__output_size == 1:        
                action = results
            else:
                best = max(results)
                action = results.index(best)
            
            if self.name == "Pendulum-v1":
                action[0] *= 2
            res = self.env.step(results if self.name == "BipedalWalker-v3" else action)

            observation = res[0]
            done = res[2]

            #self.env.render()
                
            if done:
                break

            score += res[1]

        #hedwig.debug("Network finished run")
        return score

    def run_nn_network(self, network : NCA) -> float:
        observation = self.env.reset()
        score = 0

        while True:
            results = list(network.forward(observation))
            action = results if self.__output_size == 1 else results.index(max(results))
            
            if self.name == "Pendulum-v1":
                action[0] *= 2

            res = self.env.step(results if self.name == "BipedalWalker-v3" else action)

            observation = res[0]
            done = res[2]

            if self.render:
                self.env.render()
                time.sleep(.05)
            
            score += res[1]

            if done:
                break

        return score

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size
