from gym_env.gym_env import GymEnv
from network.network_interface import NetworkInterface
import random

# The random baseline algorithm runs by taking random actions
# in the gym envoronment, and returns the score of this.
class Random:

    def run(self, fitness_iterations, gym_env_name) -> int:
        gym_env = GymEnv(gym_env_name)
        gym_env.render = False

        network = RandomNetwork()
        score = 0
        for _ in range(fitness_iterations):
            score += gym_env.run_network(network)
        score /= fitness_iterations
        
        return score

class RandomNetwork(NetworkInterface):

    def build(self) -> None:
        pass

    def run(self, input, output_size) -> 'list[float]':
        return [random.uniform(-1, 1) for _ in range(output_size)]