import domain
import numpy as np

class NeatGameMountainCar:
    def add(self):
        domain.games['MountainCar'] = domain.config.Game(
            env_name='MountainCar-v0',
            time_factor=0,
            actionSelect='prob',
            input_size=2,
            output_size=3,
            layers=[1],
            i_act=np.full(2, 1),
            h_act=np.full(1, 1),
            o_act=np.full(3, 1),
            weightCap=4.0,
            noise_bias=0.0,
            output_noise=[False, False, False],
            max_episode_length=200,
            in_out_labels=['Cart_Position', 'Cart_Velocity', 'Accelerate Left', 'No Accelerate', 'Accelerate Right']
        )