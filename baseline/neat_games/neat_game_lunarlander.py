import domain
import numpy as np

class NeatGameLunarLander:
    def add(self):
        domain.games['LunarLander'] = domain.config.Game(
            env_name='LunarLander-v2',
            time_factor=0,
            actionSelect='prob',
            input_size=8,
            output_size=4,
            layers=[1],
            i_act=np.full(8, 1),
            h_act=np.full(1, 1),
            o_act=np.full(4, 1),
            weightCap=4.0,
            noise_bias=0.0,
            output_noise=[False],
            max_episode_length=1000,
            in_out_labels=['x_coord', 'y_coord', 'x_velocity', 'y_velocity', 'angle', 'angle_velocity', 'contact1', 'contact2', 'nothing', 'left', 'main', 'right']
        )