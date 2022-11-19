import domain
import numpy as np

class NeatGamePendulum:
    def add(self):
        domain.games['Pendulum'] = domain.config.Game(
            env_name='Pendulum-v1',
            time_factor=0,
            actionSelect='all',
            input_size=3,
            output_size=1,
            layers=[1],
            i_act=np.full(3, 1),
            h_act=np.full(1, 1),
            o_act=np.full(1, 1),
            weightCap=4.0,
            noise_bias=0.0,
            output_noise=[False],
            max_episode_length=200,
            in_out_labels=['x', 'y', 'velocity', 'torque']
        )