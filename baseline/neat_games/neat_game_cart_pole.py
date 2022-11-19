import domain
import numpy as np

class NeatGameCartPole:
    def add(self):
        domain.games['CartPole'] = domain.config.Game(
            env_name='CartPole-v1',
            time_factor=0,
            actionSelect='prob',
            input_size=4,
            output_size=2,
            layers=[1],
            i_act=np.full(4, 1),
            h_act=np.full(1, 1),
            o_act=np.full(2, 1),
            weightCap=4.0,
            noise_bias=0.0,
            output_noise=[False, False, False],
            max_episode_length=500,
            in_out_labels=['Cart_Position', 'Cart_Velocity', 'Pole_Angle', 'Pole_Angular_Velocity', 'Force_left', 'Force_right']
        )