import domain
import numpy as np

class NeatGameBipedalWalker:
    def add(self):
        domain.games['BipedalWalker'] = domain.config.Game(
            env_name='BipedalWalker-v3',
            time_factor=0,
            actionSelect='all',
            input_size=24,
            output_size=4,
            layers=[1],
            i_act=np.full(24, 1),
            h_act=np.full(1, 1),
            o_act=np.full(4, 1),
            weightCap=4.0,
            noise_bias=0.0,
            output_noise=[False],
            max_episode_length=2000,
            in_out_labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
        )