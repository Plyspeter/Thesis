import domain
import numpy as np

class NeatGameAcroBot:
    def add(self):
        domain.games['Acrobot'] = domain.config.Game(
            env_name='Acrobot-v1',
            time_factor=0,
            actionSelect='prob',
            input_size=6,
            output_size=3,
            layers=[1],
            i_act=np.full(6, 1),
            h_act=np.full(1, 1),
            o_act=np.full(3, 1),
            weightCap=4.0,
            noise_bias=0.0,
            output_noise=[False, False, False],
            max_episode_length=500,
            in_out_labels=['Cos(theta1)', 'Sin(theta1)', 'Cos(theta2)', 'Sin(theta2)', 'Vel(theta1)', 'Vel(theta2)', '-1 torque', '0 torque', '1 torque']
        )