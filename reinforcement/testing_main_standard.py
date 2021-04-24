from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from testing_simulation_standard import Simulation_standard
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    plot_path = os.path.join(os.path.join(os.getcwd(), config['models_path_name'], 'model_'+str(config['model_to_test']), ''), 'test_standard', '')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)


    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation_standard(
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings_standard.ini', dst=os.path.join(plot_path, 'testing_settings_standard.ini'))

    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue_standard', xlabel='Step', ylabel='Queue lenght (vehicles)')
