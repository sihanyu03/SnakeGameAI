from game import Game
from game_controllers.controller import Controller
from game_controllers.player_controller import PlayerController
from game_controllers.basic_bot_controller import BasicBotController
from src.game_controllers.nn_controller import NNController

import pickle
import os.path
import neat
import pygame
from numpy import inf


def get_nn_controller(print_steps=False, path=None):
    if path is None:
        best_genome_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'best_nn.pkl')
    else:
        best_genome_path = os.path.join(os.path.dirname(__file__), os.path.pardir, path)
    with open(best_genome_path, 'rb') as file:
        best_genome = pickle.load(file)

    config_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'neat_config.txt')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    return NNController(best_genome, config, print_steps=print_steps)


def run_game(controller: Controller, num_obstacles=0):
    game = Game(controller, num_obstacles=num_obstacles, draw=True, start_len=1, fps=10)

    while game.running:
        game.loop()
        pygame.display.update()
        if game.fps != inf:
            game.clock.tick(game.fps)

    print(game.cause_of_death)
    pygame.quit()


if __name__ == '__main__':
    # controller = PlayerController()
    # controller = BasicBotController()

    controller = get_nn_controller(print_steps=True, path=os.path.join('nn_archive', '3', 'best_nn_500.pkl'))
    # controller = get_nn_controller(print_steps=True)
    run_game(controller, num_obstacles=0)