from game_controllers.nn_controller import NNController
from game import Game
from utils import get_checkpoint_name

import os.path
import pickle
from functools import partial
from collections import defaultdict
import multiprocessing

import neat
import pygame
from numpy import inf

causes_of_death = defaultdict(int)


def eval_genome(genome, config, i, num_obstacles, draw, start_len, verbose):
    controller = NNController(genome, config, training=True)
    game = Game(controller=controller, num_obstacles=num_obstacles, draw=draw, start_len=start_len, fps=inf)

    genome.fitness = 0

    while game.running:
        status_code = game.loop()

        if status_code == 1:
            exit('Force quit')

        if game.draw:
            pygame.display.update()
            if game.fps != inf:
                game.clock.tick(game.fps)

    genome.fitness += game.score + 0.25 * (1 / max(1, abs(game.x - game.food[0])))

    if verbose:
        causes_of_death[game.cause_of_death] += 1
        print(f'genome: {i}, score={game.score}, fitness={genome.fitness:.2f}, cause of death={game.cause_of_death}, death_cause_counts={dict(causes_of_death)}')

    if game.draw:
        pygame.quit()

    return genome.fitness


def eval_genomes(genomes, config, num_obstacles, draw, start_len, verbose):
    for i, (genome_id, genome) in enumerate(genomes):
        eval_genome(genome, config, i=i, num_obstacles=num_obstacles, draw=draw, start_len=start_len, verbose=verbose)


def eval_genomes_generator(num_obstacles, draw, start_len, verbose):
    return partial(eval_genomes, num_obstacles=num_obstacles, draw=draw, start_len=start_len, verbose=verbose)

def eval_genome_generator(num_obstacles, start_len, verbose):
    return partial(
        eval_genome,
        i=None,
        num_obstacles=num_obstacles,
        draw=False,
        start_len=start_len,
        verbose=verbose
    )


def run(config_dir,
        n,
        checkpoint_freq=None,
        continue_from_checkpoint=True,
        start_len=1,
        multiprocess=True,
        verbose=False,
        num_obstacles=0,
        draw=True
):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(config_dir, 'neat_config.txt')
    )

    if continue_from_checkpoint:
        checkpoint_name = get_checkpoint_name(os.path.join(config_dir, 'checkpoints'))
    else:
        checkpoint_name = None

    if checkpoint_name is not None:
        population = neat.Checkpointer.restore_checkpoint(os.path.join(config_dir, 'checkpoints', checkpoint_name))
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(
        checkpoint_freq,
        filename_prefix=os.path.join(config_dir, 'checkpoints', 'neat-checkpoint-')
    ))

    if not multiprocess:
        winner = population.run(eval_genomes_generator(num_obstacles=num_obstacles, draw=draw, start_len=start_len, verbose=verbose), n)
    else:
        num_workers = max(multiprocessing.cpu_count() - 1, 1)
        parallel_evaluator = neat.ParallelEvaluator(
            num_workers,
            eval_genome_generator(
                num_obstacles=num_obstacles,
                start_len=start_len,
                verbose=verbose
            )
        )
        winner = population.run(parallel_evaluator.evaluate, n)

    with open('best_nn.pkl', 'wb') as file:
        pickle.dump(winner, file)


def main():
    config_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
    n = 250
    run(
        config_dir,
        n=n,
        checkpoint_freq=n-1,
        continue_from_checkpoint=True,
        start_len=1,
        multiprocess=True,
        verbose=False,
        num_obstacles=100,
        draw=True
    )


if __name__ == '__main__':
    main()
