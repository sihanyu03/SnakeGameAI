import os.path
import pickle
import visualize
import neat


def main():
    config_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(config_dir, 'neat_config.txt')
    )

    best_genome_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'best_nn.pkl')
    with open(best_genome_path, 'rb') as file:
        best_genome = pickle.load(file)

    visualize.draw_net(config, best_genome, True)


if __name__ == '__main__':
    main()