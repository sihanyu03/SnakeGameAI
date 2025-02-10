from .controller import Controller

import numpy as np
import neat


def _is_obstacle(x, y, empty_squares):
    return 1 if (x, y) not in empty_squares else 0

def _argmax(lst):
    return max(range(len(lst) - 1, -1, -1), key=lambda i: lst[i])

def _arctan(a, b):
    if b == 0:
        return np.sign(a) * np.pi / 2
    return np.arctan2(b, a)

def _wrap_and_normalise(angle):
    return ((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi

class NNController(Controller):
    def __init__(self, genome, config, training=False, print_steps=False):
        self._net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.training = training
        self._print_steps = print_steps

    def get_response(self, game) -> tuple[int, int, bool]:
        initial_dir = (game.dx, game.dy)

        if initial_dir == (0, 0):
            dx = 0
            dy = -1
            changed_direction = True
            return dx, dy, changed_direction

        theta_dir = np.arctan2(game.dx, -game.dy)
        theta_food = np.arctan2(
            game.food[0] - game.x,
            game.y - game.food[1]
        )
        food_angle = _wrap_and_normalise(theta_food - theta_dir)

        if initial_dir == (0, -1):  # Going up
            obstacle_forward = _is_obstacle(game.x, game.y - game.square_size, game.empty_squares)
            obstacle_right = _is_obstacle(game.x + game.square_size, game.y, game.empty_squares)
            obstacle_left = _is_obstacle(game.x - game.square_size, game.y, game.empty_squares)

            output = self._get_nn_output(obstacle_forward, obstacle_right, obstacle_left, food_angle)
            if output == 0:
                (dx, dy), changed_direction = initial_dir, False
            elif output == 1:  # Turn right
                dx, dy, changed_direction = 1, 0, True
            else:  # output == 2
                dx, dy, changed_direction = -1, 0, True

        elif initial_dir == (1, 0):  # Going right
            obstacle_forward = _is_obstacle(game.x + game.square_size, game.y, game.empty_squares)
            obstacle_right = _is_obstacle(game.x, game.y + game.square_size, game.empty_squares)
            obstacle_left = _is_obstacle(game.x, game.y - game.square_size, game.empty_squares)

            output = self._get_nn_output(obstacle_forward, obstacle_right, obstacle_left, food_angle)
            if output == 0:
                (dx, dy), changed_direction = initial_dir, False
            elif output == 1:  # Turn right
                dx, dy, changed_direction = 0, 1, True
            else:  # output == 2
                dx, dy, changed_direction = 0, -1, True

        elif initial_dir == (0, 1):  # Going down
            obstacle_forward = _is_obstacle(game.x, game.y + game.square_size, game.empty_squares)
            obstacle_right = _is_obstacle(game.x - game.square_size, game.y, game.empty_squares)
            obstacle_left = _is_obstacle(game.x + game.square_size, game.y, game.empty_squares)

            output = self._get_nn_output(obstacle_forward, obstacle_right, obstacle_left, food_angle)
            if output == 0:
                (dx, dy), changed_direction = initial_dir, False
            elif output == 1:  # Turn right
                dx, dy, changed_direction = -1, 0, True
            else:  # output == 2
                dx, dy, changed_direction = 1, 0, True

        elif initial_dir == (-1, 0):  # Going left
            obstacle_forward = _is_obstacle(game.x - game.square_size, game.y, game.empty_squares)
            obstacle_right = _is_obstacle(game.x, game.y - game.square_size, game.empty_squares)
            obstacle_left = _is_obstacle(game.x, game.y + game.square_size, game.empty_squares)

            output = self._get_nn_output(obstacle_forward, obstacle_right, obstacle_left, food_angle)
            if output == 0:
                (dx, dy), changed_direction = initial_dir, False
            elif output == 1:  # Turn right
                dx, dy, changed_direction = 0, -1, True
            else:  # output == 2
                dx, dy, changed_direction = 0, 1, True

        else:
            raise ValueError(f'Invalid direction, got dx={game.dx}, dy={game.dy}')

        return dx, dy, changed_direction

    def _get_nn_output(self, obstacle_forward, obstacle_right, obstacle_left, food_angle):
        genome_input = (
            obstacle_forward,
            obstacle_right,
            obstacle_left,
            food_angle
        )
        genome_output = self._net.activate(genome_input)
        output = _argmax(genome_output)

        if self._print_steps:
            print(genome_input, genome_output, output)
        return output
