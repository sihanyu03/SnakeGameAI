from snake import Snake
from game_controllers.controller import Controller
from game_controllers.player_controller import PlayerController
from game_controllers.nn_controller import NNController
from utils import read_game_config, get_highscore

import os.path
from collections import deque
from typing import Literal, Iterable
import random
import yaml

import numpy as np
import pygame


BLACK = (0, 0, 0)
WHITE = np.array([255, 255, 255])
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BROWN = (140, 70, 20)

start_instructions = (
    'Begin the game by pressing an arrow key',
    'Instructions:',
    '\tMove with arrow or WASD keys',
    '\tSpeed up by holding SPACE',
    '\tPause by pressing ESC,'
    '\tQuit by pressing Q'
)


class Game:
    def __init__(self, controller: Controller, num_obstacles=0, draw: bool = True, start_len: int = 1, fps: int | float = 10):
        self._width, self._height, self._square_size, self._start_fps = read_game_config()
        self._height += 2 * self._square_size

        self._controller = controller
        self._num_obstacles = num_obstacles
        self._draw = draw
        self._start_len = start_len
        self._start_fps = fps

        if (
            self._width % self._square_size != 0 or self._width < 5 * self._square_size
            or self._height % self._square_size != 0 or self._height < 5 * self._square_size
        ):
            raise ValueError('Invalid values for width, height or square size. Width and height must be multiples of square size, and at least 5 times as large as the square size')

        x_edges = list(range(0, self._width, self._square_size))
        y_edges = list(range(2 * self._square_size, self._height, self._square_size))
        N_x = len(x_edges)
        N_y = len(y_edges)

        self._x_left = 0
        self._x_right = self._width - self._square_size
        self._y_top = self._square_size * 2
        self._y_bottom = self._height - self._square_size

        self._edges = list(zip(
            x_edges + [self._x_right] * N_y + x_edges + [self._x_left] * N_y,
            [self._y_top] * N_x + y_edges + [self._y_bottom] * N_x + y_edges
        ))
        del x_edges, y_edges

        if self._draw:
            pygame.init()
            pygame.display.set_caption('Snake Game')
            self._window = pygame.display.set_mode((self._width, self._height))
            self._window.fill(BLACK)
            self.clock = pygame.time.Clock()
        else:
            self._window = None
            self.clock = None

        self._running = True

        # Variables to be initialised in the initialise() method
        self._x = None
        self._y = None
        self._dx = None
        self._dy = None
        self._snake = None
        self.buffer = None
        self._empty_squares = None
        self._obstacles = None
        self._game_over = None
        self._paused = None
        self._food = None
        self._score = None
        self._moves = None
        self._fps = None
        self._fps_multiplier_hold = None
        self._fps_multiplier_press = None
        self._highscore = None
        self._remaining_blocks = None

        # Information attributes
        self._moves_since_last_score = None
        self._cause_of_death = None

        self._initialise()

    def _initialise(self):
        self._x, self._y = self._start_coords()
        self._x = self._width // self._square_size // 2 * self._square_size
        self._y = self._height // self._square_size // 2 * self._square_size
        self._dx = 0
        self._dy = 0
        self._snake = Snake(self._x, self._y)
        self.buffer = deque()

        self._empty_squares = set()
        for x in range(self._x_left + self._square_size, self._x_right, self._square_size):
            for y in range(self._y_top + self._square_size, self._y_bottom, self._square_size):
                self._empty_squares.add((x, y))

        start_coords = self._start_coords()
        self._empty_squares.remove(start_coords)
        self._empty_squares.remove((start_coords[0], start_coords[1] - self._square_size))

        self._obstacles = set()
        for i in range(self._num_obstacles):
            empty_squares_list = tuple(self._empty_squares)
            coords = random.choice(empty_squares_list)
            self._obstacles.add(coords)
            self._empty_squares.remove(coords)

        self._empty_squares.add(start_coords)
        self._empty_squares.add((start_coords[0], start_coords[1] - self._square_size))

        self._game_over = False
        self._paused = False
        self._food = self.generate_food()
        self._score = 0
        self._moves = 0
        self._fps = self._start_fps
        self._fps_multiplier_hold = 1
        self._fps_multiplier_press = 1
        self._highscore = get_highscore()
        self._remaining_blocks = self._start_len - 1
        self._moves_since_last_score = 0
        self._cause_of_death = 'None'

    def loop(self) -> int:
        if self._moves_since_last_score > 500:
            self._game_over = True
            self._cause_of_death = 'Too many moves without scoring'
            self._moves_since_last_score = 0
            return 0

        if self._draw:
            self._window.fill(BLACK)
            self._draw_line(self._edges, BROWN)
            self._draw_obstacles()

        if self._game_over:
            return self._handle_game_over()

        if self._draw:
            if pygame.key.get_pressed()[pygame.K_SPACE]:
                self._fps_multiplier_hold = 3
            else:
                self._fps_multiplier_hold = 1

            status_code = self._fill_buffer()
            if status_code == 1:
                return 1

            self._display_text(
                f'Score: {self._score}, Current speed: {(self._fps * self._fps_multiplier_hold * self._fps_multiplier_press):.2f}, Highscore: {self._highscore}',
                colour=WHITE,
                location='top'
            )

            if self._paused:
                self._display_text(
                    'Paused, resume by pressing ESC',
                    colour=WHITE,
                    location='middle',
                    fontsize=40
                )
                self._draw_snake()
                self._draw_food()
                return 0

        # Get controller response
        dx, dy, changed_direction = self._controller.get_response(game=self)

        # If not started, show beginning screen
        if dx == 0 and dy == 0:
            if self._draw:
                self._display_text(start_instructions, colour=WHITE, location='middle')
                self._draw_snake()

            return 0
        else:
            if not self._check_move_validity(dx, dy):
                raise ValueError(f'Controller gave invalid set of moves: dx={dx}, dy={dy}, even though the direction of movement was dx={self._dx}, dy={self._dy}')
            self._dx, self._dy = dx, dy

        if self._draw:
            self._draw_food()

        self._moves += changed_direction
        self._moves_since_last_score += changed_direction

        self._x += self._dx * self._square_size
        self._y += self._dy * self._square_size

        head = (self._x, self._y)
        if head in self._empty_squares:
            self._empty_squares.remove(head)

        if head == self._food:
            self._food = self.generate_food()

            if self._food is None:
                self._game_over = True
                self._cause_of_death = 'All possible squares occupied'
                return 0

            self._score += 1
            self._moves_since_last_score = 0
            self._fps = self._calculate_fps()
        elif self._remaining_blocks >= 1:
            self._remaining_blocks -= 1
        else:
            self._empty_squares.add(self._snake.pop())

        if self._check_death(head):
            self._game_over = True
            return 0

        self._snake.move(head)

        if self._draw:
            self._draw_snake()

        return 0

    def _fill_buffer(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                self._running = False
                return 1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._paused = not self._paused
                    self.buffer.clear()
                    break
                elif event.key in (
                    pygame.K_LEFT,
                    pygame.K_RIGHT,
                    pygame.K_UP,
                    pygame.K_DOWN
                ):
                    self.buffer.append(event.key)
                elif event.key == pygame.K_1:
                    self._fps_multiplier_press = 1
                elif event.key == pygame.K_2:
                    self._fps_multiplier_press = 3
                elif event.key == pygame.K_3:
                    self._fps_multiplier_press = 5
                elif event.key == pygame.K_4:
                    self._fps_multiplier_press = 10
                elif event.key == pygame.K_5:
                    self._fps_multiplier_press = np.inf


    def _handle_game_over(self):
        if isinstance(self._controller, NNController) and self._controller.training:
            self._running = False
            return 0

        if not self._draw:
            self._running = False
            return 0

        self._draw_snake()
        if self._food is not None:
            self._draw_block(self._food, RED)

        self._display_game_over_screen()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                self._running = False
                return 1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._initialise()
                    break

        return 0

    def _display_text(self, msg: str | Iterable[str], location: Literal['middle', 'top'], colour: WHITE, fontsize=25):
        if isinstance(msg, str):
            msg = [msg]

        texts = []
        max_width = 0
        max_height = 0
        total_height = 0
        font = pygame.font.SysFont('Arial', fontsize)
        for m in msg:
            m = m.replace('\t', '    ')
            text = font.render(m, True, colour)
            w, h = text.get_size()
            max_width = max(max_width, w)
            max_height = max(max_height, h)
            total_height += h
            texts.append(text)

        for i, text in enumerate(texts):
            if location == 'middle':
                coords = [self._width // 2 - max_width // 2, self._height // 3 - total_height // 2 + i * max_height]
            elif location == 'top':
                coords = [self._width // 2 - max_width // 2, (i + 1) * max_height]
            else:
                raise ValueError(f"Invalid value to parameter location, should be 'middle' or 'top', got {location}")
            self._window.blit(text, dest=coords)

    def _display_game_over_screen(self):
        display_msg = [f'Game over, {self._cause_of_death}!']
        if isinstance(self._controller, PlayerController) and self._score > self._highscore:
            display_msg.append('New highscore!')
            with open(os.path.join(os.path.dirname(__file__), os.path.pardir, 'saved_data.yaml'), 'w') as file:
                yaml.dump({'highscore': self._score}, file)
        display_msg.extend([
            f'Final score: {self._score}',
            f'Total moves: {self._moves}',
            'Press Q to quit or R to play again'
        ])

        self._display_text(display_msg, colour=RED, location='middle')

    def _draw_line(self, points: Iterable[tuple[int, int]], colour: tuple[int, int, int]):
        if self._draw:
            for point in points:
                self._draw_block(point, colour)

    def _draw_block(self, point: tuple[int, int], colour):
        if self._draw:
            pygame.draw.rect(self._window, colour, [point[0],  point[1], self._square_size, self._square_size])

    def _draw_snake(self):
        if self._draw:
            for i, point in enumerate(self._snake.blocks):
                colour = WHITE * (0.7 * np.exp(-0.1 * (len(self._snake) - 1 - i)) + 0.3)
                self._draw_block(point, colour=colour)

    def _draw_obstacles(self):
        if self._draw:
            for point in self._obstacles:
                self._draw_block(point, colour=BROWN)

    def _start_coords(self):
        return (
            self._width // self._square_size // 2 * self._square_size,
            self._height // self._square_size // 2 * self._square_size
        )

    def _calculate_fps(self):
        return 0.25 * (0.3 * self._score ** 0.7 + np.log(0.3 * self._score + 1)) + self._start_fps

    def generate_food(self):
        if self._empty_squares:
            return random.choice(list(self._empty_squares))

    def _draw_food(self):
        self._draw_block(self._food, colour=RED)

    def _check_move_validity(self, dx, dy):
        if (dx != 0 and dy != 0) or abs(dx) > 1 or abs(dy) > 1:
            return False

        if (self._dx, self._dy) == (0, -1):  # Going up
            if dy == 1:
                return False
        elif (self._dx, self._dy) == (1, 0):  # Going right
            if dx == -1:
                return False
        elif (self._dx, self._dy) == (0, 1):  # Going down
            if dy == -1:
                return False
        elif (self._dx, self._dy) == (-1, 0):  # Going left
            if dx == 1:
                return False

        return True

    def _check_death(self, head: tuple[int, int]) -> bool:
        if head[0] in (self._x_left, self._x_right) or head[1] in (self._y_top, self._y_bottom):
            self._cause_of_death = 'Hit a wall'
            return True
        elif head in self._obstacles:
            self._cause_of_death = 'Hit an obstacle'
            return True

        for i, block in enumerate(self._snake.blocks):
            if block == head and i != len(self._snake) - 1:
                self._cause_of_death = 'Hit the body'
                return True

        return False

    @property
    def running(self):
        return self._running

    @property
    def draw(self):
        return self._draw

    @property
    def fps(self):
        return self._fps * self._fps_multiplier_hold * self._fps_multiplier_press

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def food(self):
        return self._food

    @property
    def x_right(self):
        return self._x_right

    @property
    def x_left(self):
        return self._x_left

    @property
    def y_top(self):
        return self._y_top

    @property
    def y_bottom(self):
        return self._y_bottom

    @property
    def square_size(self):
        return self._square_size

    @property
    def empty_squares(self):
        return self._empty_squares

    @property
    def score(self):
        return self._score

    @property
    def moves_since_last_score(self):
        return self._moves_since_last_score

    @property
    def cause_of_death(self):
        return self._cause_of_death

