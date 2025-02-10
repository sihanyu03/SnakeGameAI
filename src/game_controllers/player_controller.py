from .controller import Controller

import pygame


class PlayerController(Controller):
    def get_response(self, game) -> tuple[int, int, bool]:
        dx, dy, changed_direction = game.dx, game.dy, False

        if game.buffer:
            key = game.buffer.popleft()
            if key == pygame.K_LEFT:
                if game.dx != 1:
                    dx = -1
                    dy = 0
                    changed_direction = True
            elif key == pygame.K_RIGHT:
                if game.dx != -1:
                    dx = 1
                    dy = 0
                    changed_direction = True
            elif key == pygame.K_UP:
                if game.dy != 1:
                    dx = 0
                    dy = -1
                    changed_direction = True
            elif key == pygame.K_DOWN:
                if game.dy != -1:
                    dx = 0
                    dy = 1
                    changed_direction = True

        return dx, dy, changed_direction