from .controller import Controller


class BasicBotController(Controller):
    def get_response(self, game):
        if (game.x, game.y) == (game.x_right - game.square_size, game.y_bottom - 2 * game.square_size):
            dx = 0
            dy = 1
        elif (game.x, game.y) == (game.x_left + game.square_size, game.y_bottom - game.square_size):
            dx = 0
            dy = -1
        elif game.y == game.y_bottom - game.square_size:
            dx = -1
            dy = 0
        elif game.x % (2 * game.square_size) == game.square_size:
            # At odd columns, going up
            if game.y != game.y_top + game.square_size:
                dx = 0
                dy = -1
            else:
                dx = 1
                dy = 0
        else:
            # At odd columns, going down
            if game.y != game.y_bottom - 2 * game.square_size:
                dx = 0
                dy = 1
            else:
                dx = 1
                dy = 0

        changed_direction = (game.dx, game.dy) != (dx, dy)
        return dx, dy, changed_direction