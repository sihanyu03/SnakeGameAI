import yaml
import os.path


def read_game_config():
    game_config_file_name = 'game_config.yaml'
    with open(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            game_config_file_name
        ),
        'r'
    ) as file:
        game_config = yaml.safe_load(file)
        for key in ('WIDTH', 'HEIGHT', 'SQUARE_SIZE', 'START_FPS'):
            if key not in game_config:
                raise ValueError(f'{game_config_file_name} file format wrong. Missing key {key}')

        WIDTH = game_config['WIDTH']
        HEIGHT = game_config['HEIGHT']
        SQUARE_SIZE = game_config['SQUARE_SIZE']
        START_FPS = game_config['START_FPS']

        return WIDTH, HEIGHT, SQUARE_SIZE, START_FPS

def get_highscore():
    try:
        with open(os.path.join(os.path.dirname(__file__), os.path.pardir, 'saved_data.yaml'), 'r') as file:
            return yaml.safe_load(file).get('highscore', 0)
    except FileNotFoundError:
        return 0

def get_checkpoint_name(path):
    dirs = os.listdir(path)
    result = None
    for d in dirs:
        if d.startswith('neat-checkpoint-') and (result is None or int(d[16:]) > int(result[16:])):
            result = d
    return result
