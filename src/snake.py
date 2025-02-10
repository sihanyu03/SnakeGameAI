from collections import deque


class Snake:
    def __init__(self, x: int, y: int):
        self.blocks = deque([(x, y)])

    def move(self, head: tuple[int, int]):
        self.blocks.append(head)

    def pop(self):
        return self.blocks.popleft()

    def __len__(self):
        return len(self.blocks)