import random


class RandomGuesser:
    def __init__(self, board):
        self.board = board

    def move(self):
        moved = False
        size = self.board.size
        while not moved and not self.board.check_if_full():
            a, b = random.randint(0, size-1), random.randint(0, size-1)
            if self.board.validate_move(a, b):
                return a, b

    def inform(self, x, y):
        pass
