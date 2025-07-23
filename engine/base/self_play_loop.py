import base.conv_calcs
import torch.cuda.memory
import torch.profiler.profiler
import base.board as Board

from abc import ABC, abstractmethod


class Abc_Player(ABC):
    def __init__(self, board):
        self.board = board

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def inform(self):
        pass


class SelfPlay:
    def __init__(self, p1: Abc_Player, p2: Abc_Player, b: Board):
        self.game_log = []
        self.p1 = p1
        self.p2 = p2
        self.b = b

    def play(self) -> None:
        p1_to_move = True
        while not base.conv_calcs.circ_check_if_won(torch.stack((self.b.white, self.b.black)).reshape(1, 2, 19, 19)) and not self.b.check_if_full() and not len(self.game_log) > 224:

            if p1_to_move:
                m = self.p1.move()
                self.b.push(*m)
                self.p2.inform(*m)
                p1_to_move = False
            else:
                m = self.p2.move()
                self.b.push(*m)
                self.p1.inform(*m)
                p1_to_move = True
            self.game_log.append(m)

    def get_log(self) -> list[tuple[int, int]]:
        return self.game_log
