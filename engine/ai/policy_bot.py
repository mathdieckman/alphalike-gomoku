from ai.abc_player import Abc_Player
import torch
import math


class Policy_Bot(Abc_Player):
    def __init__(self, board, policy, size, nodes, is_white):
        super().__init__(board)
        self.policy = policy.to("cuda")
        self.size = size
        self.is_white = is_white
        self.tree = []
        self.nodes = nodes

    def move(self):

        if self.is_white:
            input = torch.stack(
                (self.board.white, self.board.black)).to("cuda")
        else:
            input = torch.stack(
                (self.board.black, self.board.white)).to("cuda")
        input = input[None, :]

        density = self.policy(input)[0]
        density = density - 9999*(self.board.black + self.board.white)
        density = torch.relu(density)

        flat = density.flatten()
        flat += 0.00001
        flat = flat/flat.norm()
        # print(flat)
        z = torch.multinomial(flat, 1)
        # print(z)
        return math.floor(z/self.size), int(z % self.size)

    def inform(self, x, y):
        pass
