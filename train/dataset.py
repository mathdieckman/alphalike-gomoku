import torch
import pickle
import random
from base.board import Board
import os
global batches
batches = 0


class decision_point(torch.utils.data.IterableDataset):

    def __init__(self, start, end, moves_from_end):
        self.board_win_triple = []
        self.start = start
        self.end = end
        self.moves_from_end = moves_from_end
        self.current = 0
        self.game_logs = None
        self.num = 0
        self.its = 0
        self.count = None
        self.copy = []
        self.parity = 0
        self.winner = None

    def load_new_set(self, num):
        self.game_logs = [*torch.load("./train/data/"+str(num))]

    def __iter__(self):
        if len(self.copy) == 0:
            for f in os.listdir("./train/data"):
                self.load_new_set(f)
                for a, b, c in self.game_logs:
                    print(str(f))
                    winner = torch.load(
                        "./train/hack_flag/"+str(f)[0]+('0' if c else '1'))
                    print(c)
                    print(winner)
                    print("./train/hack_flag/"+str(f)[0]+('0' if c else '1'))
                    r1 = a.reshape(2, 15, 15).clone().requires_grad_()
                    r2 = torch.flip(r1, (-1,)).requires_grad_()
                    r3 = torch.flip(r1, (-2,)).requires_grad_()
                    r4 = torch.flip(r1, (-1, -2)).requires_grad_()
                    r5 = r1.transpose(-2, -1)
                    r6 = torch.flip(r5, (-1,))
                    r7 = torch.flip(r5, (-2,))
                    r8 = torch.flip(r5, (-1, -2))
                    b1 = b.reshape(1, 15, 15)
                    b2 = torch.flip(b1, (-1,))
                    b3 = torch.flip(b1, (-2,))
                    b4 = torch.flip(b1, (-1, -2))
                    b5 = b1.transpose(-2, -1)
                    b6 = torch.flip(b5, (-1,))
                    b7 = torch.flip(b5, (-2,))
                    b8 = torch.flip(b5, (-1, -2))
                    self.copy.append([r1, b1, c, winner])
                    # self.copy.append([r2,b2,c,winner])
                    # self.copy.append([r3,b3,c,winner])
                    # self.copy.append([r4,b4,c,winner])
                    # self.copy.append([r5,b5,c,winner])
                    # self.copy.append([r6,b6,c,winner])
                    # self.copy.append([r7,b7,c,winner])
                    # self.copy.append([r8,b8,c,winner])
            random.shuffle(self.copy)
        return iter(self.copy)

        # for g in self.game_logs[self.start:min(self.end,len(self.game_logs)-1)]:
        #     for _ in range(1):
        #         if len(g) == self.size*self.size:
        #             continue
        #         b=Board(self.size)
        #         r_int=random.randint(max(1,len(g)-self.moves_from_end),len(g))
        #         b.push_whole_log(g[0:r_int])
        #         if len(g)%2 and r_int%2:
        #             self.board_win_triple.append((torch.stack((b.white, b.black)),
        #                     torch.tensor((g[r_int-1][0],g[r_int-1][1],1),dtype=torch.long)))
        #         if len(g)%2 and not r_int%2:
        #             self.board_win_triple.append((torch.stack((b.black, b.white)),
        #                     torch.tensor((g[r_int-1][0],g[r_int-1][1],0),dtype=torch.long)))
        #         if not len(g)%2 and r_int%2:
        #             self.board_win_triple.append((torch.stack((b.white, b.black)),
        #                     torch.tensor((g[r_int-1][0],g[r_int-1][1],0),dtype=torch.long)))
        #         if not len(g)%2 and not r_int%2:
        #             self.board_win_triple.append((torch.stack((b.black, b.white)),
        #                     torch.tensor((g[r_int-1][0],g[r_int-1][1],1),dtype=torch.long)))
        # random.shuffle(self.board_win_triple)


if __name__ == "__main__":
    for x, y in decision_point(0, 2, 3, 2):
        print(x)
        print()
        print(y)
        print()
