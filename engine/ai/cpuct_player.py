from ai.mcts import CPUCT_MCTS
import torch
import gc
from base.self_play_loop import Abc_Player


class CPuct_Player(Abc_Player):
   
    def __init__(self, board, pv_func, is_white, max_its, record_data=False,
                 training=False, self_play=False, verbose=False, name="a"):
        self.board = board
        self.pv_func = pv_func
        self.is_white = is_white
        self.record_data = record_data
        self.d = CPUCT_MCTS(pv_func, training=training,
                            self_play=self_play, verbose=verbose)

        self.its = 0
        self.max_its = max_its
        self.training = training
        self.out_list = []
        self.name = name
        self.count = 0

    def move(self):
        self.count += 1
        self.its += 1
        while (self.its % self.max_its):
            self.its += 1
            self.d.backProp(*self.d.selectLeaf())
            if self.its % 1000 == True:
                gc.collect()
                torch.cuda.empty_cache()
        if self.record_data and self.count:
            torch.save(((torch.flip(self.d.root.board_rep.clone(), (1,)) if self.is_white else self.d.root.board_rep.clone(),
                         self.d.root.future_eval_map.clone() + self.d.root.future_visit_map.ne(0)-1,
                         not self.is_white),), './train/data/'+self.name+str(int(self.is_white))+str(self.its)+'.pth')
            torch.save(((torch.flip(self.d.root.board_rep.clone(), (1,)) if self.is_white else self.d.root.board_rep.clone(),
                         self.d.root.future_eval_map.clone() + self.d.root.future_visit_map.ne(0)-1,
                         not self.is_white),), './train/games/'+self.name+str(int(self.is_white))+str(self.its)+'.pth')

        return self.d.moveSelect()

    def inform(self, x, y):
        self.d.push(x, y)
