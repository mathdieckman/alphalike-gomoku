from typing import Tuple
from types import FunctionType
from math import log, pow

from base.graph_node import Node, print_tensor_probs

import base.conv_calcs as conv_calcs
import torch
from torch.nn.functional import pad
import gc
import random


def indexToTuple(index: int, color: int, size: int):
    return (int(index)//size, int(index) % size, color)


middle = pad(torch.ones((7, 7)), (4, 4, 4, 4)).to("cuda")


class CPUCT_MCTS:
    def __init__(self, pv_func: FunctionType, size=19,
                 training=False, self_play=False, verbose=False):
        self.pv_func = pv_func
        self.root = Node(None, False, 0, 0, None,
                         torch.zeros(
                             (1, 2, size, size),
                             requires_grad=False
                         ).to("cuda", non_blocking=True),
                         None,
                         pv_func(
                             torch.zeros(
                                 (1, 2, size, size),
                                 requires_grad=False
                             ).to("cuda", non_blocking=True),
                             0)[0]
                         )
        # must start layer zero for color math reasons
        self.size = size

        self.CPUCT_BASE = 20000  # magic constants
        self.CPUCT_INIT = 2.5
        self.nodes_so_far = 1
        self.step_down_epsilon = None
        self.selection_temp = None
        if training:
            self.step_down_epsilon = 0.025
            self.selection_temp = 3
        else:
            self.step_down_epsilon = 0.2
            self.selection_temp = 5
        self.training = training
        self.self_play = self_play
        self.verbose = verbose
        self.game_over = False
        self.am_winner = False
        self.its = 0

    def stepDown(self, node: Node) -> tuple[Node, float]:
        node.leaf = False
        next_node_index = None

        if random.random() < self.step_down_epsilon and torch.count_nonzero(torch.ones_like(node.future_visit_map)-node.future_visit_map.ne(0).long()-node.board_rep[0, 0, :, :]-node.board_rep[0, 1, :, :]):
            next_node_index = None
            try:
                next_node_index = torch.multinomial(torch.flatten(torch.ones_like(
                    node.future_visit_map)-node.future_visit_map.ne(0).long()-node.board_rep[0, 0, :, :]-node.board_rep[0, 1, :, :]).float(), 1)
            except:
                print(torch.flatten(torch.ones_like(node.future_visit_map)-node.future_visit_map.ne(
                    0).long()-node.board_rep[0, 0, :, :]-node.board_rep[0, 1, :, :]).float())
                print(node.future_visit_map)
                print(-node.board_rep[0, 0, :, :]-node.board_rep[0, 1, :, :])
                raise
        else:
            next_node_index = torch.argmax(torch.flatten(
                (node.future_eval_map - 1 + node.future_visit_map.ne(0)
                 + (node.policy) * (pow(node.visits, 0.5) / (1+node.future_visit_map)
                                    * (self.CPUCT_INIT + log((node.visits + self.CPUCT_BASE)/self.CPUCT_BASE)))
                 - 999*node.board_rep[0, 0, :, :]-999*node.board_rep[0, 1, :, :])))

        xyc = indexToTuple(next_node_index, (node.layer+1) % 2, self.size)
        x, y, c = xyc
        if xyc not in node.children.keys():
            self.its += 1
            if self.its % 100 == 0:
                print_tensor_probs(self.root, its=self.its)
            new_board_rep = node.board_rep.clone()
            new_board_rep.requires_grad = False
            new_board_rep[0, c, x, y] = True
            self.nodes_so_far = self.nodes_so_far + 1

            p, v = self.pv_func(new_board_rep, c)

            # p.requires_grad=False
            # v.requires_grad=False

            node.children[xyc] = Node(node, node.layer == 253 or
                                      node.terminal or conv_calcs.circ_check_if_won(
                                          new_board_rep),
                                      node.layer+1, v, xyc, new_board_rep, None, p)

            if node.children[xyc].terminal:
                if not node.terminal:
                    node.superterminal = True
                    node.children[xyc].current_eval = -1
                    node.current_eval = 1
                    node.visits = 9999  # stop values from changing
                    # enforce selection of this node during final selection
                    node.future_visit_map[x, y] = 9999
                    node.future_eval_map[x, y] = 1
                    if node.backprop_node:
                        node.backprop_node.future_eval_map[node.last_move[0],
                                                           node.last_move[1]
                                                           ] = -1
                else:
                    node.children[xyc].current_eval = - node.current_eval
                    node.children[xyc].visits = 9999
                    node.future_visit_map[x, y] = 9999
                    node.future_eval_map[x, y] = node.current_eval
            else:
                node.children[xyc].current_eval = v

        return node.children[xyc], node.children[xyc].current_eval

    def selectLeaf(self) -> Tuple[Node, float]:
        node = self.root
        while not node.leaf:
            if node.superterminal:
                return node, 1
            if node.terminal:
                print("wha")
                return node, -1
            node, _ = self.stepDown(node)
        return self.stepDown(node)

    def backProp(self, node: Node, value: float) -> None:
        last_node = node.backprop_node

        while last_node:
            value = -value
            last_node.updateFutures(
                node.last_move[0], node.last_move[1], value)
            last_node.update(value)

            node = last_node
            last_node = last_node.backprop_node

    def moveSelect(self) -> tuple[int, int]:
        index = None
        # mod = 0
        # top = torch.topk(torch.flatten(self.root.future_visit_map), 5)
        unmodded = torch.flatten(self.root.future_visit_map.float())
        index = torch.multinomial(torch.pow(unmodded, self.selection_temp), 1)
        new_root = self.root.children[indexToTuple(
            index, (self.root.layer+1) % 2, self.size)]

        if self.verbose:
            # pass
            # print("before:",self.root.layer)
            # print(self.nodes_ever)
            print_tensor_probs(self.root)
            # print(self.root.policy)
            # print("after:",self.root.layer)
            # print_tensor_probs(new_root)

        new_root.backprop_node = None
        new_root.last_move = None
        self.root = new_root

        if self.root.terminal:
            self.game_over = True
            self.am_winner = True

        gc.collect()
        torch.cuda.empty_cache()

        return int(index)//19, int(index) % 19

    def push(self, x, y) -> None:
        if self.self_play:
            return
        new_root = None
        if (x, y, (self.root.layer+1) % 2) in self.root.children.keys():
            new_root = self.root.children[x, y, (self.root.layer+1) % 2]
        else:
            new_board_rep = self.root.board_rep.clone()
            new_board_rep[0, (self.root.layer+1) % 2, x, y] = True
            self.nodes_so_far = self.nodes_so_far + 1
            p, v = self.pv_func(new_board_rep, (self.root.layer+1) % 2)

            new_root = Node(None, self.root.layer == 253 or
                            conv_calcs.circ_check_if_won(new_board_rep),
                            self.root.layer +
                            1, v, (x, y, (self.root.layer+1) % 2),
                            new_board_rep, None, p)

        new_root.backprop_node = None
        new_root.last_move = None

        self.root = new_root

        if self.root.terminal:
            self.game_over = True
            self.am_winner = False
