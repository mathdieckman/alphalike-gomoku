import math
from typing import Union
import torch


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
digits = "1234567890123456789"


class Node:
    def __init__(self,
                 backprop_node: Union[None, 'Node'], is_terminal: bool, layer: int, value: float,
                 last_move: Union[None, tuple[int, int, int]], board_rep: torch.Tensor,
                 move_set: Union[None, frozenset[tuple[int, int, int]]], policy: torch.Tensor):
        self.superterminal = False
        self.layer = layer
        self.backprop_node = backprop_node
        self.last_move = last_move

        self.cross = False
        self.leaf = True
        self.children = dict[tuple[int, int, int], Node]()
        self.terminal = is_terminal
        self.visits = 1
        self.move_set = move_set
        self.board_rep = board_rep
        self.policy = policy.to("cuda", non_blocking=True)
        self.future_visit_map = torch.zeros_like(
            policy, dtype=int, requires_grad=False).to("cuda", non_blocking=True)
        self.future_eval_map = torch.zeros_like(
            policy, requires_grad=False).to("cuda", non_blocking=True)
        self.current_eval = value
        # I think these must take 1 to 2 kilobytes per node. Correct memory management crucial.

    def update(self, value: float) -> None:
        self.current_eval = self.current_eval + \
            (value - self.current_eval) / self.visits
        self.visits += 1

    def updateFutures(self, x: int, y: int, value: float) -> None:
        self.future_visit_map[x, y] += 1
        self.future_eval_map[x, y] = \
            self.future_eval_map[x, y] + \
            (value - self.future_eval_map[x, y]) / self.future_visit_map[x, y]

    def __repr__(self):
        if self.last_move == None:
            return ""
        return ("X: " if self.layer % 2 == 0 else "O: ") + \
            "node("+alphabet[self.last_move[0]]+" " + \
            alphabet[self.last_move[1]] + ")"


def print_tensor_probs(node: Node, its=None):
    number_system = "ihgfedcba-\u00B7\u2022123456789"

    def get_letter(num, abs_max):
        if math.isnan(num):
            return "X"
        if num >= 0:
            return number_system[10+math.ceil(num/(abs_max / 10 + 1e-6))]
        if num < 0:
            return number_system[10+math.floor(num/(abs_max / 10 + 1e-6))]

    print(node.policy)
    policy_abs_max = node.policy.clone().abs().max()
    visit_abs_max = node.future_visit_map.clone().abs().max()
    eval_abs_max = node.future_eval_map.clone().abs().max()
    out = ""
    out += "board" if not its else "board " + str(its)
    for _ in range(17):
        out += " "
    out += "policy"
    for _ in range(17):
        out += " "
    out += "evals"
    for _ in range(17):
        out += " "
    out += "visits\n"
    out += " "
    for z in range(19):
        out += alphabet[z]
    out += "   "
    for z in range(19):
        out += alphabet[z]
    out += "   "
    for z in range(19):
        out += alphabet[z]
    out += "   "
    for z in range(19):
        out += alphabet[z]
    out += "\n"
    for x in range(19):
        out += digits[18-x]
        for y in range(19):
            if node.board_rep[0, 0, x, y]:
                out += "X"
            elif node.board_rep[0, 1, x, y]:
                out += "O"
            else:
                out += "\u00B7"
        out += digits[18-x]
        out += " "
        out += digits[18-x]
        for y in range(19):
            if node.board_rep[0, 0, x, y]:
                out += " "
            elif node.board_rep[0, 1, x, y]:
                out += " "
            else:
                out += get_letter(node.policy[x, y], policy_abs_max)
        out += digits[18-x]
        out += " "
        out += digits[18-x]
        for y in range(19):
            if node.board_rep[0, 0, x, y]:
                out += " "
            elif node.board_rep[0, 1, x, y]:
                out += " "
            else:
                out += get_letter(node.future_eval_map[x, y], eval_abs_max)
        out += digits[18-x]
        out += " "
        out += digits[18-x]
        for y in range(19):
            if node.board_rep[0, 0, x, y]:
                out += " "
            elif node.board_rep[0, 1, x, y]:
                out += " "
            else:
                out += get_letter(node.future_visit_map[x, y], visit_abs_max)
        out += digits[18-x]
        out += "\n"
    out += " "
    for z in range(19):
        out += alphabet[z]
    out += "   "
    for z in range(19):
        out += alphabet[z]
    out += "   "
    for z in range(19):
        out += alphabet[z]
    out += "   "
    for z in range(19):
        out += alphabet[z]
    out += "\nEval:"+str(node.current_eval)+"\n"+"Policy abs max" + \
        str(policy_abs_max)+"\nfuture_eval abs max"+str(eval_abs_max)
    print(out)
