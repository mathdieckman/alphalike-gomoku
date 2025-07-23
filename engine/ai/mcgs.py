# from gmku.board import check_if_won
# from typing import Tuple, Union, TypedDict
# from types import FunctionType
# from math import sqrt, inf, log,pow
# import torch
# import sys
# from mcts.node import Node,print_tensor_probs


# def indexToTuple(index:int,color:int):
#     return (int(index)//15,int(index)%15,color)

        
# class CPUCT_MCGS:
#     def __init__(self, policy_func:FunctionType,value_func:FunctionType, root:Node):
#         self.REPAIR_DIFF = 0.01
#         self.epsilon=0.1
#         self.CPUCT_BASE = 1000 # silver, et al. 2017, for a completely different game and net
#         self.CPUCT_INIT = 3 # changed this one
#         self.max_eval = -inf
#         self.min_eval = inf
#         self.nodes_by_moves = dict[frozenset[tuple[int,int,int]],Node]()
#         self.policy_func = policy_func
#         self.value_func = value_func
#         self.root = root # must start layer zero for color math reasons
#         self.nodes_ever = 0

#     def stepDown(self, node:Node) -> tuple[Node,float]:

#         next_node_index = torch.argmax(torch.flatten(
#                 (node.future_eval_map -1 + node.future_visit_map.ne(0)-999*node.board_rep[0,0,:,:]-999*node.board_rep[0,1,:,:]) +
#                 ((node.policy + self.epsilon)/(1+node.future_visit_map) *
#                  (self.CPUCT_INIT + log((node.visits + self.CPUCT_BASE)/self.CPUCT_BASE)) *
#                  (pow(node.visits,0.5)))))
#         # if self.nodes_ever%1000 == 999: 
#         #     print(
#         #         (node.future_eval_map -1 + node.future_visit_map.ne(0)-999*node.board_rep[0,0,:,:]-999*node.board_rep[0,1,:,:]) +
#         #         ((node.policy + self.epsilon) *
#         #          (self.CPUCT_INIT + log((node.visits + self.CPUCT_BASE)/self.CPUCT_BASE)) *
#         #          (pow(node.visits,0.5)/(1+node.future_visit_map))))

#         node.leaf = False

#         new_move = indexToTuple(next_node_index, (node.layer+1)%2)

#         new_set = node.move_set|frozenset((new_move,))
#         # print(new_set)
        
#         if new_set not in self.nodes_by_moves.keys():
#             new_board_rep = node.board_rep.clone()
#             new_board_rep[0,new_move[2],new_move[0],new_move[1]]=True

#             copied_board_rep = None

#             if not new_move[2]:
#                 copied_board_rep = new_board_rep
#             else:
#                 copied_board_rep = torch.flip(new_board_rep,(1,))

#             value = self.value_func(copied_board_rep)

#             if value > self.max_eval:
#                 self.max_eval = value

#             if self.min_eval > value:
#                 self.min_eval = value

#             self.nodes_ever = self.nodes_ever + 1
            
#             self.nodes_by_moves[new_set] = \
#                     Node(node, node.layer == 254 or 
#                     node.terminal or check_if_won(new_move[0],new_move[1],15,new_board_rep[0,new_move[2],:,:]), 
#                     node.layer+1, value, new_move, new_board_rep, new_set, self.policy_func(copied_board_rep))

#             if self.nodes_by_moves[new_set].terminal:
#                 if node.terminal:
#                     self.nodes_by_moves[new_set].current_eval = - node.current_eval
#                 else:
#                     self.nodes_by_moves[new_set].current_eval = 1 if (self.nodes_by_moves[new_set].layer - self.root.layer)%2 else -1
#             else:
#                 self.nodes_by_moves[new_set].current_eval = self.value_func(copied_board_rep)

#         elif self.nodes_by_moves[new_set].backprop_node.move_set != node.move_set:
#             self.nodes_by_moves[new_set].cross = True

#         return self.nodes_by_moves[new_set], self.nodes_by_moves[new_set].current_eval 

#     def selectLeaf(self) -> Tuple[Node,float]:
#         node = self.root
#         outval = None
#         made_a_node = False
#         size = len(self.nodes_by_moves)
#         while len(self.nodes_by_moves)==size:
#             next_node,outval = self.stepDown(node)
#             if next_node.cross:
#                 x,y = next_node.last_move[0], next_node.last_move[1]

#                 value_diff = node.future_eval_map[x,y] - next_node.current_eval
#                 if abs(value_diff) > self.REPAIR_DIFF:
#                     new_eval =  value_diff*node.future_visit_map[x,y] + next_node.current_eval
#                     node.future_eval_map[x,y]=max(self.min_eval,min(new_eval,self.max_eval))
#             # if next_node.terminal:
#             #     x,y = next_node.last_move[0], next_node.last_move[1]
#             #     node.future_eval_map[x,y]+=10                
#             #     return next_node,outval
#             node = next_node
#         return next_node,outval

#     def backProp(self, node:Node, value:float) -> None:
#         last_node = node.backprop_node
#         evalTarget = None
#         while last_node != None:
#             x,y = node.last_move[0],node.last_move[1]
 
#             if evalTarget != None:
#                 value_diff = last_node.future_eval_map[x,y] - evalTarget

#                 new_eval = value_diff*node.future_visit_map[x,y] + node.current_eval
#                 last_node.future_eval_map[x,y] = min(self.min_eval,max(new_eval, self.max_eval))
#             else:
#                 value = -value

#             last_node.updateFutures(x,y,value)
#             last_node.update(value)

#             if last_node.cross:
#                 evalTarget = -last_node.current_eval
#             else:
#                 evalTarget = None
                
#             node = last_node
#             last_node = last_node.backprop_node

#     def moveSelect(self) -> tuple[int,int]:
#         print("before:",self.root.layer)
#         print(self.nodes_ever)
#         index = torch.argmax(self.root.future_visit_map)
#         new_root = self.nodes_by_moves[
#                 frozenset((indexToTuple(index,(self.root.layer+1)%2),))|self.root.move_set]
#         # print(new_root)
        
#         print_tensor_probs(self.root)
#         print("after:",self.root.layer)
#         print_tensor_probs(new_root)
#         to_del =[]
#         for sparse_rep, node in self.nodes_by_moves.items():
#             if new_root.last_move not in sparse_rep or (node.backprop_node is not None and node.backprop_node.terminal):
#                 to_del.append(sparse_rep)
#             else:
#                 node.cross = False
#                 node.current_eval *= -1
#                 node.future_eval_map *= -1
#         for it in to_del:
#             self.nodes_by_moves.pop(it)
#         new_root.backprop_node = None
#         new_root.last_move = None
#         self.root = new_root

#         return int(index)//15 ,int(index)%15


