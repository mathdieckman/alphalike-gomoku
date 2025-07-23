import torch

# basic gomoku rules and logging
# the equivalent that the gpu players use are contained in conv_calcs and logic in the mcts implementation


class Board:
    def __init__(self, size):
        self.size = size
        self.moves_made = 0
        self.white = torch.zeros([size, size]).to("cuda")
        self.black = torch.zeros([size, size]).to("cuda")
        self.white_to_move = False
        self.last_move_made = (-1, -1)

    def push_whole_log(self, log):
        for x, y in log:
            self.push(x, y)

    def push(self, x, y):
        if self.white_to_move:
            self.white[x, y] = True
        else:
            self.black[x, y] = True
        self.white_to_move = not self.white_to_move
        self.moves_made += 1
        self.last_move_made = (x, y)

    def validate_move(self, x, y):
        if self.white[x, y] or self.black[x, y]:
            return False
        else:
            return True

    def validate_board(self):
        is_valid = True
        for x in range(self.size):
            for y in range(self.size):
                if self.white[x, y] and self.black[x, y]:
                    is_valid = False
                    print("double occupation on square (" +
                          str(x) + ", " + str(y) + ")")
        return is_valid

    def check_if_full(self):
        return self.moves_made == self.size * self.size

    def check_if_won(self):
        needed = 5
        x, y = self.last_move_made
        in_a_row = 0

        x_min = max(x-needed+1, 0)
        y_min = max(y-needed+1, 0)
        x_max = min(x+needed-1, self.size-1)
        y_max = min(y+needed-1, self.size-1)

        xy_x_min = x-min(x-x_min, y-y_min)
        xy_y_min = y-min(x-x_min, y-y_min)
        xy_x_max = x+min(x_max-x, y_max-y)
        xy_y_max = y+min(x_max-x, y_max-y)

        yx_x_min = x-min(x-x_min, y_max-y)
        yx_x_max = x+min(x_max-x, y-y_min)

        yx_y_min = y+min(x-x_min, y_max-y)
        yx_y_max = y-min(x_max-x, y-y_min)

        white_moved_last = not self.white_to_move
        relevant_board = None
        if white_moved_last:
            relevant_board = self.white
        else:
            relevant_board = self.black

        for u in range(x_min, x_max+1):
            if relevant_board[u, y]:
                in_a_row += 1
            else:
                in_a_row = 0
            if in_a_row == needed:
                return True
        in_a_row = 0

        for v in range(y_min, y_max+1):
            if relevant_board[x, v]:
                in_a_row += 1
            else:
                in_a_row = 0
            if in_a_row == needed:
                return True
        in_a_row = 0

        for u, v in zip(range(xy_x_min, xy_x_max+1), range(xy_y_min, xy_y_max+1)):
            if relevant_board[u, v]:
                in_a_row += 1
            else:
                in_a_row = 0
            if in_a_row == needed:
                return True
        in_a_row = 0

        for u, v in zip(range(yx_x_min, yx_x_max+1), range(yx_y_min, yx_y_max-1, -1)):
            if relevant_board[u, v]:
                in_a_row += 1
            else:
                in_a_row = 0
            if in_a_row == needed:
                return True
        in_a_row = 0

        return False

    def slow_check_if_won(self):
        for x in range(self.size):
            for y in range(self.size-4):
                if self.white[x, y] and self.white[x, y+1] and self.white[x, y+2] and self.white[x, y+3] and self.white[x, y+4]:
                    return True
                if self.black[x, y] and self.black[x, y+1] and self.black[x, y+2] and self.black[x, y+3] and self.black[x, y+4]:
                    return True

        for x in range(self.size-4):
            for y in range(self.size):
                if self.white[x, y] and self.white[x+1, y] and self.white[x+2, y] and self.white[x+3, y] and self.white[x+4, y]:
                    return True
                if self.black[x, y] and self.black[x+1, y] and self.black[x+2, y] and self.black[x+3, y] and self.black[x+4, y]:
                    return True

        for x in range(self.size-4):
            for y in range(self.size-4):
                if self.white[x, y] and self.white[x+1, y+1] and self.white[x+2, y+2] and self.white[x+3, y+3] and self.white[x+4, y+4]:
                    return True
                if self.black[x, y] and self.black[x+1, y+1] and self.black[x+2, y+2] and self.black[x+3, y+3] and self.black[x+4, y+4]:
                    return True

        for x in range(self.size-4):
            for y in range(4, self.size):
                if self.white[x, y] and self.white[x+1, y-1] and self.white[x+2, y-2] and self.white[x+3, y-3] and self.white[x+4, y-4]:
                    return True
                if self.black[x, y] and self.black[x+1, y-1] and self.black[x+2, y-2] and self.black[x+3, y-3] and self.black[x+4, y-4]:
                    return True

        return False

    def __repr__(self):
        out = ""
        for x in range(self.size):
            for y in range(self.size):
                if self.white[x, y]:
                    out += "X"
                elif self.black[x, y]:
                    out += "O"
                else:
                    out += "\u00B7"
            out += "\n"
        return out
