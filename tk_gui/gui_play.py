import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkFont

from numpy import zeros
from base.board import Board
from ai.random import RandomGuesser
from ai.cpuct_player import CPuct_Player
from nets.alphaeleven import Net, Symmetric
import torch
from base import conv_calcs
import torch.nn.utils.parametrize as parametrize
import torch.jit

global game  # runs game logic
global start, squares  # buttons
global player_color, ai  # vars controlled by dropdown
global white_img, black_img  # images of stones




class newGame:
    def __init__(self, root, ai, size, ai_is_white):
        self.b = Board(size)
        self.ai = None
        self.root = root
        self.ai_is_white = ai_is_white
        self.ones = torch.zeros((size, size))+1

        if ai == "Random":
            self.ai = RandomGuesser(self.b)
        elif ai == "MCTS":
            self.ai = CPuct_Player(
                self.b, conv_calcs.pv_func, self.ai_is_white, 6000, verbose=True)
        elif ai == "AlphaZero":

            model = Net(256)
            model.to("cuda")

            model.load_state_dict(torch.load(
                "./nets/weights/model_weights397.pth"))

            def pv(x, y):
                u, v = model(torch.flip(x, (1,)) if y else x)
                return u.view((19, 19)), v

            self.ai = CPuct_Player(
                self.b, pv, self.ai_is_white, 1000, verbose=True)
        if not ai_is_white:
            self.b.push(*self.ai.move())

    def play_move(self, x, y):
        if not self.b.validate_move(x, y):
            return "bad move", (-1, -1)
        self.b.push(x, y)
        self.ai.inform(x, y)
        if self.b.check_if_won() and not self.b.check_if_full():
            return "you win", (-1, -1)
        if self.b.check_if_full():
            return "tie game", (-1, -1)
        move = self.ai.move()
        self.b.push(*move)
        if self.b.check_if_won() and not self.b.check_if_full():
            return "you lose", move
        if self.b.check_if_full():
            return "tie game", move
        return "", move


def makeNewGame(root):
    global game
    global squares, start
    global ai, player_color

    game = newGame(root, ai.get(), 19, player_color.get() != "White")
    for row in squares:
        for square in row:
            square["image"] = tk.PhotoImage()
    if player_color.get() == "White":
        # if the ai is black then the game runner will have already had it move
        x, y = (game.b.black > 0.99).nonzero(as_tuple=True)
        squares[x][y]["image"] = black_img
    start.keep_blinking = False


class BlinkingButton(tk.Button):
    def __init__(self, root, **kwargs):
        super(BlinkingButton, self).__init__(root, **kwargs)
        self.keep_blinking = True
        self.root = root

    def blink(self):
        if not self.keep_blinking:
            self.config(background="grey")
            return

        current_color = self.cget("background")
        next_color = "grey" if current_color == "white" else "white"
        self.config(background=next_color)
        self.root.after(1000, self.blink)


class BoardSquare(tk.Button):
    def __init__(self, x, y, **kwargs):
        super(BoardSquare, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.configure(bg='gray')
        self.bind("<ButtonRelease-1>", lambda _: self.move_cycle())

    def move_cycle(self):
        global game
        global squares, start
        global player_color
        global white_img, black_img

        if not game:
            return

        prev_image = self["image"]
        if player_color.get() == "White":
            self["image"] = white_img
        else:
            self["image"] = black_img
        self.update()
        text, move = game.play_move(self.x, self.y)
        if text == "bad move":
            self["image"] = prev_image
            return

        elif move != (-1, -1):
            x, y = move
            if player_color.get() == "White":
                squares[x][y]["image"] = black_img
            else:
                squares[x][y]["image"] = white_img

        if text == "you win" or text == "you lose" or text == "tie game":
            start.keep_blinking = True
            start.blink()
            game = None


if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='gray')

    white_img = tk.PhotoImage(file="./img/white.png")
    black_img = tk.PhotoImage(file="./img/black.png")

    root.title("Gomoku")

    default_font = tkFont.Font(family="system", size=30)
    root.option_add("*Font", default_font)
    style = ttk.Style()
    style.configure('TCombobox', arrowsize=25,
                    fieldbackground='grey', background='grey')
    style.configure('Vertical.TScrollbar', width=22,
                    arrowsize=22, color='grey')

    choices = ['Black', 'White']
    player_color = tk.StringVar(root)
    player_color.set('Black')

    choices2 = ['AlphaZero', 'MCTS', 'Random']
    ai = tk.StringVar(root)
    ai.set('AlphaZero')

    tk.Label(text="color:", font=default_font, bg='grey').grid(
        row=0, column=0, columnspan=3, sticky="NSEW")

    color_dropdown = ttk.Combobox(
        root, values=choices, width=5, font=default_font, textvariable=player_color)
    color_dropdown.grid(row=0, column=5, columnspan=5, sticky="NSEW")

    tk.Label(text="ai:", font=default_font, bg='grey').grid(
        row=0, column=10, columnspan=2)

    ai_dropdown = ttk.Combobox(
        root, values=choices2, width=5, font=default_font, textvariable=ai)
    ai_dropdown.grid(row=0, column=12, columnspan=5, sticky="NSEW")

    start = BlinkingButton(root, text="start", bg='grey',
                           font=default_font, relief="solid")
    start.blink()
    start.bind("<ButtonRelease-1>", lambda _: makeNewGame(root))

    start.grid(row=0, column=17, columnspan=4, sticky="NSEW")

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    squares = []
    for i in [int(x+1) for x in range(1, 20)]:
        cols = []
        tk.Label(text=str(21-i), font=tkFont.Font(family="system", size=10),
                 master=root, bg='grey').grid(row=i, column=0)
        tk.Label(text=str(21-i), font=tkFont.Font(family="system", size=10),
                 master=root, bg='grey').grid(row=i, column=16)
        
        tk.Label(text=str(i-1), font=tkFont.Font(family="system", size=10),
                 master=root, bg='grey').grid(row=1, column=i-1)
        
        tk.Label(text=str(i-1), font=tkFont.Font(family="system", size=10),
                 master=root, bg='grey').grid(row=21, column=i-1)
        
        # tk.Label(text=alphabet[i-2], font=tkFont.Font(family="system", size=10),
        #             master=root, bg='grey').grid(row=1, column=i-1)
        # tk.Label(text=alphabet[i-2], font=tkFont.Font(family="system", size=10),
        #             master=root, bg='grey').grid(row=20, column=i-1)

        for j in [int(x) for x in range(19)]:


            cols.append(BoardSquare(i-2, j, image=tk.PhotoImage(),
                        width=60, height=60, relief="ridge"))
            cols[-1].grid(row=i, column=j+1, sticky="NSEW")



        squares.append(cols)

    root.columnconfigure(tuple(range(14)), weight=1)
    root.rowconfigure(tuple(range(1, 14)), weight=1)
    root.rowconfigure((0,), weight=1)

    with torch.no_grad():
        root.mainloop()
