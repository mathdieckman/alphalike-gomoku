import torch
import pickle
from base.board import Board
from base.self_play_loop import SelfPlay
from ai.cpuct_player import CPuct_Player
from nets.alphaeleven import Net, Symmetric
import glob
# import torch_tensorrt
import torch
import torch.nn.utils.parametrize as parametrize

# torch.jit.set_fusion_strategy([("STATIC",10)])


def get_data(name, number, its, games, nodes):
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cuda"
        model = Net(256)
        model.load_state_dict(torch.load(
            "./nets/weights/model_weights"+str(number)+".pth"))
        model.to(device)
        examp = torch.ones(1, 2, 15, 15).to("cuda")
        m = torch.jit.trace(model, examp)

        model2 = Net(256)
        model2.load_state_dict(torch.load(
            "./nets/weights/model_weights"+str(number-2)+".pth"))
        model2.to(device)
        m2 = torch.jit.trace(model2, examp)

        # traced = torch.jit.trace(model, examp,)
        # opt_mod=torch.jit.freeze(traced.eval())

        # opt_mod = torch.jit.optimize_for_inference(traced.eval())
        # trt_ts_module = torch_tensorrt.compile(traced_cell,
        #         inputs = [examp],
        #         enabled_precisions = {torch.half}) # Run with FP16)

        # torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript

        # model2 = Net(256)
        # model2.load_state_dict(torch.load("./nets/weights/model_weights"+str(max(number-its,0))+".pth"))
        # model2.to(device)

        def pv(x, y):
            u, v = m(torch.flip(x, (1,))if y else x)
            return u.view((15, 15)), v

        def pv2(x, y):
            u, v = m2(torch.flip(x, (1,))if y else x)
            return u.view((15, 15)), v

        # while len(glob.glob('./train/data/*'))<batch_size:
            # i+=1
        board = Board(15)
        a = CPuct_Player(board, pv, False, nodes, training=True,
                         record_data=True, self_play=False, verbose=True, name=name+"_b1")
        b = CPuct_Player(board, pv2, True, nodes, training=True,
                         record_data=True, self_play=False, verbose=True, name=name+"_w1")
        g = SelfPlay(a, b, board)
        g.play()
        black_wins = (torch.count_nonzero(board.white) +
                      torch.count_nonzero(board.black)) % 2
        torch.save(black_wins, "./train/hack_flag/"+name+str(0))

        board1 = Board(15)
        a1 = CPuct_Player(board, pv, True, nodes, training=True,
                          record_data=True, self_play=False, verbose=True, name=name+"_b2")
        b1 = CPuct_Player(board, pv2, False, nodes, training=True,
                          record_data=True, self_play=False, verbose=True, name=name+"_w2")
        g1 = SelfPlay(b1, a1, board1)
        g1.play()
        white_wins = not (torch.count_nonzero(board1.white) +
                          torch.count_nonzero(board1.black)) % 2
        torch.save(black_wins, "./train/hack_flag/"+name+str(1))

    return black_wins + white_wins


if __name__ == "__main__":

    get_data(15, 256*3//2, 'A')
    # p1 = Process(target=get_data, args=(15,24,'A'))
    # # p2 = Process(target=get_data, args=(15,24,'B'))
    # set_start_method('spawn')
    # p1.start()
    # p2.start()
