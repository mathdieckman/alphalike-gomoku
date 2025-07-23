import torch
import torch.nn.functional as torchF

ways_to_win = []

z = torch.zeros(2, 9, 9).to("cuda")

z_pad = torch.zeros(2, 27, 27).to("cuda")

for i in range(21):
    z_pad[0, 3+i, 3] = -1000000
    z_pad[0, 3, 3+i] = -1000000
    z_pad[0, 3+i, 23] = -1000000
    z_pad[0, 23, 3+i] = -1000000

    z_pad[1, 3+i, 3] = 1000000
    z_pad[1, 3, 3+i] = 1000000
    z_pad[1, 3+i, 23] = 1000000
    z_pad[1, 23, 3+i] = 1000000

z_pad_flip = torch.flip(z_pad, (0,)).to("cuda")

for i in range(5):
    channel = z.clone()
    for j in range(5):
        channel[0, 4, i+j] = 1
        channel[1, 4, i+j] = -1
    ways_to_win.append(channel)

for i in range(5):
    channel = z.clone()
    for j in range(5):
        channel[0, i+j, 4] = 1
        channel[1, i+j, 4] = -1
    ways_to_win.append(channel)

for i in range(5):
    channel = z.clone()
    for j in range(5):
        channel[0, i+j, i+j] = 1
        channel[1, i+j, i+j] = -1
    ways_to_win.append(channel)

for i in range(5):
    channel = z.clone()
    for j in range(5):
        channel[0, i+j, 8-(i+j)] = 1
        channel[1, i+j, 8-(i+j)] = -1
    ways_to_win.append(channel)

kernel = torch.stack(ways_to_win).to("cuda")
kernel_flip = torch.flip(kernel, (1,)).to("cuda")
cuda0 = torch.device('cuda:0')
bias = None
bias = torch.zeros(19, 19).to("cuda")
for x in range(19):
    for y in range(19):
        bias[x, y] = 1e-5*(min(x, 19-x) * min(y, 19-y))


def circ_check_if_won(state: torch.Tensor) -> bool:
    return bool(jit_circ_check_if_won(state, kernel, kernel_flip))


@torch.jit.script
def jit_circ_check_if_won(state: torch.Tensor, kernel, kernel_flip) -> torch.Tensor:
    padded = torchF.pad(state, (4, 4, 4, 4), mode='circular').to("cuda")
    return 5. <= torch.max(torch.conv2d(padded, kernel, padding="valid").max(), torch.conv2d(padded, kernel_flip, padding="valid").max())


def conv_check_if_won(state: torch.Tensor) -> bool:
    return 5 <= max(torch.conv2d(state, kernel, padding="same").max(), torch.conv2d(state, kernel_flip, padding="same").max())


def getWinOpsPlusBlocks(state: torch.Tensor) -> torch.Tensor:
    padded = torchF.pad(state, (4, 4, 4, 4)).to("cuda")
    wins = torch.sum(torch.square(torchF.relu(torch.conv2d(
        padded+z_pad, kernel, padding="valid"))), (0, 1,))

    blocks = torch.sum(torch.square(torchF.relu(torch.conv2d(
        padded+z_pad_flip, kernel_flip, padding="valid"))), (0, 1,))
    return (wins + blocks)*(1-state[0, 0, :, :])*(1-state[0, 1, :, :])


def policy(state: torch.Tensor) -> torch.Tensor:
    e = torch.exp(bias + getWinOpsPlusBlocks(state)/2)
    return e/e.sum()


def value(state: torch.Tensor) -> float:
    padded = torchF.pad(state, (4, 4, 4, 4))
    return torch.tanh((torch.conv2d(padded+z_pad, kernel, padding="same").max() - torch.conv2d(padded+z_pad_flip, kernel_flip, padding="same").max()))


def pv_func(state: torch.Tensor, b: int) -> tuple[torch.Tensor, float]:
    # handcrafted func is currently symetrical
    return policy(state), value(state)
    # return policy(state if not b else torch.flip(state,(1,))), value(state)


if __name__ == "__main__":
    torch.set_printoptions(linewidth=160, precision=2)
    b = torch.zeros(1, 2, 19, 19).to("cuda")
    b[0, 0, 7, 7] = 1
    print(policy(b))
    b[0,1,4,5]=1
    b[0,1,5,5]=1
    b[0,1,6,5]=1
    b[0,1,8,5]=1

    print(getWinOpsPlusBlocks(b))
    print(policy(b))

    print(getWinOpsPlusBlocks(torch.flip(b,(1,))))
    print(policy(torch.flip(b,(1,))))
    c = b.clone()
    d = b.clone()
    c[0,0,7,5]=1
    d[0,0,9,5]=1

    print(value(c))
    print(value(d))
