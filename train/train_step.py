from numpy import dtype, zeros_like
import torch
from nets.alphaeleven import Net
import train.dataset as dataset
from torch.utils.data import DataLoader
import math


device = "cuda" if torch.cuda.is_available() else "cuda"

# def mobius(x):
# return x/(7-6*x) # smoothly maps 0 to 0, 1 to 1, and 0.5 to 0.125
# return x


def my_loss(pred_dist, pred_value, evals, player, winner, writer, run):
    pred_as_log_probs = torch.log(torch.relu(
        pred_dist)/(torch.relu(pred_dist).sum().sum()+1e-9)+1e-4)
    target_as_probs = ((evals+1)/2)/(1e-9+((evals+1)/2).sum().sum())+1e-4

    policy_loss = 225 * \
        (winner.shape[0]) * \
        torch.nn.functional.kl_div(pred_as_log_probs, target_as_probs)
    eval_loss = torch.square(pred_value.reshape(
        (-1,))-(1.-2.*(player.ne(winner)))).mean()
    # print(pred_value.shape,winner.shape,player.shape)
    # print(torch.stack((pred_value.reshape((-1,)),winner,player,torch.square( pred_value.reshape((-1,))-(1-2*(player.ne(winner))))),0))

    writer.add_scalar('policy_loss', policy_loss, run)
    writer.add_scalar('eval_loss', eval_loss, run)

    return policy_loss+eval_loss


def train(lr, batch_size, run, writer, optimizer):
    model = Net(256)
    model = model.to(device)
    batches = 0
    # torch.autograd.set_detect_anomaly(True)
    for board_rep, policy, player, winner in DataLoader(dataset.decision_point(0, 0, 10000), batch_size=batch_size):
        model.load_state_dict(torch.load(
            "./nets/weights/model_weights"+str(run)+".pth"))
        batches += 1
        player = player.to('cuda')
        board_rep.requires_grad_ = True
        optimizer.zero_grad()
        pred = model(board_rep)
        pred = pred[0].reshape(-1, 1, 15, 15), pred[1]
        loss = my_loss(pred[0], pred[1], policy, player, winner, writer, run)
        print(player)
        print(winner)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), )
        optimizer.step()  # This will update the shared parameters

        conv_total = 0
        conv_var = 0
        conv_the_max = 0
        conv_count = 0
        total = 0
        var = 0
        the_max = 0
        count = 0
        for n, p in model.named_parameters():
            if p.grad is not None:
                total += p.grad.abs().mean()
                var += torch.square(p.grad.abs()).mean()
                the_max = max(the_max, p.grad.abs().max())
                count += 1
            if "conv" in n and p.grad is not None:
                conv_total += p.grad.abs().mean()
                conv_var += torch.square(p.grad.abs()).mean()
                conv_the_max = max(conv_the_max, p.grad.abs().max())

        writer.add_scalar('batch size', board_rep.shape[0], run)

        writer.add_scalar('grad_mean', total/count, run)
        writer.add_scalar('grad_std', torch.sqrt(var/count), run)
        writer.add_scalar('grad_max', the_max, run)

        writer.add_scalar('conv_grad_mean', conv_total/count, run)
        writer.add_scalar('conv_grad_std', torch.sqrt(
            conv_var/conv_count), run)
        writer.add_scalar('conv_grad_max', conv_the_max, run)
        torch.save(model.state_dict(),
                   "./nets/weights/model_weights"+str(run)+".pth")
