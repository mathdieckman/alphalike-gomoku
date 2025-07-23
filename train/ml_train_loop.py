import torch
import time
from train.train_data_gen import get_data
from train_step import train
# from tester import test
# from train.tester import test
from nets.alphaeleven import Net
import os
import glob
from torch.utils.tensorboard import SummaryWriter
# import torchvision
# import torchvision.transforms as transforms
writer = SummaryWriter('logs/current_run')
# default `log_dir` is "runs" - we'll be more specific here

# blank_slate = True
# if blank_slate:
#     os.remove('./nets/weights/model_weights2.pth')

run = 386
lr = 0.02
games = 1
batch_size = 9999

w = 0


# img_grid = torchvision.utils.make_grid()
# a,b,c,d=iter(DataLoader(dataset.decision_point(0,0,10000,size),batch_size=32)).next()
# model = Net(256,15).to("cuda")
# writer
# .add_graph(model,a)
# writer.close()
# raise
alphabet = "abcdefghijklmnopqrstuvwxyz1234567890"
model = Net(256)

wins = 0
its = 0
# model.load_state_dict(torch.load("./nets/weights/init.pth"))
torch.save(model.state_dict(), './nets/weights/init.pth')
torch.save(model.state_dict(), './nets/weights/model_weights0.pth')

optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)


while True:
    # try:
    model = Net(256)
    model.load_state_dict(torch.load(
        "./nets/weights/model_weights"+str(run)+".pth"))
    run += 1
    print("run ", run)
    torch.save(model.state_dict(),
               './nets/weights/model_weights'+str(run)+'.pth')
    files = glob.glob('./train/data/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./hack_flag/data/*')
    for f in files:
        os.remove(f)
    for i in range(1, games+1):
        wins += get_data(alphabet[i], run, i, games, 2500)
    train(lr, batch_size, run, writer, optimizer)
    print("train done", time.ctime())
    # lr *= 0.98
    # break
    # except:
    #     pass
