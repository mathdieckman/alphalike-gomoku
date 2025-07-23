import torch.nn as nn
import torch

center_weight = torch.zeros((15, 15)).to("cuda")

padding_mode = "circular"


class Symmetric(nn.Module):
    def forward(self, X):
        return (X + X.transpose(-1, -2) +
                torch.flip(X, (-1,)) + torch.flip(X.transpose(-1, -2), (-1,)) +
                torch.flip(X, (-2,)) + torch.flip(X.transpose(-1, -2), (-2,)) +
                torch.flip(X, (-1, -2,)) + torch.flip(X.transpose(-1, -2), (-1, -2)))/8


class SqueezeExcite(nn.Module):
    def __init__(self, filters: int, se_filters: int):
        super(SqueezeExcite, self).__init__()
        self.sig = nn.Sigmoid()
        self.filters = filters
        self.se_filters = se_filters
        self.linear1 = nn.Linear(2*filters, se_filters, bias=False)
        self.linear2 = nn.Linear(se_filters, 2*filters, bias=False)
        self.ave = nn.AvgPool2d((15, 15))
        self.max = nn.MaxPool2d((15, 15))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        squeezed = torch.cat((self.ave(x), self.max(x)), dim=-1)
        squeezed = self.linear1(squeezed.reshape(-1, 2*self.filters))
        self.relu(squeezed)
        squeezed = self.linear2(squeezed)
        squeezed = self.sig(squeezed)
        out = x * squeezed.narrow(1, 0, self.filters).reshape((-1, self.filters, 1, 1)).expand_as(x) +\
            squeezed.narrow(1, self.filters, self.filters).reshape(
                (-1, self.filters, 1, 1)).expand_as(x)
        return out


class SqueezeExciteResidual(nn.Module):
    def __init__(self, filters: int):
        super(SqueezeExciteResidual, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(
            filters, filters, 3, padding='same', bias=False, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(filters, affine=False)
        self.conv2 = nn.Conv2d(
            filters, filters, 3, padding='same', bias=False, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(filters, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(self.filters, self.filters//4)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        self.relu2(out)
        out = self.se(out)
        out += identity
        return out


class PolicyHead(nn.Module):
    def __init__(self, filters: int):
        super(PolicyHead, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(
            filters, filters, 3, bias=False, padding='same', padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(filters, 1, 3, bias=False,
                               padding='same', padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ValueHead(nn.Module):
    def __init__(self, filters: int):
        super(ValueHead, self).__init__()
        self.filters = filters
        self.linear1 = nn.Linear(2*filters, 32)
        self.linear2 = nn.Linear(32, 1)
        self.ave = nn.AvgPool2d((15, 15))
        self.max = nn.MaxPool2d((15, 15))
        self.relu = nn.ReLU(inplace=True)
        self.tnh = nn.Tanh()

    def forward(self, x):
        y = torch.cat((self.ave(x), self.max(x)), dim=-1)
        y = self.linear1(y.reshape((-1, 2*self.filters)))
        y = self.relu(y)
        y = self.linear2(y)
        return self.tnh(y)


class Net(nn.Module):
    def __init__(self, filters: int):
        super(Net, self).__init__()
        self.filters = filters
        self.conv0 = nn.Conv2d(2, filters, 3, padding='same',
                               bias=False, padding_mode=padding_mode)
        self.res1 = SqueezeExciteResidual(filters)
        self.res2 = SqueezeExciteResidual(filters)
        self.res3 = SqueezeExciteResidual(filters)
        self.res4 = SqueezeExciteResidual(filters)
        self.res5 = SqueezeExciteResidual(filters)
        self.res6 = SqueezeExciteResidual(filters)
        self.res7 = SqueezeExciteResidual(filters)
        self.res8 = SqueezeExciteResidual(filters)
        self.res9 = SqueezeExciteResidual(filters)
        self.res10 = SqueezeExciteResidual(filters)
        self.res11 = SqueezeExciteResidual(filters)
        self.res12 = SqueezeExciteResidual(filters)
        self.res13 = SqueezeExciteResidual(filters)
        self.res14 = SqueezeExciteResidual(filters)
        self.res15 = SqueezeExciteResidual(filters)
        self.res16 = SqueezeExciteResidual(filters)
        self.res17 = SqueezeExciteResidual(filters)
        self.res18 = SqueezeExciteResidual(filters)
        self.res19 = SqueezeExciteResidual(filters)
        self.res20 = SqueezeExciteResidual(filters)
        self.pol = PolicyHead(filters)
        self.val = ValueHead(filters)
        self.flat = torch.nn.Flatten()
        self.max = torch.nn.LogSoftmax(1)

    def forward(self, x):
        z = x[:, 0, :, :] + x[:, 1, :, :]
        q = 1-z
        q = q.reshape(-1, 1, 15, 15)
        y = self.conv0(x)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)
        y = self.res10(y)
        y = self.res11(y)
        y = self.res12(y)
        y = self.res13(y)
        y = self.res14(y)
        y = self.res15(y)
        y = self.res16(y)
        y = self.res17(y)
        y = self.res18(y)
        y = self.res19(y)
        y = self.res20(y)
        out1 = self.pol(y)
        out2 = self.val(y)
        out1 = out1*q  # removing illegal moves here is convenient
        out1 = self.flat(out1)
        out1 = self.max(1.75*out1)
        out1 = torch.exp(out1)
        out1 = out1.reshape(-1, 1, 15, 15)
        return out1, out2


if __name__ == "__main__":
    net = Net(256)
    print(net)
    [print(layer.size()) for layer in net.parameters()]

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
