import math

import numpy
import torch
import torch.nn as nn
from torch.nn import parameter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_params import *
device = torch.device(DEVICE)

class Conv2Linear(nn.Module):
    def __init__(self):
        super(Conv2Linear, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear2Conv(nn.Module):
    def __init__(self, out_ch, width, height):
        super(Linear2Conv, self).__init__()
        self.out_ch = out_ch
        self.width = width
        self.height = height

    def forward(self, x):
        return x.reshape(x.size()[0], self.out_ch, self.width, self.height)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            Conv2Linear(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class PlainCipher(object):
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def encryptor(self):
        return PlainCipher()
    
    def encrypt(self, x):
        return x
    
    def decrypt(self, x):
        return x

class PlainNetwork(Net):
    def __init__(self, encryptor=None):
        super(PlainNetwork, self).__init__()
        if encryptor:
            self.cipher = encryptor
        else:
            self.cipher = PlainCipher()
        self.encode_batch_size = 128

    @property
    def encryptor(self):
        return self.cipher.encryptor

    def pack_grad(self, clipping=1.0, return_size=False):
        packed_res = []
        flat_grad = torch.tensor([]).to(device)
        for name, params in self.named_parameters():
            flat_grad = torch.cat([flat_grad, torch.flatten(params.grad.clamp(-clipping, clipping))], dim=0)
        flat_grad = flat_grad.to(device).numpy()
        for i in range(math.ceil(len(flat_grad)/self.encode_batch_size)):
            ct = self.cipher.encrypt(flat_grad[i * self.encode_batch_size:(i + 1) * self.encode_batch_size])
            packed_res.append(ct)
        if return_size:
            return packed_res, len(flat_grad)
        else:
            return packed_res

    def unpack_grad(self, packed_grad):
        grad_dict = {}
        gradients = numpy.array([])
        for ct in packed_grad:
            flat_grad = self.cipher.decrypt(ct)
            gradients = numpy.concatenate([gradients, flat_grad])
        start = 0
        for name, params in self.named_parameters():
            param_size = 1
            for s in params.size():
                param_size *= s
            grad_dict[name] = torch.from_numpy(gradients[start:start+param_size]).reshape(params.size())
            start += param_size
        return grad_dict


if __name__ == "__main__":
    EPOCH = 10
    BATCH_SIZE = 512
    LR = 1E-3
    svd_k = 10
    import random
    import numpy as np

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


    set_seed(999)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    train_file = datasets.FashionMNIST(
        root='../dataset/',
        train=True,
        transform=transform,
        download=True
    )
    test_file = datasets.FashionMNIST(
        root='../dataset/',
        train=False,
        transform=transform
    )

    train_data = train_file.data
    train_targets = train_file.targets
    print(train_data.size)
    print(len(train_targets))

    train_loader = DataLoader(
        dataset=train_file,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_file,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # model = Net().cpu()
    model = PlainNetwork().cpu()
    optim = torch.optim.Adam(model.parameters(), LR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.25, patience=10, verbose=True, min_lr=1E-5)
    lossf = nn.CrossEntropyLoss()
    last_pack = None
    for epoch in range(EPOCH):
        max_diff = 0.0
        for step, (data, targets) in enumerate(train_loader):
            optim.zero_grad()
            data = data.cpu()
            targets = targets.cpu()
            output = model(data)
            loss = lossf(output, targets)
            loss.backward()
            x, pack_size = model.pack_grad(return_size=True)
            y = model.unpack_grad(x)
            for name, param in model.named_parameters():
                diff = torch.abs(y[name].cpu() - param.grad) 
                max_diff = torch.max(diff) if torch.max(diff) > max_diff else max_diff
            for k, v in model.named_parameters():
                v.grad = y[k].cpu().type(dtype=v.grad.dtype)
            optim.step()
            if step % 10 == 0:
                loss = 0
                total = 0
                correct = 0
                with torch.no_grad():
                    for data, targets in test_loader:
                        data = data.cpu()
                        targets = targets.cpu()
                        output = model(data)
                        loss += lossf(output, targets)
                        correct += (output.argmax(1) == targets).sum()
                        total += data.size(0)
                loss = loss.item()/len(test_loader)
                acc = correct.item()/total
                # scheduler.step(loss)
                print("epoch: {}, step: {}, loss: {}, acc: {}, max_diff: {}".format(epoch, step, loss, acc, max_diff))
