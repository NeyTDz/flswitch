import math

import numpy
import torch
import torch.nn as nn
from torch.nn import parameter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from network.batchcrypt import BatchCrypt
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

def _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    bn = nn.BatchNorm2d(num_features=out_channels)
    return nn.Sequential(conv, bn)

def _conv2d_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
    conv2d_bn = _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding)
    relu = nn.ReLU(inplace=True)
    layers = list(conv2d_bn.children())
    layers.append(relu)
    return nn.Sequential(*layers)

class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=False):
        super(_BasicBlock, self).__init__()
        self.down_sampler = None
        stride = 1
        if downscale:
            self.down_sampler = _conv2d_bn(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
            stride = 2
        self.conv_bn_relu1 = _conv2d_bn_relu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # don't relu here! relu on (H(x) + x)
        self.conv_bn2 = _conv2d_bn(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_out = nn.ReLU(inplace=True)
        # residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # residual = nn.BatchNorm2d(num_features=out_channels)
        # residual = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        if self.down_sampler:
            input = self.down_sampler(x)
        residual = self.conv_bn_relu1(x)
        residual = self.conv_bn2(residual)
        out = self.relu_out(input + residual)
        return out

class ResNet(nn.Module):
    def __init__(self, num_layer_stack=3):
        super(ResNet, self).__init__()
        self.conv1 = _conv2d_bn_relu(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.__make_layers(num_layer_stack, in_channels=16, out_channels=16, downscale=False)
        self.layer2 = self.__make_layers(num_layer_stack, in_channels=16, out_channels=32, downscale=True)
        self.layer3 = self.__make_layers(num_layer_stack, in_channels=32, out_channels=64, downscale=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=10 if DATASET=='CIFAR10' else 100)

    def __make_layers(self, num_layer_stack, in_channels, out_channels, downscale):
        layers = []
        layers.append(_BasicBlock(in_channels=in_channels, out_channels=out_channels, downscale=downscale))
        for i in range(num_layer_stack - 1):
            layers.append(_BasicBlock(in_channels=out_channels, out_channels=out_channels, downscale=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
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
        self.batchsize = 128
    
    @property
    def encryptor(self):
        return PlainCipher()
    
    def encrypt(self, x):
        return x
    
    def decrypt(self, x):
        return x

class PlainNetwork(ResNet if "CIFAR" in DATASET else MnistNet):
    def __init__(self, encry_method = 'plain',encryptor=None):
        super(PlainNetwork, self).__init__()
        self.device = torch.device(DEVICE)
        if encry_method == 'batchcrypt':
            self.cipher = BatchCrypt(encryptor=encryptor)
        elif encry_method == 'plain':
            self.cipher = PlainCipher()
        else:
            assert(0)
        self.encode_batch_size = self.cipher.batch_size
        self.encrypt_zeros = self.cipher.encrypt(numpy.zeros(self.encode_batch_size))
        self.flat_grad = torch.tensor([]).cpu().numpy()

    @property
    def encryptor(self):
        return self.cipher.encryptor
    
    def get_raw_grads(self,clipping=1.0):
        '''Get flat grad of all gradients
            Note that self.falt_grad updates here
            Return array:flat_grad
        '''
        #self.flat_grad = torch.tensor([]).cpu()
        device = self.device
        flat_grad = torch.tensor([]).to(device)
        for name, params in self.named_parameters():
            flat_grad = torch.cat([flat_grad, torch.flatten(params.grad.clamp(-clipping, clipping))], dim=0)
        flat_grad = flat_grad.cpu().numpy().flatten()
        self.flat_grad = flat_grad
        return flat_grad

    def pack_grad_with_anchors(self,part_grad = [],border = []):
        '''Encrypt part of anchors
            zeros need not encryption, append self.encrypt_zeros

            Return list:packed_res
        '''
        if len(part_grad) == 0 or len(border) == 0:
            assert 0
        [begin,end] = border
        batch_size = self.encode_batch_size
        packed_res = []
        for i in range(math.ceil(len(self.flat_grad)/batch_size)):
            if i*batch_size < begin or i*batch_size >= end:
                ct = self.encrypt_zeros
            else:
                ct = self.cipher.encrypt(part_grad[(i*batch_size-begin):((i+1)*batch_size-begin)])
            packed_res.append(ct)
        return packed_res

    def pack_grad(self, clipping=1.0, return_size=False, raw_grad = []):
        '''
        if raw_grad, this func only as an encryptor tool to encode the raw_grad
        '''
        device = self.device
        if len(raw_grad):
            flat_grad = raw_grad
        else:
            # self.flat_grad updates here
            flat_grad = torch.tensor([]).to(device)
            for name, params in self.named_parameters():
                flat_grad = torch.cat([flat_grad, torch.flatten(params.grad.clamp(-clipping, clipping))], dim=0)
            flat_grad = flat_grad.cpu().numpy()
            self.flat_grad = flat_grad
        packed_res = []
        for i in range(math.ceil(len(flat_grad)/self.encode_batch_size)):
            batch_grads = flat_grad[i * self.encode_batch_size:(i + 1) * self.encode_batch_size]
            if len(batch_grads) < self.encode_batch_size:
                batch_grads = np.append(batch_grads,np.zeros(self.encode_batch_size-len(batch_grads)))
            ct = self.cipher.encrypt(batch_grads)
            packed_res.append(ct)
        if return_size:
            return packed_res, len(flat_grad)
        else:
            return packed_res

    def unpack_grad(self, packed_grad, packed_size):
        grad_dict = {}
        gradients = numpy.array([])
        for ct in packed_grad:
            flat_grad = self.cipher.decrypt(ct)
            gradients = numpy.concatenate([gradients, flat_grad])
        gradients = gradients[:packed_size]
        start = 0
        for name, params in self.named_parameters():
            param_size = 1
            for s in params.size():
                param_size *= s
            grad_dict[name] = torch.from_numpy(gradients[start:start+param_size]).reshape(params.size())
            start += param_size
        return grad_dict
    
    def unpack_grad_with_residues(self,raw_anchors,packed_grad,residues,sparse_index,stand_power,remask):
        '''
        Decrypt global enc_anchors
        Add remask to raw order
        Complement sparse data(optional)
        Add global anchors and residues to global flat gradients
        Trans flat gradients to grad_dict

        Return: dict:grad_dict
        '''
        grad_dict = {}
        anchors = numpy.array([])
        for ct in packed_grad:
            anch = self.cipher.decrypt(ct)
            anchors = numpy.concatenate([anchors, anch])
        anchors = anchors[remask]
        if ADDSPARSE:  # if ADDSPARSE, use clients' own anchors complement sparse data
            anchors[sparse_index] = raw_anchors[sparse_index]
        gradients = anchors * (10**-stand_power)
        gradients[:len(residues)] += residues
        start = 0
        for name, params in self.named_parameters():
            param_size = 1
            for s in params.size():
                param_size *= s
            grad_dict[name] = torch.from_numpy(gradients[start:start+param_size]).reshape(params.size())
            start += param_size
        return grad_dict

    def plain_grad_process(self,gradients):
        grad_dict = {}
        start = 0
        for name, params in self.named_parameters():
            param_size = 1
            for s in params.size():
                param_size *= s
            grad_dict[name] = torch.from_numpy(gradients[start:start+param_size]).reshape(params.size())
            start += param_size
        return grad_dict    

    def plain_anchor_residues_process(self,anchors,residues,stand_power,raw_anchors,sparse_index):
        grad_dict = {}
        gradients = anchors * (10**-stand_power)
        gradients[sparse_index] = anchors[sparse_index]
        gradients[:len(residues)] += residues
        start = 0
        for name, params in self.named_parameters():
            param_size = 1
            for s in params.size():
                param_size *= s
            grad_dict[name] = torch.from_numpy(gradients[start:start+param_size]).reshape(params.size())
            start += param_size
        return grad_dict  
    
    def test(self,test_loader,lossf):
        device = self.device
        loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)
                output = self(data)
                loss += lossf(output, targets)
                correct += (output.argmax(1) == targets).sum()
                total += data.size(0)
        loss = loss.item()/len(test_loader)
        acc = correct.item()/total
        return loss,acc
    
    def save_model(self,path):
        torch.save(self.state_dict(),path)


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
