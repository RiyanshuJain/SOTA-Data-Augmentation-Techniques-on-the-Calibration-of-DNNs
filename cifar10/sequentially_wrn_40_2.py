import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.utils.data.dataset import Dataset
from torch.optim import SGD, lr_scheduler
from torchvision.datasets import CIFAR10
from torchvision import transforms

import os
import math
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageOps
import warnings
warnings.filterwarnings('ignore')

torch.cuda.is_available()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""## <font color = 'lightblue'>**Wide ResNet Model**</font>"""

__all__ = ['wrn']

class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.avgpool = nn.AvgPool2d(8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def distill_seq(self):
        feat_m = nn.ModuleList([])  
        #feat_m.append(self.conv1)       
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(nn.Sequential(
            self.block3, self.bn1, self.relu))
        feat_m.append(nn.Sequential(
            self.avgpool,
            #Rearrange('b h w -> (b h w) 1'),
            Flatten(),
            self.fc))
        return feat_m

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        # f3 = out
        out = self.relu(self.bn1(out))
        f3 = out
        out = self.avgpool(out)
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)
        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)
            return [f0, f1, f2, f3, f4], out
        else:
            return out


def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


def wrn_40_2(**kwargs):
    model = WideResNet(depth=40, widen_factor=2, **kwargs)
    return model


# Cutmix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# loss for mixup and cutmix
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def save_checkpoint(state, path):
    # Save checkpoint.
    print('Saving..')
    torch.save(state, path+'/best_ckpt')
    print('Saved model to {}'.format(path))

def compute_calibration_metrics(num_bins=100, net=None, loader=None, device='cuda'):
    """
    Computes the calibration metrics ECE along with the acc and conf values
    :param num_bins: 100 is used
    :param net: trained network
    :param loader: dataloader for the dataset
    :param device: cuda or cpu
    :return: ECE, acc, conf
    """
    acc_counts = [0 for _ in range(num_bins+1)]
    conf_counts = [0 for _ in range(num_bins+1)]
    overall_conf = []
    n = float(len(loader.dataset))
    counts = [0 for i in range(num_bins+1)]
    net.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images, is_feat=False, preact=False)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confs, preds = probabilities.max(1)
            for (conf, pred, label) in zip(confs, preds, labels):
                bin_index = int(((conf * 100) // (100/num_bins)).cpu())
                try:
                    if pred == label:
                        acc_counts[bin_index] += 1.0
                    conf_counts[bin_index] += conf.cpu()
                    counts[bin_index] += 1.0
                    overall_conf.append(conf.cpu())
                except:
                    print(bin_index, conf)
                    raise AssertionError('Bin index out of range!')


    avg_acc = [0 if count == 0 else acc_count / count for acc_count, count in zip(acc_counts, counts)]
    avg_conf = [0 if count == 0 else conf_count / count for conf_count, count in zip(conf_counts, counts)]
    ECE = 0
    for i in range(num_bins):
        ECE += (counts[i] / n) * abs(avg_acc[i] - avg_conf[i])

    return ECE, avg_acc, avg_conf, round(sum(acc_counts) / n, 6), counts

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# importing the dataset
mean = (0.4915, 0.4823, 0.4468)
std_dev = (0.2470, 0.2435, 0.2616)

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std_dev)
    ])

transform_original = transforms.Compose(
      [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std_dev)])

train_d = CIFAR10(root='./data/', train=True, transform=transform_original, download=True)
test_d = CIFAR10(root='./data/', train=False, transform=transform_test, download=True)

batch_size = 128
learning_rate = 0.05
momentum = 0.9
learning_rate_milestones = [50, 75, 100]
learning_gamma = 0.1
weight_decay = 5e-4

epochs = 120
NUM_BINS = 100

train_loader = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_d, shuffle=False, num_workers=8, batch_size=batch_size)

model = wrn_40_2(num_classes = 10)
model = model.to(device)

optimiser_simple = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler_simple = lr_scheduler.MultiStepLR(optimiser_simple, milestones=learning_rate_milestones, gamma=learning_gamma)

criterion_cls = nn.CrossEntropyLoss().to(device)

checkpoint = os.path.join('./weights/wrn40_2/sequentially')
if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)


# model training
best_acc = 0 
best_epoch = 0
best_ece = 0
state = {}
losses = AverageMeter()

n_holes = 1
length = 16
alpha = 0.3
beta = 1

# training on no augmented images if (r >= 0.9)
# then training on mixup images if (r >= 0 and r < 0.3)
# then training on cutout images if (r >= 0.3 and r < 0.6)
# then training on cutmix images if (r >= 0.6 and r < 0.9)

prob_mu = 0.3
prob_co = 0.6
prob_cm = 0.9

for epoch in range(epochs):
    model.train()

    r = np.random.rand(1)

    if (r >= 0.9):
        progress = tqdm(enumerate(train_loader), desc="Epoch: {}".format(epoch), total=len(train_loader))
        for iter, data in progress:
            inputs, targets = data[0].to(device), data[1].to(device)
            optimiser_simple.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = criterion_cls(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            loss.backward()
            optimiser_simple.step()

            progress.update(1)

    # applying mixup
    elif (r >= 0 and r < prob_mu):
        progress = tqdm(enumerate(train_loader), desc="Epoch: {}".format(epoch), total=len(train_loader))
        for iter, data in progress:
            inputs, targets = data[0].to(device), data[1].to(device)
            optimiser_simple.zero_grad()

            lam_mu = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            mixed_x = lam_mu * inputs + (1 - lam_mu) * inputs[rand_index, :]
            target_a_mu = targets
            target_b_mu = targets[rand_index]
            inputs = mixed_x
            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = mixup_criterion(criterion_cls, outputs, target_a_mu, target_b_mu, lam_mu)
        
            losses.update(loss.item(), inputs.size(0))
            loss.backward()
            optimiser_simple.step()

            progress.update(1)

    # applying cutout
    elif (r >= 0.3 and r < prob_co):
        progress = tqdm(enumerate(train_loader), desc="Epoch: {}".format(epoch), total=len(train_loader))
        for iter, data in progress:
            inputs, targets = data[0].to(device), data[1].to(device)
            optimiser_simple.zero_grad()

            black_cut = torch.zeros(inputs.shape).to(device)
            h = inputs.size(2)
            w = inputs.size(3)

            for n in range(n_holes):   # n_holes = 1, length = 16
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                
                inputs[:, :, y1:y2, x1:x2] = black_cut[:, :, y1:y2, x1:x2]

            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = criterion_cls(outputs, targets)
        
            losses.update(loss.item(), inputs.size(0))
            loss.backward()
            optimiser_simple.step()

            progress.update(1)
    
    # applying cutmix
    elif (r >= 0.6 and r < prob_cm):
        progress = tqdm(enumerate(train_loader), desc="Epoch: {}".format(epoch), total=len(train_loader))
        for iter, data in progress:
            inputs, targets = data[0].to(device), data[1].to(device)
            optimiser_simple.zero_grad()
            
            lam_cm = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            target_a_cm = targets
            target_b_cm = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam_cm)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam_cm = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            
            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = mixup_criterion(criterion_cls, outputs, target_a_cm, target_b_cm, lam_cm)
    
            losses.update(loss.item(), inputs.size(0))
            loss.backward()
            optimiser_simple.step()

            progress.update(1)
    
    scheduler_simple.step()

    print("\nTrain_loss : ",losses.avg)
    ece, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(num_bins=NUM_BINS, net=model, loader=test_loader, device=device)
    
    if (acc > best_acc):
        best_acc = acc
        best_epoch = epoch
        best_ece = ece
        state = {
            'state_dict': model.state_dict(),
            'acc': best_acc,
            'ece': ece,
            'epoch': best_epoch,
            'rng_state': torch.get_rng_state() 
        }
        save_checkpoint(state, path=checkpoint)

    print('Accuracy: {}'.format(acc))
    print('ECE: {}'.format(ece))
    print(f'Best Accuracy till now : {best_acc} at epoch {best_epoch}\n')

print("Best Accuracy achieved --> ", best_acc, end=" ")
print("at epoch --> ", best_epoch)
print("ece achieved --> ", best_ece)