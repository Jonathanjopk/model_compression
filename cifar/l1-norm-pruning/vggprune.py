import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg(dataset=args.dataset, depth=args.depth)
if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

acc = test(model)
cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]
# original cfg: 
#     [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

cfg_mask = [] # 每層有哪些 filters 要被 mask
layer_id = 0
for m in model.modules():
    if isinstance(m, nn.Conv2d): # dimension: [output, input, k, k]
        out_channels = m.weight.data.shape[0] # dimension 0 是 output_channels
        if out_channels == cfg[layer_id]: # 如果該層的 filters 數量跟剪枝設定相同，表示不剪枝可跳過
            cfg_mask.append(torch.ones(out_channels)) # 該層全部保留，給 1
            layer_id += 1
            continue
        weight_copy = m.weight.data.abs().clone() # 將 filters weights 取絕對值
        weight_copy = weight_copy.cpu().numpy()
        L1_norm = np.sum(weight_copy, axis=(1, 2, 3)) # dimension 0 是 output_channels
        arg_max = np.argsort(L1_norm) # ascending sort
        arg_max_rev = arg_max[::-1][:cfg[layer_id]] # 1. reverse array 2. 取權重最大的前 cfg[layer_id] 個 filters position
        assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
        mask = torch.zeros(out_channels) # output_channels 預設是 masked 的
        mask[arg_max_rev.tolist()] = 1
        cfg_mask.append(mask)
        layer_id += 1
    elif isinstance(m, nn.MaxPool2d):
        layer_id += 1


newmodel = vgg(dataset=args.dataset, cfg=cfg)
if args.cuda:
    newmodel.cuda()

start_mask = torch.ones(3) # 初始 input_channel, 也就是 RGB
layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d): # 注意， batch norm 基本上都是
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) # 從 mask 中將值為 1 的 positions 全部找出來變成一個 array
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask
        if layer_id_in_cfg < len(cfg_mask):  # 如果還沒有到 cfg_mask 的最後，可以繼續 assign 下一個 mask list
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy()))) # input channel 根據 mask 中將值為 1 的 positions 全部找出來變成一個 array
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) # output channel mask 中將值為 1 的 positions 全部找出來變成一個 array
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone() # 先把 input 需要的 filters 選出來
        w1 = w1[idx1.tolist(), :, :, :].clone() # 再把 output 需要的 filters 選出來
        m1.weight.data = w1.clone() # 把裁減過後的 filter weights 放到新模型中
    elif isinstance(m0, nn.Linear):
        if layer_id_in_cfg == len(cfg_mask): # 第一次的 linear layer
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy()))) # 最後一層可能被裁減的 filter 層
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone() # [out_channel, in_channel]
            m1.bias.data = m0.bias.data.clone()
            layer_id_in_cfg += 1
            continue
        m1.weight.data = m0.weight.data.clone() # 第二次的 linear layer 就不會受到裁減影響了
        m1.bias.data = m0.bias.data.clone()
    elif isinstance(m0, nn.BatchNorm1d): # linear layer 後的 1 dimensional batch norm
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
print(newmodel)
model = newmodel
acc = test(model)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
with open(os.path.join(args.save, "prune.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")