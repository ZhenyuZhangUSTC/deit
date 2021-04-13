import os 
import copy 
import torch
import argparse
import numpy as np
import torch.nn as nn
from timm.models import create_model
from datasets import build_dataset
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, Dataset

import models
from pruning_utils import *
from layers import Conv2d, Linear

parser = argparse.ArgumentParser('Pruning DeiT', add_help=False)
# Pruning parameters
parser.add_argument('--sparsity', default=0.5, type=float)
parser.add_argument('--pretrained', default=None, type=str, help='init or trained')
parser.add_argument('--save_file', default=None, type=str, help='save file name')

parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
args = parser.parse_args()


def prune_loop(model, loss, pruner, dataloader, device, sparsity, scope, epochs, train_mode=False):

    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in range(epochs):
        pruner.score(model, loss, dataloader, device)

        sparse = sparsity**((epoch + 1) / epochs)
        print(sparse)
        pruner.mask(sparse, scope)

        current_mask = extract_mask(model.state_dict())
        check_sparsity_dict(current_mask)


def prune_conv_linear(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prune_conv_linear(model=module)

        if isinstance(module, nn.Linear):
            bias=True
            if module.bias == None:
                bias=False
            layer_new = Linear(module.in_features, module.out_features, bias)
            model._modules[name] = layer_new

        if isinstance(module, nn.Conv2d):
            layer_new = Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride)
            model._modules[name] = layer_new

    return model

number_examples = 100
data = torch.ones(number_examples, 3, args.input_size, args.input_size)
target = torch.ones(number_examples)
data_set = torch.utils.data.TensorDataset(data, target)
loader = torch.utils.data.DataLoader(data_set, 
    batch_size=1, shuffle=False, 
    num_workers=2, pin_memory=True)

print(f"Creating model: {args.model}")
model = create_model(
    args.model,
    pretrained=False,
    num_classes=1000,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    drop_block_rate=None,
)
torch.save(model.state_dict(), 'random_init.pt')
#pretrained or not
if args.pretrained:
    print('loading pretrained weight')
    checkpoint = torch.hub.load_state_dict_from_url(
        args.pretrained, map_location='cpu', check_hash=True)
    model.load_state_dict(checkpoint['model'])

save_state_dict = copy.deepcopy(model.state_dict())
prune_conv_linear(model)

for key in save_state_dict.keys():
    if not key in model.state_dict().keys():
        print('can not load key = {}'.format(key))

model.load_state_dict(save_state_dict, strict=False)
model.cuda()

pruner = SynFlow(masked_parameters(model))
prune_loop(model, None, pruner, loader, torch.device('cuda:0'), args.sparsity, scope='global', epochs=10, train_mode=True)
print('sparsity = {}'.format(args.sparsity))

current_mask = extract_mask(model.state_dict())
check_sparsity_dict(current_mask)

torch.save(current_mask, args.save_file)



