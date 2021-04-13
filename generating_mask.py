import os 
import torch
import argparse
import numpy as np
from timm.models import create_model
from datasets import build_dataset
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, Dataset

import models 
from pruning_utils import *

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

        pruner.mask(sparse, scope)
        check_sparsity(model)


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
torch.save('deit_tiniy_random_init.pth')
#pretrained or not
if args.pretrained:
    print('loading pretrained weight')
    checkpoint = torch.hub.load_state_dict_from_url(
        args.pretrained, map_location='cpu', check_hash=True)
    model.load_state_dict(checkpoint['model'])

model.cuda()
check_sparsity(model) 
# Identity pruning
prune_model_identity(model)
pruner = SynFlow(masked_parameters(model))

prune_loop(model, None, pruner, loader, torch.device('cuda:0'), args.sparsity, scope='global', epochs=100, train_mode=True)

print('sparsity = {}'.format(args.sparsity))
check_sparsity(model) 
current_mask = extract_mask(model.state_dict())
torch.save(current_mask, args.save_file)



