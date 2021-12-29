import argparse
import os
import torch

from torch import nn
from torchvision import datasets, transforms
from torch.nn.utils import prune

from models.YourNet import YourNet
from eval.metrics import get_accuracy

# --------------- Arguments ---------------

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-dir', type=str, required=True)
parser.add_argument('--last-checkpoint', type=str, default=None)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, required=True)

args = parser.parse_args()
local = 1
LEARNING_RATE = 0.15


# --------------- Loading ---------------
# def prune_and_retrain(model:nn.Module, train_loader, test_loader, optimizer, loss_fn):
#     parameters = (
#         (model.conv1, 'weight'),
#         (model.conv2, 'weight'),
#         (model.fc1, 'weight'),
#         (model.fc2, 'weight'),
#         (model.fc3, 'weight'),
#     )
#
#     prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=0.9)
#     for parameter in parameters:
#         prune.remove(parameter[0], parameter[1])
#
#     torch.save(model.state_dict(), args.checkpoint_dir + f'mid.pth')
#
#     named_parameters = list(model.named_parameters())
#     mask = dict()
#     for named_parameter in named_parameters:
#         name, parameter = named_parameter[0], named_parameter[1]
#         temp = torch.where(parameter == 0, torch.zeros(parameter.size()).to(parameter.device),
#                            torch.ones(parameter.size()).to(parameter.device)).to(parameter.device)
#         mask.update({name: temp})
#
#     for epoch in range(args.epoch_start, args.epoch_end):
#         print(f"Epoch {epoch}\n-------------------------------")
#         size = len(train_loader.dataset)
#         model.train()
#         for batch_idx, (X, y) in enumerate(train_loader):
#
#             X, y = X.to(args.device), y.to(args.device)
#
#             # Compute prediction error
#             pred_y = model(X)
#             loss = loss_fn(pred_y, y)
#
#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             for named_parameter in named_parameters:
#                 name, parameter = named_parameter[0], named_parameter[1]
#                 temp = mask[name]
#                 parameter.data = torch.mul(temp, parameter).data
#
#             if batch_idx % 100 == 0:
#                 loss, current = loss.item(), batch_idx * len(X)
#                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#         accuracy = get_accuracy(model, test_loader, args.device)
#         print("Accuracy: %.3f" % accuracy)
#
#         torch.save(model.state_dict(), args.checkpoint_dir + f'new_epoch-{epoch}.pth')


def train(model, train_loader, test_loader, optimizer, loss_fn):
    train_cnt = args.epoch_end - args.epoch_start
    cnt = 0
    for epoch in range(args.epoch_start, args.epoch_end):
        cnt += 1
        if cnt > train_cnt // 2:
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE / 3)
        print(f"Epoch {epoch}\n-------------------------------")
        size = len(train_loader.dataset)
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):

            X, y = X.to(args.device), y.to(args.device)

            # Compute prediction error
            pred_y = model(X)
            loss = loss_fn(pred_y, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch_idx % 100 == 0:
            #     loss, current = loss.item(), batch_idx * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        accuracy = get_accuracy(model, test_loader, args.device)
        print("Accuracy: %.3f" % accuracy)

        torch.save(model.state_dict(), args.checkpoint_dir + f'epoch-{epoch}.pth')


if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    model: nn.Module
    model = YourNet().to(device=args.device)

    if args.last_checkpoint is not None:
        model.load_state_dict(torch.load(args.last_checkpoint, map_location=args.device))

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train(model, train_loader, test_loader, optimizer, loss_fn)

    # modules_to_fuse = [['conv1', 'relu1'], ['conv2', 'relu2']]
    # torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # model.eval()
    # model_prepared = torch.quantization.prepare(model)
    # for _, data in enumerate(train_loader):
    #     model_prepared(data[0])
    # model = torch.quantization.convert(model_prepared)
    # print(model)

    # model.qconfig = torch.quantization.default_qconfig
    # model = torch.quantization.prepare(model)
    #
    # # Convert to quantized model
    # model = torch.quantization.convert(model)
    # print(model.state_dict())
    #
    # torch.save(model.state_dict(), args.checkpoint_dir + 'final.pth')
    # print('finish')
