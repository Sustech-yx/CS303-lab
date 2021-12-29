import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import prune
from torch.quantization import *


class YourNet(nn.Module):
    def __init__(self):
        super(YourNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, (3, 3), stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 8, (3, 3), stride=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(8 * 5 * 5, 80)  # 4x4 image dimension
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = F.max_pool2d(self.relu1(self.conv1(x)), (2, 2))
        x = F.max_pool2d(self.relu2(self.conv2(x)), (2, 2))
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


'''
Acknowledge:
Net work pruning studying:  https://github.com/joe-papa/pytorch-book/blob/main/06_05_Pruning.ipynb
                            https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
'''


class FooBarPruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask


def foobar_unstructured(module, name):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    FooBarPruningMethod.apply(module, name)
    return module